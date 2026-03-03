#!/usr/bin/env python3
"""
Deterministic flight briefing tool.
Collects user inputs, fetches public weather data (METAR/TAF), stores pilot and
passenger profiles in sqlite, and generates the standardized HTML briefing that
can be reused for each request.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "flight_briefing.db"
TEMPLATE_PATH = BASE_DIR / "briefing_template.html"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "current_briefing.html"

# GitHub Pages deployment target (docs/ at repo root so Pages can serve the site)
REPO_ROOT = BASE_DIR.parent
GHPAGES_DIR = REPO_ROOT / "docs"
GHPAGES_DIR.mkdir(exist_ok=True)
GHPAGES_HTML = GHPAGES_DIR / "index.html"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
AIRPORTS_CSV = DATA_DIR / "airports.csv"
AIRPORTS_URL = "https://ourairports.com/data/airports.csv"

API_TIMEOUT = 10
METAR_API = "https://aviationweather.gov/api/data/metar"
TAF_API = "https://aviationweather.gov/api/data/taf"

EMPTY_WEIGHT = 514.5  # kg
EMPTY_LEVER = 0.439  # m
FUEL_START_ROLL = 3  # L per engine start
FUEL_CLIMB = 5  # L for climb to 4000 ft
FUEL_CRUISE_PER_HOUR = 22  # L/h at 65%
FUEL_RESERVE = 11  # L fixed reserve
FUEL_DENSITY = 0.72  # kg/L
CG_ENVELOPE = [
    (560, 240),
    (560, 290),
    (750, 390),
    (750, 320),
]
CG_LINE_MIN = 0.427
CG_LINE_MAX = 0.515

_airport_cache: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class Passenger:
    name: str
    weight: float
    height: float


@dataclass
class RunwayInfo:
    runway: str
    surface: str
    tora: float
    lda: float


@dataclass
class AirportSegment:
    icao: str
    name: str
    elevation_ft: float
    runways: List[RunwayInfo]


@dataclass
class WindInfo:
    direction: Optional[float]
    speed_kt: Optional[float]
    gust_kt: Optional[float]
    variable: bool
    var_from: Optional[float]
    var_to: Optional[float]


@dataclass
class FlightInputs:
    departure_icao: str
    airport_name: str
    airport_elevation_ft: float
    runways: List[RunwayInfo]
    airports: List[AirportSegment]
    flight_type: str
    estimated_time_hours: float
    stopovers: List[str]
    passengers: List[Passenger]
    baggage_weight: float
    pilot_name: str
    pilot_weight: float
    pilot_height: float


@dataclass
class WeatherData:
    metars: List[Dict[str, Any]]
    taf: List[Dict[str, Any]]
    lowest_qnh: Optional[float]
    oat_c: Optional[float]
    wind: Optional[WindInfo]
    weather_summary: str
    nearby_airports: List[Dict[str, Any]]
    metar_entries: List[Dict[str, Any]]


@dataclass
class BriefingData:
    inputs: FlightInputs
    weather: WeatherData
    fuel: Dict[str, Any]
    mass_balance: Dict[str, Any]
    performance: Dict[str, Any]
    warnings: List[str]
    airport_sections: List[Dict[str, Any]]
    generation_time: str


def prompt(text: str, validator=None) -> str:
    while True:
        value = input(text).strip()
        if not value:
            print("Value required.")
            continue
        if validator:
            try:
                validator(value)
            except ValueError as exc:
                print(exc)
                continue
        return value


def ensure_template() -> None:
    TEMPLATE_PATH.write_text(DEFAULT_TEMPLATE, encoding="utf-8")


def ensure_airports_data() -> None:
    if AIRPORTS_CSV.exists():
        return
    resp = requests.get(AIRPORTS_URL, timeout=API_TIMEOUT)
    resp.raise_for_status()
    AIRPORTS_CSV.write_bytes(resp.content)


def load_airports() -> Dict[str, Dict[str, Any]]:
    global _airport_cache
    if _airport_cache is not None:
        return _airport_cache
    ensure_airports_data()
    airports: Dict[str, Dict[str, Any]] = {}
    with AIRPORTS_CSV.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ident = (row.get("ident") or "").strip().upper()
            if len(ident) != 4 or not ident.isalpha():
                continue
            try:
                lat = float(row.get("latitude_deg") or 0.0)
                lon = float(row.get("longitude_deg") or 0.0)
            except ValueError:
                continue
            airports[ident] = {
                "ident": ident,
                "name": row.get("name", "").strip(),
                "lat": lat,
                "lon": lon,
                "elevation_ft": row.get("elevation_ft"),
            }
    _airport_cache = airports
    return airports


def get_airport(icao: str) -> Optional[Dict[str, Any]]:
    return load_airports().get(icao.upper())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def find_nearby_airports(icao: str, radius_km: float = 50.0) -> List[Dict[str, Any]]:
    airports = load_airports()
    home = airports.get(icao.upper())
    if not home:
        return [{"ident": icao.upper(), "name": "", "distance_km": 0.0}]
    nearby: List[Dict[str, Any]] = []
    for info in airports.values():
        lat = info.get("lat")
        lon = info.get("lon")
        if lat is None or lon is None:
            continue
        distance = haversine_km(home["lat"], home["lon"], lat, lon)
        if distance <= radius_km:
            nearby.append({
                "ident": info["ident"],
                "name": info.get("name", ""),
                "distance_km": round(distance, 1),
            })
    nearby.sort(key=lambda item: item["distance_km"])
    return nearby


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_metars_for_ids(icaos: List[str]) -> List[Dict[str, Any]]:
    if not icaos:
        return []
    station_list = sorted(set(filter(None, (code.strip().upper() for code in icaos))))
    if not station_list:
        return []
    params = {"ids": ",".join(station_list), "format": "json"}
    print(f"METAR request: {METAR_API} params={params}")
    resp = requests.get(METAR_API, params=params, timeout=API_TIMEOUT)
    print(f"METAR status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    print(f"METAR payload type: {type(data)}")
    if isinstance(data, dict):
        return data.get('data', [])
    if isinstance(data, list):
        return data
    return []


def fetch_tafs_for_ids(icaos: List[str]) -> List[Dict[str, Any]]:
    if not icaos:
        return []
    station_list = sorted(set(filter(None, (code.strip().upper() for code in icaos))))
    if not station_list:
        return []
    params = {"ids": ",".join(station_list), "format": "json"}
    print(f"TAF request: {TAF_API} params={params}")
    resp = requests.get(TAF_API, params=params, timeout=API_TIMEOUT)
    print(f"TAF status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    print(f"TAF payload type: {type(data)}")
    if isinstance(data, dict):
        return data.get('data', [])
    if isinstance(data, list):
        return data
    return []


QNH_MIN_HPA = 850
QNH_MAX_HPA = 1100

def parse_temperature_from_raw(raw: str) -> Optional[float]:
    if not raw:
        return None
    match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", raw)
    if not match:
        return None
    token = match.group(1)
    negative = token.startswith("M")
    value = int(token[1:] if negative else token)
    return float(-value if negative else value)

def parse_qnh_entries(raw: str) -> List[Dict[str, Optional[float]]]:
    if not raw:
        return []
    entries: List[Dict[str, Optional[float]]] = []
    temp = parse_temperature_from_raw(raw)
    for match in re.findall(r"\bQ(\d{4})\b", raw):
        hpa = float(match)
        if QNH_MIN_HPA <= hpa <= QNH_MAX_HPA:
            entries.append({"qnh": hpa, "temp": temp, "raw": raw})
    for match in re.findall(r"\bA(\d{4})\b", raw):
        inhg = float(match) / 100.0
        hpa = inhg * 33.8639
        if QNH_MIN_HPA <= hpa <= QNH_MAX_HPA:
            entries.append({"qnh": round(hpa, 1), "temp": temp, "raw": raw})
    return entries

def parse_lowest_qnh_and_temp(metars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    entries: List[Dict[str, Optional[float]]] = []
    for metar in metars:
        raw = metar.get("rawOb") or metar.get("raw_text") or ""
        entries.extend(parse_qnh_entries(raw))
    if not entries:
        return None, None, None
    best = min(entries, key=lambda item: item["qnh"])
    return best["qnh"], best.get("temp"), best.get("raw")


def extract_station_id(metar: Dict[str, Any]) -> str:
    return (
        metar.get("icaoId")
        or metar.get("stationId")
        or metar.get("station")
        or (metar.get("rawOb") or metar.get("raw_text") or "")[:4]
        or ""
    ).upper()


def extract_qnh_from_metar(metar: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    raw = metar.get("rawOb") or metar.get("raw_text") or ""
    entries = parse_qnh_entries(raw)
    if entries:
        primary = entries[0]
        return primary.get("qnh"), primary.get("temp"), raw
    return None, parse_temperature_from_raw(raw), raw if raw else None


def extract_oat(metars: List[Dict[str, Any]]) -> Optional[float]:
    for metar in metars:
        temp = metar.get("temp_c")
        if temp is not None:
            return float(temp)
    return None


def parse_wind_from_raw(raw: str) -> Optional[WindInfo]:
    if not raw:
        return None
    match = re.search(r"\b(?P<dir>\d{3}|VRB)(?P<speed>\d{2,3})(G(?P<gust>\d{2,3}))?KT\b", raw)
    if not match:
        return None
    dir_token = match.group("dir")
    direction = None if dir_token == "VRB" else float(dir_token)
    speed = float(match.group("speed"))
    gust_group = match.group("gust")
    gust = float(gust_group) if gust_group else None
    var_match = re.search(r"\b(?P<from>\d{3})V(?P<to>\d{3})\b", raw)
    var_from = float(var_match.group("from")) if var_match else None
    var_to = float(var_match.group("to")) if var_match else None
    variable = (dir_token == "VRB") or (var_from is not None and var_to is not None)
    return WindInfo(
        direction=direction,
        speed_kt=speed,
        gust_kt=gust,
        variable=variable,
        var_from=var_from,
        var_to=var_to,
    )


def summarize_weather(metars: List[Dict[str, Any]]) -> str:
    if not metars:
        return "No METAR data available."
    entries = []
    for metar in metars:
        raw = metar.get("rawOb") or metar.get("raw_text") or ""
        obs_time = metar.get("obsTime") or metar.get("obs_time")
        if obs_time:
            entries.append(f"{obs_time}: {raw}")
        else:
            entries.append(raw)
    return "\n".join(entries)


def build_metar_entries(metars: List[Dict[str, Any]], nearby: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    distance_lookup = {item["ident"]: item.get("distance_km") for item in nearby}
    entries = []
    for metar in metars:
        station = (
            metar.get("icaoId")
            or metar.get("stationId")
            or metar.get("station")
            or (metar.get("rawOb") or "")[:4]
            or ""
        ).upper()
        raw = metar.get("rawOb") or metar.get("raw_text") or ""
        obs_time = metar.get("obsTime") or metar.get("obs_time")
        entries.append(
            {
                "station": station,
                "distance_km": distance_lookup.get(station),
                "obs_time": obs_time,
                "raw": raw,
            }
        )
    return entries


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS passengers (
            name TEXT PRIMARY KEY,
            weight REAL NOT NULL,
            height REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pilot (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT NOT NULL,
            height REAL NOT NULL,
            weight REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS airport_profiles (
            icao TEXT PRIMARY KEY,
            elevation_ft REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS airport_runways (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icao TEXT NOT NULL,
            runway TEXT NOT NULL,
            surface TEXT NOT NULL,
            tora REAL NOT NULL,
            lda REAL NOT NULL,
            FOREIGN KEY (icao) REFERENCES airport_profiles(icao) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    return conn


def load_pilot(conn: sqlite3.Connection) -> Dict[str, float]:
    cur = conn.execute("SELECT name, height, weight FROM pilot WHERE id = 1")
    row = cur.fetchone()
    if row:
        name, height, weight = row
        print(f"Stored pilot profile: {name} ({height:.1f} cm, {weight:.1f} kg)")
        confirm = input("Use stored pilot data? (y/n): ").strip().lower()
        if confirm.startswith("y"):
            return {"name": name, "height": height, "weight": weight}
    name = prompt("Pilot name: ")
    height = float(prompt("Pilot height (cm): "))
    weight = float(prompt("Pilot weight (kg): "))
    conn.execute(
        "INSERT OR REPLACE INTO pilot (id, name, height, weight) VALUES (1, ?, ?, ?)",
        (name, height, weight),
    )
    conn.commit()
    return {"name": name, "height": height, "weight": weight}


def fetch_passenger(conn: sqlite3.Connection, name: str) -> Optional[Passenger]:
    cur = conn.execute("SELECT weight, height FROM passengers WHERE name = ?", (name,))
    row = cur.fetchone()
    if not row:
        return None
    weight, height = row
    confirm = input(
        f"Confirm using stored data for {name}: Weight {weight} kg Height {height} cm? (y/n): "
    ).strip().lower()
    if confirm.startswith("y"):
        return Passenger(name=name, weight=weight, height=height)
    return None


def store_passenger(conn: sqlite3.Connection, passenger: Passenger) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO passengers (name, weight, height) VALUES (?, ?, ?)",
        (passenger.name, passenger.weight, passenger.height),
    )
    conn.commit()


def collect_passengers(conn: sqlite3.Connection) -> List[Passenger]:
    passengers = []
    while True:
        name = input("Passenger name (leave empty to finish): ").strip()
        if not name:
            break
        stored = fetch_passenger(conn, name)
        if stored:
            passengers.append(stored)
            continue
        weight = float(prompt(f"Weight for {name} (kg): "))
        height = float(prompt(f"Height for {name} (cm): "))
        passenger = Passenger(name=name, weight=weight, height=height)
        store_passenger(conn, passenger)
        passengers.append(passenger)
    if not passengers:
        print("No passengers captured. Mass & balance will use pilot only.")
    return passengers


def summarize_runways(runways: List[RunwayInfo]) -> str:
    if not runways:
        return "No runway data stored."
    parts = []
    for rw in runways:
        parts.append(
            f"RWY {rw.runway} ({rw.surface}, TORA {rw.tora} m, LDA {rw.lda} m)"
        )
    return "; ".join(parts)


def load_airport_profile(conn: sqlite3.Connection, icao: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT elevation_ft FROM airport_profiles WHERE icao = ?",
        (icao.upper(),),
    )
    row = cur.fetchone()
    if not row:
        return None
    elevation_ft = row[0]
    cur = conn.execute(
        "SELECT runway, surface, tora, lda FROM airport_runways WHERE icao = ? ORDER BY runway",
        (icao.upper(),),
    )
    runways = [
        RunwayInfo(runway=r, surface=s, tora=float(t), lda=float(l))
        for r, s, t, l in cur.fetchall()
    ]
    return {
        "icao": icao.upper(),
        "elevation_ft": elevation_ft,
        "runways": runways,
    }


def store_airport_profile(
    conn: sqlite3.Connection,
    icao: str,
    elevation_ft: float,
    runways: List[RunwayInfo],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO airport_profiles (icao, elevation_ft) VALUES (?, ?)",
        (icao.upper(), elevation_ft),
    )
    conn.execute("DELETE FROM airport_runways WHERE icao = ?", (icao.upper(),))
    for rw in runways:
        conn.execute(
            "INSERT INTO airport_runways (icao, runway, surface, tora, lda) VALUES (?, ?, ?, ?, ?)",
            (icao.upper(), rw.runway, rw.surface, rw.tora, rw.lda),
        )
    conn.commit()


def collect_runway_entries() -> List[RunwayInfo]:
    runways: List[RunwayInfo] = []
    while True:
        runway_id = input("Runway designator (leave empty to finish): ").strip()
        if not runway_id:
            if runways:
                break
            print("At least one runway entry is required.")
            continue
        surface = input("Runway surface (asphalt/grass/etc.): ").strip() or "unknown"
        tora = float(prompt(f"TORA for runway {runway_id} (m): "))
        lda = float(prompt(f"LDA for runway {runway_id} (m): "))
        runways.append(RunwayInfo(runway=runway_id, surface=surface, tora=tora, lda=lda))
    return runways


def ensure_airport_profile(conn: sqlite3.Connection, icao: str) -> Dict[str, Any]:
    icao = icao.upper()
    existing = load_airport_profile(conn, icao)
    if existing:
        print(
            f"Stored airport data for {icao}: elevation {existing['elevation_ft']} ft; "
            f"{summarize_runways(existing['runways'])}"
        )
        use_stored = input("Use stored airport data? (y/n): ").strip().lower()
        if use_stored.startswith("y"):
            return existing
    print(f"Enter airport details for {icao}.")
    elevation_ft = float(prompt("Airport elevation (ft): "))
    runways = collect_runway_entries()
    store_airport_profile(conn, icao, elevation_ft, runways)
    return load_airport_profile(conn, icao) or {
        "icao": icao,
        "elevation_ft": elevation_ft,
        "runways": runways,
    }


def compute_fuel(flight_type: str, time_hours: float, stopovers: List[str]) -> Dict[str, Any]:
    legs = len(stopovers) + 1
    rolling_liters = FUEL_START_ROLL if flight_type in ("pattern", "local") else legs * FUEL_START_ROLL
    trip_liters = FUEL_CLIMB + (time_hours * FUEL_CRUISE_PER_HOUR)
    reserve_liters = FUEL_RESERVE
    total_liters = rolling_liters + trip_liters + reserve_liters
    fuel_mass = total_liters * FUEL_DENSITY
    return {
        "legs": legs,
        "rolling_liters": round(rolling_liters, 1),
        "rolling_label": f"{legs} x Leg" if legs == 1 else f"{legs} x Legs",
        "trip_liters": round(trip_liters, 1),
        "trip_label": f"{time_hours:.1f} h @ 22 L/h",
        "reserve_liters": round(reserve_liters, 1),
        "reserve_label": "11 L as in SOP",
        "total_liters": round(total_liters, 1),
        "fuel_mass": round(fuel_mass, 1),
    }


def lever_from_height(height_cm: float) -> float:
    return 0.484 if height_cm < 175 else 0.580


def build_mass_balance(inputs: FlightInputs, fuel: Dict[str, Any]) -> Dict[str, Any]:
    stations = []
    stations.append(
        {
            "label": "Empty Weight",
            "mass": EMPTY_WEIGHT,
            "lever": EMPTY_LEVER,
            "moment": EMPTY_WEIGHT * EMPTY_LEVER,
        }
    )
    pilot_lever = lever_from_height(inputs.pilot_height)
    stations.append(
        {
            "label": f"Pilot ({inputs.pilot_name})",
            "mass": inputs.pilot_weight,
            "lever": pilot_lever,
            "moment": inputs.pilot_weight * pilot_lever,
        }
    )
    for pax in inputs.passengers:
        lever = lever_from_height(pax.height)
        stations.append(
            {
                "label": f"Passenger ({pax.name})",
                "mass": pax.weight,
                "lever": lever,
                "moment": pax.weight * lever,
            }
        )
    if inputs.baggage_weight > 0:
        stations.append(
            {
                "label": "Baggage",
                "mass": inputs.baggage_weight,
                "lever": 1.3,
                "moment": inputs.baggage_weight * 1.3,
            }
        )
    fuel_mass = fuel["fuel_mass"]
    stations.append(
        {
            "label": "Fuel",
            "mass": fuel_mass,
            "lever": 0.325,
            "moment": fuel_mass * 0.325,
        }
    )
    total_mass = sum(s["mass"] for s in stations)
    total_moment = sum(s["moment"] for s in stations)
    cg = total_moment / total_mass
    inside_envelope = point_in_polygon((total_mass, total_moment), CG_ENVELOPE)
    line_position = (cg - CG_LINE_MIN) / (CG_LINE_MAX - CG_LINE_MIN)
    return {
        "stations": stations,
        "total_mass": round(total_mass, 1),
        "total_moment": round(total_moment, 1),
        "cg": round(cg, 3),
        "inside_envelope": inside_envelope,
        "line_position": line_position,
        "envelope": [{"mass": m, "moment": mom} for m, mom in CG_ENVELOPE],
    }


def point_in_polygon(point, polygon):
    x, y = point
    num = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(num + 1):
        px2, py2 = polygon[i % num]
        if min(py1, py2) < y <= max(py1, py2) and x <= max(px1, px2):
            if py1 != py2:
                xints = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-9) + px1
            else:
                xints = px1
            if px1 == px2 or x <= xints:
                inside = not inside
        px1, py1 = px2, py2
    return inside


def compute_performance(inputs: FlightInputs, weather: WeatherData) -> Dict[str, Any]:
    qnh = weather.lowest_qnh
    oat = weather.oat_c
    if qnh is None:
        pa = None
    else:
        pa = inputs.airport_elevation_ft + (1013 - qnh) * 30
    isa_temp = 15 - 2 * (pa / 1000) if pa is not None else None
    delta_isa = (oat - isa_temp) if (isa_temp is not None and oat is not None) else None
    if pa is not None and delta_isa is not None:
        da = pa + 120 * delta_isa
    else:
        da = None
    return {
        "pressure_altitude_ft": round(pa, 0) if pa is not None else None,
        "density_altitude_ft": round(da, 0) if da is not None else None,
        "isa_temp_c": round(isa_temp, 1) if isa_temp is not None else None,
        "delta_isa_c": round(delta_isa, 1) if delta_isa is not None else None,
    }


def collect_inputs(conn: sqlite3.Connection) -> FlightInputs:
    def icao_validator(code: str) -> None:
        if len(code.strip()) != 4:
            raise ValueError("ICAO must be 4 letters")

    departure = prompt("From which airport will the flight depart? (ICAO): ", icao_validator)
    airport_profile = ensure_airport_profile(conn, departure)
    airport_name = input("Departure airport name (optional): ").strip() or departure.upper()
    elevation_ft = float(airport_profile["elevation_ft"])
    runways = airport_profile.get("runways", [])
    segments: List[AirportSegment] = [
        AirportSegment(
            icao=departure.upper(),
            name=airport_name,
            elevation_ft=elevation_ft,
            runways=runways,
        )
    ]
    def flight_type_validator(ftype: str) -> None:
        if ftype.lower() not in {"pattern", "local", "cross-country"}:
            raise ValueError("Invalid type")
    flight_type = prompt("What type of flight? (pattern/local/cross-country): ", flight_type_validator).lower()
    if flight_type == "cross-country":
        total_time = float(prompt("Total estimated flight time (hours): "))
        stop_count = int(prompt("Number of stopovers: "))
        stopovers: List[str] = []
        for idx in range(stop_count):
            stop_icao = prompt(f"Stopover {idx + 1} ICAO: ", icao_validator).upper()
            stop_profile = ensure_airport_profile(conn, stop_icao)
            stop_name = input(f"Stopover {idx + 1} name (optional): ").strip() or stop_icao
            stop_elevation = float(stop_profile["elevation_ft"])
            stop_runways = stop_profile.get("runways", [])
            segments.append(
                AirportSegment(
                    icao=stop_icao,
                    name=stop_name,
                    elevation_ft=stop_elevation,
                    runways=stop_runways,
                )
            )
            stopovers.append(stop_icao)
    else:
        total_time = float(prompt("Estimated flight time (hours): "))
        stopovers = []
    passengers = collect_passengers(conn)
    baggage_weight = float(prompt("Baggage weight (kg, 0 if none): "))
    pilot = load_pilot(conn)
    return FlightInputs(
        departure_icao=departure.upper(),
        airport_name=airport_name,
        airport_elevation_ft=elevation_ft,
        runways=runways,
        airports=segments,
        flight_type=flight_type,
        estimated_time_hours=total_time,
        stopovers=stopovers,
        passengers=passengers,
        baggage_weight=baggage_weight,
        pilot_name=pilot["name"],
        pilot_weight=pilot["weight"],
        pilot_height=pilot["height"],
    )


def build_briefing(inputs: FlightInputs) -> BriefingData:
    warnings: List[str] = []

    segments = inputs.airports or [
        AirportSegment(
            icao=inputs.departure_icao.upper(),
            name=inputs.airport_name,
            elevation_ft=inputs.airport_elevation_ft,
            runways=inputs.runways,
        )
    ]

    neighbor_map: Dict[str, List[Dict[str, Any]]] = {}
    all_codes: List[str] = []

    for seg in segments:
        neighbors = find_nearby_airports(seg.icao)
        if not neighbors:
            neighbors = [{"ident": seg.icao.upper(), "name": seg.name, "distance_km": 0.0}]
        neighbor_map[seg.icao.upper()] = neighbors
        code = seg.icao.upper()
        if code not in all_codes:
            all_codes.append(code)
        for entry in neighbors:
            ident = (entry.get("ident") or "").strip().upper()
            if len(ident) == 4 and ident not in all_codes:
                all_codes.append(ident)
    if not all_codes:
        all_codes.append(inputs.departure_icao.upper())

    try:
        metars = fetch_metars_for_ids(all_codes)
    except Exception as exc:
        warnings.append(f"METAR fetch failed: {exc}")
        metars = []
    try:
        tafs = fetch_tafs_for_ids(all_codes)
    except Exception as exc:
        warnings.append(f"TAF fetch failed: {exc}")
        tafs = []

    metar_by_station = {extract_station_id(m): m for m in metars}
    taf_by_station = {extract_station_id(t): t for t in tafs}

    def build_segment_payload(seg: AirportSegment) -> Dict[str, Any]:
        neighbors = neighbor_map.get(seg.icao.upper(), [])
        metar_entries: List[Dict[str, Any]] = []
        metar_objects: List[Dict[str, Any]] = []
        seen_metar: set[str] = set()
        for neighbor in neighbors:
            station = (neighbor.get("ident") or "").strip().upper()
            if not station:
                continue
            metar = metar_by_station.get(station)
            if not metar:
                continue
            metar_objects.append(metar)
            metar_entries.append(
                {
                    "station": station,
                    "distance_km": neighbor.get("distance_km"),
                    "obs_time": metar.get("obsTime") or metar.get("obs_time"),
                    "raw": metar.get("rawOb") or metar.get("raw_text") or "",
                }
            )
            seen_metar.add(station)
        if seg.icao.upper() not in seen_metar:
            metar = metar_by_station.get(seg.icao.upper())
            if metar:
                metar_objects.insert(0, metar)
                metar_entries.insert(
                    0,
                    {
                        "station": seg.icao.upper(),
                        "distance_km": 0.0,
                        "obs_time": metar.get("obsTime") or metar.get("obs_time"),
                        "raw": metar.get("rawOb") or metar.get("raw_text") or "",
                    },
                )
        lowest_qnh, temp, _ = parse_lowest_qnh_and_temp(metar_objects)
        if temp is None:
            temp = extract_oat(metar_objects)
        weather_summary = "\n".join(entry["raw"] for entry in metar_entries if entry.get("raw"))
        primary_raw = metar_entries[0]["raw"] if metar_entries else None
        wind_info = parse_wind_from_raw(primary_raw or "") if primary_raw else None
        if wind_info is None:
            for metar in metar_objects:
                raw = metar.get("rawOb") or metar.get("raw_text") or ""
                wind_info = parse_wind_from_raw(raw)
                if wind_info:
                    break
        taf_entries: List[Dict[str, Any]] = []
        taf_objects: List[Dict[str, Any]] = []
        seen_taf: set[str] = set()
        for neighbor in neighbors:
            station = (neighbor.get("ident") or "").strip().upper()
            if not station or station in seen_taf:
                continue
            taf = taf_by_station.get(station)
            if taf:
                taf_entries.append(
                    {
                        "station": station,
                        "issue": taf.get("issueTime") or taf.get("issue_time") or taf.get("bulletinTime"),
                        "raw": taf.get("rawTAF") or taf.get("raw_text") or "",
                    }
                )
                taf_objects.append(taf)
                seen_taf.add(station)
        if seg.icao.upper() not in seen_taf:
            taf = taf_by_station.get(seg.icao.upper())
            if taf:
                taf_entries.insert(
                    0,
                    {
                        "station": seg.icao.upper(),
                        "issue": taf.get("issueTime") or taf.get("issue_time") or taf.get("bulletinTime"),
                        "raw": taf.get("rawTAF") or taf.get("raw_text") or "",
                    },
                )
                taf_objects.insert(0, taf)
        weather_payload = {
            "metar_entries": metar_entries,
            "taf_entries": taf_entries,
            "lowest_qnh": round(lowest_qnh, 1) if lowest_qnh is not None else None,
            "oat_c": round(temp, 1) if temp is not None else None,
            "wind": wind_info,
        }
        if metar_entries:
            weather_payload["primary_station"] = metar_entries[0]["station"]
            weather_payload["primary_distance_km"] = metar_entries[0].get("distance_km")
        return {
            "weather": weather_payload,
            "metars": metar_objects,
            "tafs": taf_objects,
            "weather_summary": weather_summary,
            "nearby_airports": neighbors,
        }

    segment_payloads: Dict[str, Dict[str, Any]] = {}
    airport_sections: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        payload = build_segment_payload(seg)
        segment_payloads[seg.icao.upper()] = payload
        role = "Departure" if idx == 0 else f"Stop {idx}"
        airport_sections.append(
            {
                "icao": seg.icao.upper(),
                "name": seg.name,
                "role": role,
                "elevation_ft": seg.elevation_ft,
                "runways": seg.runways,
                "weather": payload["weather"],
            }
        )

    primary_payload = segment_payloads[segments[0].icao.upper()]
    primary_weather = primary_payload["weather"]
    weather = WeatherData(
        metars=primary_payload["metars"],
        taf=primary_payload["tafs"],
        lowest_qnh=primary_weather["lowest_qnh"],
        oat_c=primary_weather["oat_c"],
        wind=primary_weather["wind"],
        weather_summary=primary_payload["weather_summary"],
        nearby_airports=primary_payload["nearby_airports"],
        metar_entries=primary_weather["metar_entries"],
    )
    if primary_weather["lowest_qnh"] is None:
        warnings.append("Reference QNH unavailable; performance needs manual input.")
    fuel = compute_fuel(inputs.flight_type, inputs.estimated_time_hours, inputs.stopovers)
    mass_balance = build_mass_balance(inputs, fuel)
    if not mass_balance["inside_envelope"]:
        warnings.append("Center of gravity outside envelope")
    performance = compute_performance(inputs, weather)
    if performance["pressure_altitude_ft"] is None:
        warnings.append("Pressure altitude missing")
    generation_time = datetime.now(timezone.utc).isoformat()
    return BriefingData(
        inputs=inputs,
        weather=weather,
        fuel=fuel,
        mass_balance=mass_balance,
        performance=performance,
        warnings=[w for w in warnings if w],
        airport_sections=airport_sections,
        generation_time=generation_time,
    )


def inject_into_template(data: BriefingData) -> None:
    ensure_template()
    payload = json.loads(json.dumps(asdict(data)))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__BRIEFING_DATA__", json.dumps(payload))
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    GHPAGES_HTML.write_text(html, encoding="utf-8")
    print("Briefing written to {output} and published to {gh}".format(output=OUTPUT_HTML, gh=GHPAGES_HTML))
    _deploy_to_github()


def _deploy_to_github() -> None:
    """Commit docs/index.html and push to origin main for GitHub Pages."""
    import subprocess, shutil
    if not shutil.which("git"):
        print("[deploy] git not found, skipping GitHub Pages deploy.")
        return
    repo = REPO_ROOT
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cmds = [
        ["git", "-C", str(repo), "add", "docs/index.html"],
        ["git", "-C", str(repo), "commit", "-m", f"briefing: auto-deploy {ts}"],
        ["git", "-C", str(repo), "push", "origin", "main"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # commit returns 1 if nothing to commit — that's fine
            if "nothing to commit" in result.stdout + result.stderr:
                print("[deploy] nothing new to commit.")
                return
            print(f"[deploy] warning: {' '.join(cmd[3:])} → {result.stderr.strip()}")
            return
    print(f"[deploy] ✓ Pushed to GitHub Pages — https://fradpnt.github.io/flight_briefing.oc/")


def main() -> None:
    ensure_template()
    conn = init_db()
    inputs = collect_inputs(conn)
    data = build_briefing(inputs)
    inject_into_template(data)
    print("Done. Open the HTML in a browser to review and capture screenshots.")


DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Flight Briefing</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; background:#f5f7fb; color:#0d1b2a; }
main { max-width: 1200px; margin: auto; padding: 2rem; }
section { background:#fff; margin-bottom:1.5rem; border-radius:12px; padding:1.25rem; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
h2 { margin-top:0; font-size:1.2rem; letter-spacing:0.05em; text-transform:uppercase; color:#415a77; }
h3 { margin-top:1rem; color:#1b263b; }
table { width:100%; border-collapse:collapse; }
th, td { padding:0.5rem; border-bottom:1px solid #e0e7ff; text-align:left; vertical-align:top; }
.warning { border-left:4px solid #d90429; background:#ffe8ec; padding:0.75rem 1rem; margin-bottom:0.75rem; border-radius:8px; }
.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.75rem; }
.badge-row a { text-decoration:none; color:#0d1b2a; background:#edf2fb; padding:0.35rem 0.9rem; border-radius:999px; font-size:0.85rem; border:1px solid #dbe4ff; }
.badge-row a:hover { background:#dbe4ff; }
.airport-list { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; margin-bottom:0.75rem; }
.airport-list span { background:#e9ecef; border-radius:999px; padding:0.3rem 0.8rem; font-size:0.85rem; color:#1b263b; border:1px solid #dee2e6; }
canvas { width:100%; max-width:none; height:auto; border:1px solid #dbe4ff; border-radius:8px; background:#fafcff; min-height:600px; }
.flex { display:flex; flex-direction:column; gap:1.5rem; }
.flex > div { width:100%; }
.airport-weather { border:1px solid #dbe4ff; border-radius:12px; padding:1rem; margin-bottom:1rem; background:#fdfcff; }
.airport-weather h4 { margin:0 0 0.5rem; color:#1b263b; }
.weather-flex { display:flex; gap:1rem; flex-wrap:wrap; }
.weather-flex > div { flex:1; min-width:260px; }
.runway-block { border:1px solid #e0e7ff; border-radius:12px; padding:1rem; margin-top:1rem; background:#ffffff; }
.runway-block h4 { margin:0 0 0.5rem; }
.metar-table code { font-family:"SFMono-Regular", Menlo, monospace; white-space:pre-wrap; word-break:break-word; display:block; }
.metar-table th:nth-child(4), .metar-table td:nth-child(4) { width:55%; }
.total-row th, .total-row td { font-weight:700; }
.weather-info { margin-top:0.75rem; font-weight:600; }
.runway-table th, .runway-table td { border-bottom:1px solid #e0e7ff; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Flight Briefing</h1>
    <div id="meta"></div>
  </header>
  <div id="warnings"></div>
  <section>
    <h2>1 — Maintenance</h2>
    <p>Review aircraft maintenance status in logbook.</p>
  </section>
  <section>
    <h2>2 — Weather</h2>
    <div id="weather-sections"></div>
    <div class="badge-row">
      <a href="https://www.flugwetter.de/fw/gafor/index.htm" target="_blank" rel="noreferrer noopener">GAFOR</a>
      <a href="https://www.flugwetter.de/fw/warn/index.htm" target="_blank" rel="noreferrer noopener">SIGMET</a>
      <a href="https://www.flugwetter.de/fw/chartsga/skyview/index.htm" target="_blank" rel="noreferrer noopener">Surface Weather Chart</a>
      <a href="https://www.flugwetter.de/fw/bilder/sat/index.htm?type=ir_rgb_eu" target="_blank" rel="noreferrer noopener">Satellite Europe</a>
      <a href="https://www.flugwetter.de/fw/bilder/rad/index.htm?type=rx" target="_blank" rel="noreferrer noopener">Radar DE</a>
      <a href="https://www.flugwetter.de/fw/bilder/rad/index.htm?type=eu" target="_blank" rel="noreferrer noopener">Radar EU</a>
      <a href="https://apps.apple.com/app/windy-wind-weather-forecast/id1094311790" target="_blank" rel="noreferrer noopener">Windy (iOS)</a>
    </div>
  </section>
  <section>
    <h2>3 — NOTAMs</h2>
    <p>Check NOTAMs for departure, destination and alternates.</p>
  </section>
  <section>
    <h2>4 — Fuel Planning</h2>
    <div id="fuel-table"></div>
  </section>
  <section>
    <h2>5 — Mass & Balance</h2>
    <div class="flex">
      <div>
        <div id="mass-table"></div>
      </div>
      <div>
        <canvas id="cg-chart" width="1500" height="1000"></canvas>
      </div>
    </div>
  </section>
  <section>
    <h2>6 — Performance</h2>
    <div id="performance"></div>
    <p>Reminders: verify takeoff and landing distances.</p>
  </section>
  <section>
    <h2>7 — Charts</h2>
    <p>Review charts for departure, destination and alternates.</p>
  </section>
</main>
<script>
const data = __BRIEFING_DATA__;
const airportSections = data.airport_sections || [];
const qs = s => document.querySelector(s);
const toNumber = val => (val === null || val === undefined ? null : Number(val));
const normalizeDegrees = deg => {
  const num = toNumber(deg);
  if (num === null || Number.isNaN(num)) return null;
  let normalized = num % 360;
  if (normalized < 0) normalized += 360;
  return normalized;
};
const angleDiff = (a, b) => {
  const na = normalizeDegrees(a);
  const nb = normalizeDegrees(b);
  if (na === null || nb === null) return null;
  let diff = Math.abs(na - nb);
  return diff > 180 ? 360 - diff : diff;
};
const runwayHeading = runwayId => {
  if (!runwayId) return null;
  const match = runwayId.match(/(\\d{2})/);
  if (!match) return null;
  let value = parseInt(match[1], 10);
  if (Number.isNaN(value)) return null;
  if (value === 36) value = 0;
  return value * 10;
};
const runwayNumber = runwayId => {
  if (!runwayId) return null;
  const match = runwayId.match(/(\\d{2})/);
  if (!match) return null;
  let value = parseInt(match[1], 10);
  if (Number.isNaN(value)) return null;
  if (value === 0) value = 36;
  return value;
};
const runwayPairLabel = runwayId => {
  const number = runwayNumber(runwayId);
  if (number === null) return runwayId || 'RWY';
  let reciprocal = number + 18;
  if (reciprocal > 36) reciprocal -= 36;
  const parts = [number, reciprocal].sort((a, b) => a - b).map(n => String(n).padStart(2, '0'));
  return `${parts[0]}/${parts[1]}`;
};
const groupRunwaysByPair = runways => {
  const map = new Map();
  runways.forEach(rw => {
    const label = runwayPairLabel(rw.runway);
    const surfaceKey = (rw.surface || 'unknown').toLowerCase();
    const key = `${label}__${surfaceKey}`;
    if (!map.has(key)) {
      map.set(key, { pairLabel: label, surface: rw.surface || 'unknown', members: [] });
    }
    map.get(key).members.push(rw);
  });
  return Array.from(map.values()).map(entry => {
    entry.members.sort((a, b) => {
      const aNum = runwayNumber(a.runway) ?? 0;
      const bNum = runwayNumber(b.runway) ?? 0;
      return aNum - bNum;
    });
    return entry;
  }).sort((a, b) => {
    if (a.pairLabel === b.pairLabel) {
      return (a.surface || '').localeCompare(b.surface || '');
    }
    return a.pairLabel.localeCompare(b.pairLabel);
  });
};
const componentFor = (speed, heading, direction) => {
  const diff = angleDiff(heading, direction);
  if (diff === null) return null;
  return speed * Math.cos((diff * Math.PI) / 180);
};
const worstComponentForSpeed = (speed, heading, wind) => {
  const spd = toNumber(speed);
  if (spd === null || Number.isNaN(spd) || heading === null || !wind) return null;
  const dirs = [];
  const nominal = normalizeDegrees(wind.direction);
  if (nominal !== null) dirs.push(nominal);
  const varFrom = normalizeDegrees(wind.var_from);
  const varTo = normalizeDegrees(wind.var_to);
  if (varFrom !== null && varTo !== null) {
    dirs.push(varFrom, varTo);
  }
  if (!dirs.length) {
    return wind.variable ? -spd : null;
  }
  let worstAngle = -1;
  let worstComponent = null;
  dirs.forEach(dir => {
    const diff = angleDiff(heading, dir);
    if (diff === null) return;
    if (diff > worstAngle) {
      worstAngle = diff;
      worstComponent = componentFor(spd, heading, dir);
    }
  });
  return worstComponent;
};
const formatComponentText = (base, gust) => {
  if (base === null || base === undefined || Number.isNaN(Number(base))) return 'N/A';
  const baseNum = Number(base);
  const magnitude = Math.abs(baseNum);
  const baseLabel = magnitude < 0.1 ? 'Calm' : baseNum >= 0 ? `Head ${magnitude.toFixed(1)} kt` : `Tail ${magnitude.toFixed(1)} kt`;
  if (gust === null || gust === undefined || Number.isNaN(Number(gust))) {
    return baseLabel;
  }
  const gustNum = Number(gust);
  const gustMag = Math.abs(gustNum);
  const gustLabel = gustMag < 0.1 ? 'Calm' : gustNum >= 0 ? `Head ${gustMag.toFixed(1)} kt` : `Tail ${gustMag.toFixed(1)} kt`;
  return gustLabel === 'Calm' ? baseLabel : `${baseLabel} (G ${gustLabel})`;
};
const describeWind = wind => {
  if (!wind) return 'Wind: N/A';
  const baseSpeed = toNumber(wind.speed_kt);
  const gustSpeed = toNumber(wind.gust_kt);
  if (baseSpeed === null && gustSpeed === null) return 'Wind: N/A';
  const directionText = wind.direction === null || wind.direction === undefined
    ? 'VRB'
    : `${String(Math.round(toNumber(wind.direction) ?? 0)).padStart(3, '0')}°`;
  let text = `Wind: ${directionText}/`;
  text += baseSpeed !== null ? `${baseSpeed.toFixed(0)} kt` : '—';
  if (gustSpeed !== null) {
    text += ` (G${gustSpeed.toFixed(0)} kt)`;
  }
  if (wind.var_from != null && wind.var_to != null) {
    const fromTxt = String(Math.round(toNumber(wind.var_from) ?? 0)).padStart(3, '0');
    const toTxt = String(Math.round(toNumber(wind.var_to) ?? 0)).padStart(3, '0');
    text += ` ${fromTxt}°-${toTxt}°`;
  }
  return text;
};
const describeField = (members, field) => {
  const values = members.map(m => m[field]).filter(v => v !== undefined && v !== null);
  if (!values.length) return 'N/A';
  const unique = [...new Set(values.map(v => `${v}`))];
  if (unique.length === 1) {
    return `${unique[0]} m`;
  }
  return members.map(m => `${m.runway}: ${m[field] ?? 'N/A'} m`).join(' / ');
};
const describeComponents = (members, wind) => {
  if (!wind) return 'N/A';
  const baseWindSpeed = toNumber(wind.speed_kt);
  const gustWindSpeed = toNumber(wind.gust_kt);
  const rows = members.map(m => {
    const heading = runwayHeading(m.runway);
    const baseComponent = worstComponentForSpeed(baseWindSpeed, heading, wind);
    const gustComponent = gustWindSpeed !== null ? worstComponentForSpeed(gustWindSpeed, heading, wind) : null;
    return { runway: m.runway, text: formatComponentText(baseComponent, gustComponent) };
  });
  const unique = [...new Set(rows.map(r => r.text))];
  if (unique.length === 1) {
    return unique[0];
  }
  return rows.map(r => `${r.runway}: ${r.text}`).join(' / ');
};
const meta = `Departure: ${data.inputs.departure_icao} (${data.inputs.airport_name}) · Flight type: ${data.inputs.flight_type} · ETA: ${data.inputs.estimated_time_hours}h · Generated ${new Date(data.generation_time).toLocaleString()}`;
qs('#meta').textContent = meta;
const warnings = data.warnings || [];
const warnEl = qs('#warnings');
warnings.forEach(msg => {
  const div = document.createElement('div');
  div.className = 'warning';
  div.textContent = msg;
  warnEl.appendChild(div);
});
const weatherContainer = qs('#weather-sections');
if (!airportSections.length) {
  weatherContainer.innerHTML = '<div class="warning">No airport weather data available.</div>';
} else {
  weatherContainer.innerHTML = airportSections.map((section, index) => renderWeatherSection(section, index)).join('');
}
const fuel = data.fuel;
qs('#fuel-table').innerHTML = `
  <table>
    <tr><th>Rolling fuel (${fuel.rolling_label})</th><td>${fuel.rolling_liters} L</td></tr>
    <tr><th>Trip fuel (${fuel.trip_label})</th><td>${fuel.trip_liters} L</td></tr>
    <tr><th>Reserve (${fuel.reserve_label})</th><td>${fuel.reserve_liters} L</td></tr>
    <tr class="total-row"><th>Total fuel</th><td><strong>${fuel.total_liters} L</strong></td></tr>
  </table>`;
const mass = data.mass_balance;
let rows = mass.stations.map(st => `<tr><td>${st.label}</td><td>${st.mass.toFixed(1)}</td><td>${st.lever.toFixed(3)}</td><td>${st.moment.toFixed(1)}</td></tr>`).join('');
rows += `<tr><th>Total</th><th>${mass.total_mass}</th><th></th><th>${mass.total_moment}</th></tr>`;
qs('#mass-table').innerHTML = `<table><tr><th>Station</th><th>Mass (kg)</th><th>Lever (m)</th><th>Moment (kg·m)</th></tr>${rows}</table>`;
const cgInfo = document.createElement('p');
cgInfo.textContent = `CG: ${mass.cg.toFixed(3)} m`;
qs('#mass-table').appendChild(cgInfo);
const perf = data.performance;
const formatDelta = value => {
  if (value === null || value === undefined) return 'N/A';
  const num = Number(value);
  if (Number.isNaN(num)) return 'N/A';
  return `${num > 0 ? '+' : ''}${num.toFixed(1)}`;
};
const primaryWeather = data.weather || {};
const primaryWind = primaryWeather.wind || null;
const deltaIsa = formatDelta(perf.delta_isa_c);
const perfInfo = `QNH: ${primaryWeather.lowest_qnh ?? 'N/A'} hPa · Field elevation: ${data.inputs.airport_elevation_ft} ft · Temperature: ${primaryWeather.oat_c ?? 'N/A'} °C · ${describeWind(primaryWind)} · ΔISA: ${deltaIsa} °C`;
const perfTable = `<table>
  <tr><th>Pressure altitude (ft)</th><td>${perf.pressure_altitude_ft ?? 'N/A'}</td></tr>
  <tr><th>Density altitude (ft)</th><td>${perf.density_altitude_ft ?? 'N/A'}</td></tr>
</table>`;
const runwayHtml = airportSections.length
  ? airportSections.map((section, index) => renderRunwaySection(section, index)).join('')
  : '<div class="warning">No runway data stored.</div>';
qs('#performance').innerHTML = `<p>${perfInfo}</p>${perfTable}${runwayHtml}`;
function buildMetarTable(entries) {
  let rows = '<tr><th>Station</th><th>Distance (km)</th><th>Observed</th><th>METAR</th></tr>';
  entries.forEach(entry => {
    rows += `<tr><td>${entry.station || 'N/A'}</td><td>${entry.distance_km ?? '—'}</td><td>${entry.obs_time || 'N/A'}</td><td><code>${entry.raw || ''}</code></td></tr>`;
  });
  return `<table class="metar-table">${rows}</table>`;
}
function buildTafTable(entries) {
  let rows = '<tr><th>Station</th><th>Issued</th><th>TAF</th></tr>';
  entries.forEach(entry => {
    rows += `<tr><td>${entry.station || 'N/A'}</td><td>${entry.issue || 'N/A'}</td><td><code>${entry.raw || ''}</code></td></tr>`;
  });
  return `<table class="metar-table">${rows}</table>`;
}
function renderWeatherSection(section, index) {
  const roleLabel = section.role || (index === 0 ? 'Departure' : `Stop ${index}`);
  const metarEntries = (section.weather && section.weather.metar_entries) || [];
  const tafEntries = (section.weather && section.weather.taf_entries) || [];
  const qnhText = section.weather && section.weather.lowest_qnh !== null && section.weather.lowest_qnh !== undefined
    ? section.weather.lowest_qnh
    : 'N/A';
  const oatText = section.weather && section.weather.oat_c !== null && section.weather.oat_c !== undefined
    ? section.weather.oat_c
    : 'N/A';
  const metarTable = metarEntries.length ? buildMetarTable(metarEntries) : '<div class="warning">No METAR data available.</div>';
  const tafTable = tafEntries.length ? buildTafTable(tafEntries) : '<div class="warning">No TAF data available.</div>';
  return `
    <div class="airport-weather">
      <h4>${section.icao} — ${section.name} (${roleLabel})</h4>
      <p class="weather-info">QNH: ${qnhText} hPa · OAT: ${oatText} °C · ${describeWind(section.weather ? section.weather.wind : null)}</p>
      <div class="weather-flex">
        <div>${metarTable}</div>
        <div>${tafTable}</div>
      </div>
    </div>`;
}
function renderRunwaySection(section, index) {
  const runways = section.runways || [];
  const roleLabel = section.role || (index === 0 ? 'Departure' : `Stop ${index}`);
  if (!runways.length) {
    return `<div class="runway-block"><h4>${section.icao} — ${section.name} (${roleLabel})</h4><p>No runway data stored.</p></div>`;
  }
  const grouped = groupRunwaysByPair(runways);
  const wind = section.weather ? section.weather.wind : null;
  let runwayRows = '<tr><th>Runway</th><th>Surface</th><th>TORA (m)</th><th>LDA (m)</th><th>Head/Tailwind</th></tr>';
  grouped.forEach(entry => {
    runwayRows += `<tr><td>${entry.pairLabel}</td><td>${entry.surface}</td><td>${describeField(entry.members, 'tora')}</td><td>${describeField(entry.members, 'lda')}</td><td>${describeComponents(entry.members, wind)}</td></tr>`;
  });
  return `
    <div class="runway-block">
      <h4>${section.icao} — ${section.name} (${roleLabel})</h4>
      <p class="weather-info">${describeWind(wind)}</p>
      <table class="runway-table">${runwayRows}</table>
    </div>`;
}
const canvas = document.getElementById('cg-chart');
const ctx = canvas.getContext('2d');
const poly = data.mass_balance.envelope || [
  {mass:560,moment:240},{mass:560,moment:290},{mass:750,moment:390},{mass:750,moment:320}
];
const points = poly.concat([poly[0]]);
const massMin = 550;
const massMax = 760;
const momentMin = 230;
const momentMax = 420;
const margin = { left: 70, right: 40, top: 30, bottom: 60 };
const usableWidth = canvas.width - margin.left - margin.right;
const usableHeight = canvas.height - margin.top - margin.bottom;
const unitSize = Math.min(usableWidth / (momentMax - momentMin), usableHeight / (massMax - massMin));
const plotWidth = unitSize * (momentMax - momentMin);
const plotHeight = unitSize * (massMax - massMin);
const originX = margin.left + (usableWidth - plotWidth) / 2;
const originY = margin.top + plotHeight;
const scaleX = moment => originX + (moment - momentMin) * unitSize;
const scaleY = mass => originY - (mass - massMin) * unitSize;
ctx.clearRect(0,0,canvas.width,canvas.height);
ctx.strokeStyle = '#edf2fb';
ctx.lineWidth = 1;
for (let moment = momentMin; moment <= momentMax; moment += 10) {
  const x = scaleX(moment);
  ctx.beginPath();
  ctx.moveTo(x, originY);
  ctx.lineTo(x, originY - plotHeight);
  ctx.stroke();
}
for (let massValue = massMin; massValue <= massMax; massValue += 10) {
  const y = scaleY(massValue);
  ctx.beginPath();
  ctx.moveTo(originX, y);
  ctx.lineTo(originX + plotWidth, y);
  ctx.stroke();
}
ctx.strokeStyle = '#8d99ae';
ctx.lineWidth = 1.5;
ctx.strokeRect(originX, originY - plotHeight, plotWidth, plotHeight);
ctx.fillStyle = '#1b263b';
ctx.font = '16px system-ui';
const momentTicks = [];
for (let val = momentMin; val <= momentMax; val += 10) momentTicks.push(val);
const massTicks = [];
for (let val = massMin; val <= massMax; val += 10) massTicks.push(val);
momentTicks.forEach(val => {
  const x = scaleX(val);
  ctx.fillText(val.toString(), x - 18, originY + 28);
  ctx.fillText(val.toString(), x - 18, originY - plotHeight - 12);
});
massTicks.forEach(val => {
  const y = scaleY(val);
  ctx.fillText(val.toString(), originX - 60, y + 6);
  ctx.fillText(val.toString(), originX + plotWidth + 20, y + 6);
});
ctx.fillStyle = '#0d1b2a';
ctx.font = '20px system-ui';
ctx.fillText('Moment (kg·m)', canvas.width / 2 - 80, canvas.height - 20);
ctx.save();
ctx.translate(margin.left / 2, canvas.height / 2);
ctx.rotate(-Math.PI / 2);
ctx.fillText('Mass (kg)', 0, 0);
ctx.restore();
ctx.strokeStyle = '#415a77';
ctx.lineWidth = 2;
ctx.beginPath();
ctx.moveTo(scaleX(points[0].moment), scaleY(points[0].mass));
for (let i=1;i<points.length;i++) {
  ctx.lineTo(scaleX(points[i].moment), scaleY(points[i].mass));
}
ctx.closePath();
ctx.stroke();
ctx.fillStyle = 'rgba(65,90,119,0.15)';
ctx.fill();
ctx.fillStyle = mass.inside_envelope ? '#2b9348' : '#d90429';
const cgX = scaleX(mass.total_moment);
const cgY = scaleY(mass.total_mass);
ctx.beginPath();
ctx.arc(cgX, cgY, 6, 0, Math.PI*2);
ctx.fill();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
