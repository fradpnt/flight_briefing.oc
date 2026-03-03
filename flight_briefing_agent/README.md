# Flight Briefing Agent Tool

Deterministic CLI that collects structured flight inputs, reuses pilot/passenger
profiles via SQLite, fetches public weather (METAR/TAF) data, runs fuel and
mass/balance calculations, and produces a reusable HTML briefing that can be
reopened for edits or screenshots.

## Requirements

- Python 3.11+
- `requests` library (`pip install requests`)

## Usage

```bash
cd flight_briefing_agent
python3 flight_briefing.py
```

### Flow

1. Departure airport (ICAO) + elevation
2. Flight type (pattern, local, cross-country)
3. Flight time and stopovers (cross-country only)
4. Passenger data (stored in `flight_briefing.db` with confirmation before reuse)
5. Baggage weight
6. Pilot confirmation or entry (stored once)

The script fetches METAR/TAF via the public AviationWeather.gov API. If weather
is unavailable, warnings are injected into the HTML.

## Output

- Reusable template: `briefing_template.html`
- Latest filled briefing: `output/current_briefing.html`

Open the output HTML in your browser to review, tweak fields, or capture a
screenshot for delivery.

## GitHub Pages Deployment

Running `python3 flight_briefing.py` now writes the latest briefing to `flight_briefing_agent/output/current_briefing.html` **and** mirrors the same HTML to the repository-level `docs/index.html`. Push `docs/` to GitHub and enable **Pages → Deploy from branch → main → /docs** to publish the always-latest briefing without needing Google Drive.
