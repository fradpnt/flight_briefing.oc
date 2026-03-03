# Flight Briefing Generator

Generates a printable HTML briefing with weather, fuel, mass & balance, and runway summaries. Running the CLI saves the artifact under `flight_briefing_agent/output/current_briefing.html` and mirrors it to `docs/index.html` so GitHub Pages can host the latest version.

## Prerequisites

- Python 3.11+
- `requests` (install via `pip install -r requirements.txt` if you create one, or `pip install requests`)

## Usage

```bash
cd flight_briefing_agent
python3 flight_briefing.py
```

The script is interactive: provide departure ICAO, flight profile, passengers, baggage, and it will fetch METAR/TAF data before writing the final HTML.

## GitHub Pages Deployment

1. Ensure `docs/index.html` exists (the CLI writes it automatically after each run).
2. Enable **Settings → Pages → Deploy from branch → main → /docs** in GitHub.
3. Each time you regenerate the briefing, commit and push the updated HTML. Pages will refresh automatically.

## Automation Helper

`scripts/deploy_briefing.sh` wraps the common workflow:

- Runs the CLI (for new inputs)
- Copies the resulting HTML into `docs/`
- Commits the relevant files
- Pushes to `origin main`

Use it whenever you need to update the public briefing:

```bash
./scripts/deploy_briefing.sh
```

(You will still answer the interactive prompts the CLI asks for.)
