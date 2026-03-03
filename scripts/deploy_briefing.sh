#!/bin/zsh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 flight_briefing_agent/flight_briefing.py

cp flight_briefing_agent/output/current_briefing.html docs/index.html

git add flight_briefing_agent/output/current_briefing.html \
        flight_briefing_agent/README.md \
        flight_briefing_agent/flight_briefing.py \
        docs/index.html \
        scripts/deploy_briefing.sh \
        .gitignore

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
git commit -m "Update briefing ${timestamp}" || echo "No changes to commit"

git push origin main
