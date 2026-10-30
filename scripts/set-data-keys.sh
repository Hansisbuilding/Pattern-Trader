#!/bin/sh
# Prompts for the free data keys (input hidden) and saves them to .env (git-ignored).
#   Alpha Vantage (earnings surprises): https://www.alphavantage.co/support/#api-key
#   FRED (Treasury yields, credit spreads): https://fredaccount.stlouisfed.org/apikeys
# Press Enter to skip either one.
cd "$(dirname "$0")/.." || exit 1
. scripts/env-lib.sh
printf "Alpha Vantage API key: "
stty -echo; read -r AV; stty echo; echo
printf "FRED API key: "
stty -echo; read -r FRED; stty echo; echo
[ -n "$AV" ] && set_env ALPHAVANTAGE_API_KEY "$AV" && echo "Alpha Vantage key saved."
[ -n "$FRED" ] && set_env FRED_API_KEY "$FRED" && echo "FRED key saved."
.venv/bin/python -m patterntrader.external --check
