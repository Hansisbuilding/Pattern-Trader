#!/bin/sh
# Prompts for Alpaca PAPER keys (input hidden) and saves them to .env (git-ignored).
cd "$(dirname "$0")/.." || exit 1
. scripts/env-lib.sh
printf "Alpaca PAPER API key (starts with PK): "
stty -echo; read -r KEY; stty echo; echo
printf "Alpaca PAPER secret key: "
stty -echo; read -r SECRET; stty echo; echo
case "$KEY" in
  PK*) ;;
  *) echo "That isn't a paper key (paper keys start with PK). Nothing saved."; exit 1 ;;
esac
set_env ALPACA_API_KEY "$KEY"
set_env ALPACA_SECRET_KEY "$SECRET"
echo "Saved to .env (readable only by you, never committed)."
.venv/bin/patterntrader account
