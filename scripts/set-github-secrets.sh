#!/bin/sh
# Uploads the bot's keys to GitHub as encrypted repo secrets for the trading-day workflow.
# Prompts for each one in turn (input hidden). Press Enter to reuse the value already in .env,
# or type "-" to skip that secret. Values go straight from this terminal to GitHub; they are
# never printed or written anywhere else.
#   scripts/set-github-secrets.sh                          asks for every secret
#   scripts/set-github-secrets.sh CLAUDE_CODE_OAUTH_TOKEN  asks for just the ones named
cd "$(dirname "$0")/.." || exit 1
GH="${GH:-$(command -v gh || echo "$HOME/.local/bin/gh")}"
REPO="Hansisbuilding/Pattern-Trader"

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "First, sign in to GitHub (one time). A browser page will open."
  "$GH" auth login --hostname github.com --git-protocol https --web || exit 1
fi

from_env() {  # value of NAME in .env, without surrounding quotes
  [ -f .env ] || return 0
  sed -n "s/^$1=//p" .env | tail -n 1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

ask() {  # ask NAME "description"
  if [ -n "$ONLY" ]; then case " $ONLY " in *" $1 "*) ;; *) return ;; esac; fi
  current=$(from_env "$1")
  if [ -n "$current" ]; then hint="Enter = use the one in .env, - = skip"; else hint="- = skip"; fi
  printf "%s (%s)\n  [%s]: " "$1" "$2" "$hint"
  stty -echo
  if [ "$1" = CLAUDE_CODE_OAUTH_TOKEN ]; then
    # Claude prints this token across two or more lines, so a paste arrives as several lines:
    # read them all until an empty line (the extra Enter).
    value=""; while read -r line && [ -n "$line" ]; do value="$value$line"; done
  else
    read -r value
  fi
  stty echo; echo
  [ "$value" = "-" ] && { echo "  skipped"; return; }
  [ -z "$value" ] && value="$current"
  if [ -z "$value" ]; then echo "  skipped (nothing entered)"; return; fi
  case "$1" in
    ALPACA_API_KEY) case "$value" in PK*) ;; *) echo "  refused: paper keys start with PK"; return ;; esac ;;
    CLAUDE_CODE_OAUTH_TOKEN)
      value=$(printf %s "$value" | tr -d '[:space:]')   # a token copied across wrapped lines
      case "$value" in sk-ant-oat*) ;; *) echo "  refused: the token starts with sk-ant-oat"; return ;; esac
      echo "  checking the token with Claude..."
      reply=$(cd /tmp && CLAUDE_CODE_OAUTH_TOKEN="$value" claude -p "Reply with just OK" --max-turns 1 \
        --setting-sources "" --strict-mcp-config --tools "" --no-session-persistence 2>&1)
      case "$reply" in *OK*) echo "  token works" ;; *) echo "  refused: Claude rejected this token: $reply"; return ;; esac ;;
  esac
  printf %s "$value" | "$GH" secret set "$1" --repo "$REPO" >/dev/null && echo "  saved to GitHub"
}

ONLY="$*"
echo "Setting secrets for $REPO"
ask ALPACA_API_KEY "Alpaca PAPER API key, starts with PK"
ask ALPACA_SECRET_KEY "Alpaca paper secret key"
ask SEC_USER_AGENT "name and email for sec.gov, e.g. PatternTrader you@example.com"
ask ALPHAVANTAGE_API_KEY "optional, earnings surprises"
ask FRED_API_KEY "optional, Treasury yields and credit spreads"
ask CLAUDE_CODE_OAUTH_TOKEN "for the reviewer: paste ALL the lines of the token from 'claude setup-token', then press Enter twice"
echo
echo "Secrets now on GitHub (names only):"
"$GH" secret list --repo "$REPO"
