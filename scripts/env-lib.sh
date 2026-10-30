# Shared helper: set NAME=VALUE in .env without touching other lines (file stays private).
set_env() {
  umask 077
  touch .env
  grep -v "^$1=" .env > .env.tmp || true
  printf '%s=%s\n' "$1" "$2" >> .env.tmp
  mv .env.tmp .env
}
