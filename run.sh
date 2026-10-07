#!/usr/bin/env bash
# يشغّل NEXS ويعيد تشغيله تلقائياً إذا توقف لأي سبب.
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
while true; do
  python3 -m nexs.app
  echo "NEXS stopped (exit $?), restarting in 5s..." >&2
  sleep 5
done
