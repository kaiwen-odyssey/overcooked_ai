#!/usr/bin/env bash
set -euo pipefail

webui_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$webui_dir"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 22.13.1 or newer is required." >&2
  exit 1
fi

node -e '
const [major, minor] = process.versions.node.split(".").map(Number);
if (major < 22 || (major === 22 && minor < 13)) {
  console.error(`Node.js 22.13.1 or newer is required; found ${process.versions.node}.`);
  process.exit(1);
}
'

if [[ ! -x node_modules/.bin/vinext ]]; then
  npm ci
fi

npm run build

burger_webui_host="${BURGER_WEBUI_HOST:-0.0.0.0}"
burger_webui_port="${BURGER_WEBUI_PORT:-3000}"
echo "Burger WebUI: http://localhost:${burger_webui_port}"
exec env HOST="$burger_webui_host" PORT="$burger_webui_port" \
  node dist/standalone/server.js
