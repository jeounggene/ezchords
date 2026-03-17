#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Refresh YouTube cookies on Render
#
# Usage:
#   ./refresh_cookies.sh
#
# Prerequisites (one-time setup):
#   1. Get your Render API key:  https://dashboard.render.com/account/api-keys
#   2. Get your service ID from the Render dashboard URL:
#      e.g. https://dashboard.render.com/web/srv-XXXXX  →  srv-XXXXX
#   3. Set them as env vars (add to ~/.zshrc for persistence):
#      export RENDER_API_KEY="rnd_xxxxxxxx"
#      export RENDER_SERVICE_ID="srv-xxxxxxxx"
#
# What it does:
#   - Exports fresh cookies from Chrome via yt-dlp
#   - Base64-encodes them
#   - Pushes to Render as the YT_COOKIES_B64 env var
#   - Triggers a redeploy
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Validate config ──────────────────────────────────────────────────────────
if [[ -z "${RENDER_API_KEY:-}" ]]; then
  echo "❌  RENDER_API_KEY not set. Get one at https://dashboard.render.com/account/api-keys"
  exit 1
fi
if [[ -z "${RENDER_SERVICE_ID:-}" ]]; then
  echo "❌  RENDER_SERVICE_ID not set. Find it in your Render dashboard URL (srv-XXXXX)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COOKIE_FILE="$SCRIPT_DIR/cookies.txt"

# ── Export fresh cookies from Chrome ──────────────────────────────────────────
echo "🍪  Exporting cookies from Chrome…"
yt-dlp --cookies-from-browser chrome \
       --cookies "$COOKIE_FILE" \
       --skip-download \
       --quiet \
       "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>/dev/null || true

if [[ ! -f "$COOKIE_FILE" ]]; then
  echo "❌  Failed to export cookies. Make sure Chrome is installed and you're logged into YouTube."
  exit 1
fi

# ── Base64-encode ─────────────────────────────────────────────────────────────
echo "📦  Encoding cookies…"
COOKIE_B64=$(base64 -i "$COOKIE_FILE")
echo "   → $(echo -n "$COOKIE_B64" | wc -c | tr -d ' ') chars"

# ── Push to Render ────────────────────────────────────────────────────────────
echo "🚀  Updating YT_COOKIES_B64 on Render…"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X PUT \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "[{\"key\": \"YT_COOKIES_B64\", \"value\": $(echo -n "$COOKIE_B64" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}]" \
  "https://api.render.com/v1/services/$RENDER_SERVICE_ID/env-vars")

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
  echo "✅  Cookies updated on Render (HTTP $HTTP_CODE)"
else
  echo "❌  Render API returned HTTP $HTTP_CODE"
  echo "   Check your RENDER_API_KEY and RENDER_SERVICE_ID"
  exit 1
fi

# ── Trigger redeploy ─────────────────────────────────────────────────────────
echo "🔄  Triggering redeploy…"
DEPLOY_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/$RENDER_SERVICE_ID/deploys")

if [[ "$DEPLOY_CODE" -ge 200 && "$DEPLOY_CODE" -lt 300 ]]; then
  echo "✅  Redeploy triggered!"
else
  echo "⚠️  Redeploy request returned HTTP $DEPLOY_CODE (env var was still updated)"
fi

echo ""
echo "Done! Your Render app will have fresh cookies after the deploy finishes."
