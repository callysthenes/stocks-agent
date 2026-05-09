#!/usr/bin/env bash
# stocks-tunnel-start.sh
# Starts a Cloudflare Quick Tunnel to the dashboard (port 8502 on localhost).
# On startup, parses the assigned URL and sends it via Telegram.
# Designed to be run by a user systemd service.
set -euo pipefail

ENV_FILE="/home/pv/Projects/stocks_agent/.env"
DASHBOARD_PORT=8502

# Load vars from .env
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
fi

TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

send_telegram() {
    local text="$1"
    if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
        curl -s -X POST \
            "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=${text}" \
            --data-urlencode "parse_mode=Markdown" \
            -o /dev/null || true
    fi
}

echo "[tunnel] Starting Cloudflare Quick Tunnel → localhost:${DASHBOARD_PORT}"

# cloudflared prints the URL on stderr; capture both streams
cloudflared tunnel \
    --url "http://localhost:${DASHBOARD_PORT}" \
    --no-autoupdate \
    2>&1 | while IFS= read -r line; do
    echo "$line"
    if echo "$line" | grep -qE 'https://[a-z0-9-]+\.trycloudflare\.com'; then
        TUNNEL_URL=$(echo "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
        echo "[tunnel] Assigned URL: ${TUNNEL_URL}"
        send_telegram "📈 *StocksAgent Dashboard Online*

URL: ${TUNNEL_URL}

Accede con tu usuario o regístrate en la primera visita."
    fi
done
