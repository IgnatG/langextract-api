#!/usr/bin/env bash
# Extract structured data from inline text, then poll until done.
#
# Usage:
#   bash examples/curl/extract_text.sh

API_BASE="${API_BASE:-http://localhost:8000/api/v1}"

echo "==> Submitting extraction..."
resp=$(curl -s -X POST "${API_BASE}/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "AGREEMENT dated January 16, 2026 between Acme Corporation (Seller) and Beta LLC (Buyer) for the purchase of 1500 widgets at $25 each, totalling $12,500. Payment terms: net 30 days. Governed by Delaware law.",
    "provider": "gpt-4o",
    "passes": 3,
    "idempotency_key": "demo-text-001",
    "extraction_config": {
      "temperature": 0.2
    }
  }')

echo "$resp" | python -m json.tool

TASK_ID=$(echo "$resp" | python -c "import sys,json; print(json.load(sys.stdin)['task_id'])")
echo ""
echo "==> Polling task ${TASK_ID} ..."
bash "$(dirname "$0")/poll_status.sh" "$TASK_ID"
