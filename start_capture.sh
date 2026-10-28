#!/usr/bin/env bash
# Start the Melting Rainbow operator UI with the Vial Fill camera setup that
# was validated by Linewatch. The UI owns the camera, records its raw signal,
# and runs the rule + AI tests in one process.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_PYTHON="${SCRIPT_DIR}/venv/bin/python"
OPERATOR_APP="${SCRIPT_DIR}/melting-rainbow/operator_ui/operator_app.py"
AI_MODEL="${SCRIPT_DIR}/melting-rainbow/operator_ui/pair_model.pt"
OUTPUT_DIR="${SCRIPT_DIR}/results/operator_output"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0}"

if [[ ! -x "${APP_PYTHON}" ]]; then
    echo "Python environment not found: ${APP_PYTHON}" >&2
    echo "Create it with: python3 -m venv venv" >&2
    exit 1
fi

if [[ ! -e "${CAMERA_DEVICE}" ]]; then
    echo "Vial Fill camera not found: ${CAMERA_DEVICE}" >&2
    echo "Reconnect it or set CAMERA_DEVICE to its /dev/v4l/by-id path." >&2
    exit 1
fi

cd "${SCRIPT_DIR}"
exec "${APP_PYTHON}" "${OPERATOR_APP}" \
    --camera "${CAMERA_DEVICE}" \
    --batch-size 3 \
    --ai-model "${AI_MODEL}" \
    --out "${OUTPUT_DIR}" \
    --autostart \
    "$@"
