#!/usr/bin/env bash
# Record a terminal demo (API smoke test with curl) as a GIF.
# Requires: asciinema, agg (https://github.com/asciinema/agg)
#
# For the browser UI (recommended for recruiters), use ScreenToGif, Kap, or
# LICEcap and save to assets/demo.gif — see README "Demo" section.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p assets

echo "Recording demo — press Ctrl+D when done..."
asciinema rec /tmp/eng-sp-translator-demo.cast

echo "Converting to GIF..."
agg /tmp/eng-sp-translator-demo.cast assets/demo.gif

echo "Demo saved to assets/demo.gif"
echo "Uncomment the image line in README under ## Demo"
