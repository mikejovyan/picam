#!/bin/sh
set -e

set -a
. "$(dirname "$0")/picam.env"
set +a

ssh "$PI_USER@$PI_HOST" '~/picam/.venv/bin/pip install -q ~/picam && ~/picam/.venv/bin/picam'
