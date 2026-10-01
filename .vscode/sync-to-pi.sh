#!/bin/sh
set -e

set -a
. "$(dirname "$0")/picam.env"
set +a

rsync -av --delete --exclude='.*' --filter=':- .gitignore' -e ssh \
	"$(dirname "$0")/../" "$PI_USER@$PI_HOST:~/picam/"
