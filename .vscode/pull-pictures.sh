#!/bin/sh
set -e

set -a
. "$(dirname "$0")/picam.env"
set +a

rsync -av --remove-source-files -e ssh \
	"$PI_USER@$PI_HOST:~/picam/pictures/" "$(dirname "$0")/../pictures/"
