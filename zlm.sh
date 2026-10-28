#!/usr/bin/env sh
# Run the freshly built zlm with a project-local store.
#   ./zlm.sh <command> [args]
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export ZLM_HOME="$here/.zlm"
exec "$here/zig-out/bin/zlm" "$@"
