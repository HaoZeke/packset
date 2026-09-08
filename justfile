root := justfile_directory()

test-py:
    .pixi/envs/default/bin/pytest -q scripts

# Regenerate the Python goldens the Rust port is checked against. Read the
# diff: a change here is a change in what the daemon accepts.
goldens:
    cd scripts && ../.pixi/envs/default/bin/python gen_goldens.py

lint:
    .pixi/envs/default/bin/ruff check scripts

milli:
    #!/usr/bin/env bash
    set -euo pipefail
    host="$(hostname -s || hostname)"
    case "$host" in
        terra|rg.terra|*.terra) ;;
        *)
            echo "just milli: build on the remote builder, not $host" >&2
            exit 1
            ;;
    esac
    cargo build -p packset-milli --release

ensure:
    bin/packset ensure
