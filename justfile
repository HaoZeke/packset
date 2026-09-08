root := justfile_directory()

test-py:
    .pixi/envs/default/bin/pytest -q scripts

# Both writers against one store, in both directions. The port is a swap and
# not a migration, which is a claim about the file on disk.
interop:
    cargo build -p packset-daemon --examples
    cd scripts && ../.pixi/envs/default/bin/python interop_check.py ../target/debug/examples

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
