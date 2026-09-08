root := justfile_directory()

test-py:
    .pixi/envs/default/bin/pytest -q scripts

# Both writers against one store, in both directions. The port is a swap and
# not a migration, which is a claim about the file on disk.
interop:
    cargo build -p packset-daemon --examples
    cd scripts && ../.pixi/envs/default/bin/python interop_check.py ../target/debug/examples

# The same, with the search projection present, so the milli path is the one
# under test rather than the fallback.
surface-milli: milli
    cargo build -p packset-daemon
    cd scripts && PACKSET_MILLI="$PWD/../target/release/packset-milli" \
        ../.pixi/envs/default/bin/python surface_check.py ../target/debug/packsetd

# One request sequence against both writers, every status code and body
# compared. A port is finished when a client cannot tell which one answered.
surface:
    cargo build -p packset-daemon
    cd scripts && ../.pixi/envs/default/bin/python surface_check.py ../target/debug/packsetd

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
    # The compute nodes are named rgamNterra, so a bare `terra` prefix match
    # would refuse to build on the builder itself.
    case "$host" in
        terra|rg.terra|*.terra|*terra) ;;
        *)
            echo "just milli: build on the remote builder, not $host" >&2
            exit 1
            ;;
    esac
    cargo build -p packset-milli --release

ensure:
    bin/packset ensure
