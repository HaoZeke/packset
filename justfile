root := justfile_directory()

# Everything but the search projection, which needs its own toolchain.
check:
    cargo fmt --all --check
    cargo clippy --locked --workspace --exclude packset-milli --all-targets -- -D warnings
    cargo test --locked --workspace --exclude packset-milli --no-fail-fast

# What a pack costs as it grows, and the shape of the graph inside it.
bench sizes="":
    cargo run --release -p packset-daemon --example bench -- {{sizes}}

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
