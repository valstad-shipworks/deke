#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Crates the publish workflow never releases: deke-bench-retimers (publish = false)
# and deke-topp3tcp-nlp (git-only sleipnir dependency). Their versions may diverge.
EXCLUDE="deke-bench-retimers deke-topp3tcp-nlp"

package_version() {
    awk '
        /^\[/ { in_pkg = ($0 ~ /^\[package\]/) }
        in_pkg && /^[[:space:]]*version[[:space:]]*=/ {
            gsub(/.*=[[:space:]]*"/, ""); gsub(/".*/, ""); print; exit
        }
    ' "$1"
}

reference_major=""
reference_crate=""
mismatch=0

while IFS= read -r manifest; do
    crate="$(basename "$(dirname "$manifest")")"
    case " $EXCLUDE " in *" $crate "*) continue ;; esac

    version="$(package_version "$manifest")"
    if [ -z "$version" ]; then
        printf '%-22s no [package] version found\n' "$crate"
        mismatch=1
        continue
    fi

    major="${version%%.*}"
    printf '%-22s %s (major %s)\n' "$crate" "$version" "$major"

    if [ -z "$reference_major" ]; then
        reference_major="$major"
        reference_crate="$crate"
    elif [ "$major" != "$reference_major" ]; then
        mismatch=1
    fi
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -name Cargo.toml -not -path '*/target/*' | sort)

echo
if [ "$mismatch" -ne 0 ]; then
    echo "FAIL: crates do not all share major version $reference_major (baseline: $reference_crate)"
    exit 1
fi

echo "OK: every crate (except $EXCLUDE) is on major version $reference_major"
