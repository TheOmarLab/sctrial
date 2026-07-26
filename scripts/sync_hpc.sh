#!/bin/bash
# Keep local and HPC in sync, in one direction at a time, with no stale bytecode.
#
#   scripts/sync_hpc.sh push      # local code -> HPC   (deletes removed files)
#   scripts/sync_hpc.sh pull      # HPC results -> local
#   scripts/sync_hpc.sh check     # report any drift without changing anything
#
# `push` uses --delete so a file deleted locally is deleted on the cluster. That
# is the point: a superseded module left behind on HPC is still importable there,
# and stale .pyc silently executes pre-fix code even after the .py is replaced.
set -euo pipefail

HOST=esplhpccompbio-lv01
REMOTE=/common/omarmlab/members/omar/projects/sctrial
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUSCRIPT_LOCAL="$(cd "$LOCAL/../.." && pwd)/manuscript"

EXCL=(--exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='.DS_Store')

case "${1:-check}" in
  deploy)
    # PROVENANCE-GRADE DEPLOY. `push` rsyncs files over whatever the cluster
    # happens to have checked out, so the remote git state stops describing what
    # actually runs -- measured: HEAD stuck at an old commit with 109 dirty files
    # while the running code was many commits newer. A manifest recording that
    # HEAD would have been worse than no manifest, because it would look precise.
    #
    # This checks out an exact commit instead, so the cluster tree IS that commit.
    # Use it before freezing and before the definitive run; `push` remains for
    # fast iteration.
    SHA="${2:-$(git -C "$LOCAL" rev-parse HEAD)}"
    echo "=== deploy $SHA -> $HOST:$REMOTE ==="
    if [ -n "$(git -C "$LOCAL" status --porcelain)" ]; then
      echo "ERROR: local tree is dirty; commit before deploying." >&2
      exit 1
    fi
    git -C "$LOCAL" push -q origin HEAD 2>/dev/null || true
    ssh "$HOST" "cd $REMOTE && \
      git fetch --all --quiet --tags && \
      git checkout --quiet --detach $SHA && \
      git reset --hard --quiet $SHA && \
      git clean -qfd src scripts tests && \
      find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
      find . -name '*.pyc' -delete 2>/dev/null; \
      micromamba run -n sctrial pip install -e . --no-deps --force-reinstall -q && \
      echo \"remote HEAD:  \$(git rev-parse --short HEAD)\" && \
      echo \"remote dirty: \$(git status --porcelain | wc -l) file(s)\" && \
      micromamba run -n sctrial python -c \"
import sctrial.benchmark as b, inspect, sys
assert 'simulator_v2' in inspect.getsource(b.orchestrator), 'HPC orchestrator is stale'
try:
    import sctrial.benchmark.simulator  # noqa
    sys.exit('FAIL: the deleted simulator is still importable on HPC')
except ModuleNotFoundError:
    pass
print('HPC package OK:', b.CORE_METHODS)\""
    ;;
  push)
    echo "=== push: $LOCAL -> $HOST:$REMOTE ==="
    for d in src scripts tests manuscript_figures; do
      [ -d "$LOCAL/$d" ] || continue
      rsync -az --delete "${EXCL[@]}" "$LOCAL/$d/" "$HOST:$REMOTE/$d/"
      echo "  synced $d"
    done
    # Purge bytecode and reinstall so the installed package matches the source.
    ssh "$HOST" "cd $REMOTE && \
      find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
      find . -name '*.pyc' -delete 2>/dev/null; \
      micromamba run -n sctrial pip install -e . --no-deps --force-reinstall -q && \
      micromamba run -n sctrial python -c \"
import sctrial.benchmark as b, inspect, sys
assert 'simulator_v2' in inspect.getsource(b.orchestrator), 'HPC orchestrator is stale'
try:
    import sctrial.benchmark.simulator  # noqa
    sys.exit('FAIL: the deleted simulator is still importable on HPC')
except ModuleNotFoundError:
    pass
print('HPC package OK:', b.CORE_METHODS)\""
    ;;
  pull)
    echo "=== pull: $HOST:$REMOTE/manuscript -> $MANUSCRIPT_LOCAL ==="
    mkdir -p "$MANUSCRIPT_LOCAL"
    rsync -az "${EXCL[@]}" "$HOST:$REMOTE/manuscript/benchmark/" \
        "$MANUSCRIPT_LOCAL/benchmark/"
    echo "  pulled manuscript/benchmark"
    ;;
  check)
    echo "=== drift check ==="
    for d in src scripts tests; do
      out=$(rsync -aznc --delete "${EXCL[@]}" "$LOCAL/$d/" "$HOST:$REMOTE/$d/" | grep -v '^$' || true)
      if [ -n "$out" ]; then
        echo "DRIFT in $d:"; echo "$out" | sed 's/^/    /'
      else
        echo "  $d in sync"
      fi
    done
    ;;
  *)
    echo "usage: $0 {push|pull|check}" >&2; exit 2 ;;
esac
