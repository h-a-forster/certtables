#!/bin/bash
# Restart loop for run_julia.jl: a point whose pending marker stays unchanged past LIMIT seconds is
# killed; the restart records it as HANG and moves on. Stops after MAXHANG kills.
# Usage: run_julia_watch.sh points.tsv out.tsv log LIMIT MAXHANG
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"  # the study folder
export JULIA_DEPOT_PATH=$R/distributionsjl/depot JULIA_NUM_THREADS=1
J=$R/distributionsjl/jl/julia-1.13.3/bin/julia.exe
PTS=$1; OUT=$2; LOG=$3; LIMIT=$4; MAXHANG=$5; kills=0
while :; do
  "$J" --project=distributionsjl/proj --startup-file=no scripts/run_julia.jl "$PTS" "$OUT" >> "$LOG" 2>&1 &
  pid=$!; last=""; since=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    cur=$(cat "$OUT.pending" 2>/dev/null)
    now=$(date +%s)
    if [ "$cur" != "$last" ]; then last=$cur; since=$now; fi
    if [ $((now-since)) -gt $LIMIT ]; then
      echo "KILL after ${LIMIT}s: $cur" >> "$LOG"
      taskkill //F //IM julia.exe > /dev/null 2>&1; kills=$((kills+1)); break
    fi
  done
  wait $pid 2>/dev/null
  if grep -q '^finished' <(tail -1 "$LOG"); then echo DONE >> "$LOG"; break; fi
  if [ $kills -ge $MAXHANG ]; then echo "STOP: $kills hangs" >> "$LOG"; break; fi
done
