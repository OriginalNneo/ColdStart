#!/bin/bash
# Bounded-parallel runner (4-wide) for remaining tracks. 4x ~2.6GB peak ~= 10GB, safe vs 14GB.
# The earlier crash was 16-wide; 4-wide is the controlled speedup.
cd /home/natty/nat-playground/ColdStart
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=.venv/bin/python
MAXJOBS=4
: > scratch_parallel.log
echo "START $(date +%H:%M:%S) availMB=$(free -m | awk '/^Mem:/{print $7}')" >> scratch_parallel.log

run_one () {
  local f="$1"; local base="${f%.py}"
  echo "  BEGIN $f $(date +%H:%M:%S)" >> scratch_parallel.log
  $PY "$f" > "${base}.log" 2>&1
  local rc=$?
  local verdict=$(grep -E "BEST PASSING|NO CANDIDATE PASSES|wrote scratch|Traceback|Error" "${base}.log" | tail -2 | tr '\n' ' ')
  echo "  END   $f rc=$rc $(date +%H:%M:%S) availMB=$(free -m | awk '/^Mem:/{print $7}') :: ${verdict}" >> scratch_parallel.log
}

for f in scratch_agent2_prune.py scratch_agent3_blockweight.py scratch_agent4_estimator.py \
         scratch_agent5_stylo.py scratch_agent6_length.py scratch_agent7_shipfeat.py \
         scratch_agent8_external.py; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
  run_one "$f" &
done
wait
echo "ALL_DONE $(date +%H:%M:%S)" >> scratch_parallel.log