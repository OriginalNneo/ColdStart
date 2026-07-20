#!/bin/bash
# Serial runner for the 8-track refinement campaign (fixes the 16-concurrent RAM thrash).
cd /home/natty/nat-playground/ColdStart
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: > scratch_serial.log
echo "START $(date +%H:%M:%S) availMB=$(free -m | awk '/^Mem:/{print $7}')" >> scratch_serial.log
for f in scratch_agent1_nbsvm.py scratch_agent2_prune.py scratch_agent3_blockweight.py \
         scratch_agent4_estimator.py scratch_agent5_stylo.py scratch_agent6_length.py \
         scratch_agent7_shipfeat.py scratch_agent8_external.py; do
  base="${f%.py}"
  echo "----- RUN $f $(date +%H:%M:%S) -----" >> scratch_serial.log
  .venv/bin/python "$f" > "${base}.log" 2>&1
  rc=$?
  # pull the verdict line(s) into the serial summary
  verdict=$(grep -E "BEST PASSING|NO CANDIDATE PASSES|wrote scratch|Traceback|Error" "${base}.log" | tail -3)
  echo "  rc=$rc $(date +%H:%M:%S) availMB=$(free -m | awk '/^Mem:/{print $7}')" >> scratch_serial.log
  echo "  verdict: ${verdict}" >> scratch_serial.log
done
echo "ALL_DONE $(date +%H:%M:%S)" >> scratch_serial.log