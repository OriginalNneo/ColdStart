#!/bin/bash
cd /home/natty/nat-playground/ColdStart
empty=0
for t in $(seq 1 240); do
  n=$(ps aux | grep "[s]cratch_agent" | grep python | wc -l)
  avail=$(free -m | awk '/^Mem:/{print $7}')
  if [ "$n" -eq 0 ]; then empty=$((empty+1)); else empty=0; fi
  # memory guard: if under 700MB available, kill the largest agent python
  if [ "$avail" -lt 700 ]; then
    big=$(ps -eo pid,rss,args --sort=-rss | grep "[s]cratch_agent" | grep python | head -1 | awk '{print $1}')
    [ -n "$big" ] && kill "$big" && echo "$(date +%H:%M:%S) MEMGUARD killed pid=$big (avail=${avail}MB)" >> scratch_monitor.log
  fi
  echo "$(date +%H:%M:%S) running=$n availMB=$avail" >> scratch_monitor.log
  if [ "$empty" -ge 3 ]; then echo "ALL_AGENT_PY_DONE" >> scratch_monitor.log; break; fi
  sleep 15
done
echo "=== FINAL LOG TAILS ===" >> scratch_monitor.log
for f in scratch_agent*.log; do echo "----- $f -----" >> scratch_monitor.log; tail -40 "$f" >> scratch_monitor.log 2>/dev/null; done
ls -la scratch_agent*_pred.csv >> scratch_monitor.log 2>/dev/null
echo "MONITOR_EXIT" >> scratch_monitor.log
