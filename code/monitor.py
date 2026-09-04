"""
Shared system-resource monitor, used both client-side (benchmark.py) and
server-side (rest_server.py / soap_server.py) so client and server cost
can be measured separately.

Reports two kinds of numbers:
  - avg/peak CPU% and memory (MB): background-thread sampling, coarse-grained.
  - cpu_seconds: exact user+sys CPU time between start() and stop(), taken as
    a single before/after delta so it stays accurate even for runs shorter
    than the sampling interval (e.g. the vanilla method finishes in ~8ms).
"""

import os
import threading
import time
import psutil


class SystemMonitor:
    def __init__(self, interval=0.3):
        self.interval = interval
        self._cpu_samples = []
        self._mem_samples = []
        self._running = False
        self._thread = None
        self._proc = psutil.Process(os.getpid())
        self._cpu_times_start = None
        self._cpu_times_end = None

    def start(self):
        self._cpu_samples.clear()
        self._mem_samples.clear()
        self._proc.cpu_percent(interval=None)  # prime psutil's internal counter
        self._cpu_times_start = self._proc.cpu_times()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._cpu_times_end = self._proc.cpu_times()

    def _run(self):
        while self._running:
            try:
                self._cpu_samples.append(self._proc.cpu_percent(interval=None))
                self._mem_samples.append(self._proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self):
        cpu = self._cpu_samples or [0]
        mem = self._mem_samples or [0]
        cpu_seconds = 0.0
        if self._cpu_times_start and self._cpu_times_end:
            cpu_seconds = (
                (self._cpu_times_end.user - self._cpu_times_start.user)
                + (self._cpu_times_end.system - self._cpu_times_start.system)
            )
        return {
            "avg_cpu_percent": round(sum(cpu) / len(cpu), 2),
            "peak_cpu_percent": round(max(cpu), 2),
            "avg_memory_mb": round(sum(mem) / len(mem), 2),
            "peak_memory_mb": round(max(mem), 2),
            "cpu_seconds": round(cpu_seconds, 4),
        }
