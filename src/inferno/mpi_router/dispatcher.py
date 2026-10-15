"""
MPI dispatcher (rank 0): replays an open-loop request stream, routing each request to a replica rank
by least-outstanding-requests, and collects results asynchronously.

Point-to-point isend/iprobe only (no collectives). The dispatch loop and the result-drain loop share one
thread and interleave, so a slow result never delays a scheduled arrival. Both MPI hops are timestamped so
coordination overhead is measured directly (routing.overhead_us) rather than inferred by subtraction.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from ..metrics import RequestResult
from ..workload.generator import Request
from .routing import TAG_JOB, TAG_RESULT, LeastOutstanding, overhead_us

_POLL_SLEEP = 0.0002   # 200 us; documented because it bounds dispatcher-side timing resolution


class Dispatcher:
    def __init__(self, comm=None):
        from mpi4py import MPI

        self.MPI = MPI
        self.comm = comm or MPI.COMM_WORLD
        self.n_replicas = self.comm.Get_size() - 1     # rank 0 is the dispatcher
        if self.n_replicas < 1:
            raise RuntimeError("Need at least 2 MPI ranks: 1 dispatcher + >=1 replica")
        self.picker = LeastOutstanding(self.n_replicas)

    def _drain(self, rows: list[dict]) -> None:
        status = self.MPI.Status()
        while self.comm.iprobe(source=self.MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            src = status.Get_source()
            msg = self.comm.recv(source=src, tag=TAG_RESULT)
            msg["t_dispatcher_recv"] = time.monotonic()
            self.picker.finish(src - 1)
            rows.append(msg)

    def run(self, requests: list[Request]) -> tuple[list[RequestResult], list[dict]]:
        """Returns (results, per-request overhead rows). Blocks until every request completed."""
        rows: list[dict] = []
        pending_sends = []
        t0 = time.monotonic()
        for req in sorted(requests, key=lambda r: r.arrival_time):
            while time.monotonic() - t0 < req.arrival_time:
                self._drain(rows)
                time.sleep(_POLL_SLEEP)
            idx = self.picker.pick()
            job = {"req": asdict(req), "t0": t0, "t_dispatch": time.monotonic()}
            pending_sends.append(self.comm.isend(job, dest=idx + 1, tag=TAG_JOB))
        while len(rows) < len(requests):
            self._drain(rows)
            time.sleep(_POLL_SLEEP)
        for rank in range(1, self.n_replicas + 1):
            self.comm.send(None, dest=rank, tag=TAG_JOB)       # shutdown sentinel
        self.MPI.Request.Waitall(pending_sends)

        results, overheads = [], []
        for row in rows:
            ov = overhead_us(row)
            overheads.append({"request_id": row["request_id"], "replica": row["worker_rank"] - 1, **ov})
            results.append(RequestResult(
                request_id=row["request_id"], scheduled_time=row["scheduled_time"], start_time=row["start_time"],
                first_token_time=row["first_token_time"], end_time=row["end_time"],
                prompt_tokens=row["prompt_tokens"], output_tokens=row["output_tokens"], ok=row["ok"],
                error=row["error"], replica=row["worker_rank"] - 1,
                route_overhead_us=ov["mpi_dispatch_us"] + ov["mpi_result_us"]))
        return results, overheads
