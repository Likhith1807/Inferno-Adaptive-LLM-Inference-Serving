"""
MPI replica worker (rank 1..N): owns one vLLM endpoint, executes jobs from rank 0 concurrently, and
returns per-request timing (including its own receive/send timestamps for overhead accounting).

The worker multiplexes many in-flight requests on one asyncio loop; MPI is polled with iprobe between
awaits (single-threaded MPI, no MPI_THREAD_MULTIPLE requirement).
"""
from __future__ import annotations

import asyncio
import time

import httpx

from ..serving.client import send_request
from ..workload.generator import Request
from .routing import TAG_JOB, TAG_RESULT

_POLL_SLEEP = 0.0002


class ReplicaWorker:
    def __init__(self, base_url: str, model: str, comm=None):
        from mpi4py import MPI

        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model

    async def _handle(self, client: httpx.AsyncClient, msg: dict, sends: list) -> None:
        t_recv = msg["t_worker_recv"]
        res = await send_request(client, self.url, self.model, Request(**msg["req"]), msg["t0"])
        row = res.to_row()
        row.update(worker_rank=self.rank, t_dispatch=msg["t_dispatch"], t_worker_recv=t_recv,
                   t_worker_send=time.monotonic())
        sends.append(self.comm.isend(row, dest=0, tag=TAG_RESULT))

    async def serve(self) -> None:
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=128)
        sends: list = []
        tasks: set[asyncio.Task] = set()
        shutdown = False
        async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:
            while not (shutdown and not tasks):
                while not shutdown and self.comm.iprobe(source=0, tag=TAG_JOB):
                    msg = self.comm.recv(source=0, tag=TAG_JOB)
                    if msg is None:
                        shutdown = True
                        break
                    msg["t_worker_recv"] = time.monotonic()
                    task = asyncio.create_task(self._handle(client, msg, sends))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                await asyncio.sleep(_POLL_SLEEP)
        for s in sends:
            s.wait()

    def run(self) -> None:
        asyncio.run(self.serve())
