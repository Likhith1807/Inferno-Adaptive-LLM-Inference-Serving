"""
Pure-MPI coordination microbenchmarks. No GPU, no model: workers echo instantly, so everything measured
is coordination cost. This isolates the H2 question "what does routing itself cost, and how does it grow
with replica count?" from the (GPU-bound) inference work.

    mpirun -n 2  python -m inferno.mpi_router.microbench pingpong
    mpirun -n 9  python -m inferno.mpi_router.microbench dispatch --jobs 20000 --payload 512

pingpong : round-trip latency between two ranks vs message size (pickled objects, as the router sends them).
dispatch : sustained dispatch rate and per-job round-trip latency through the same least-outstanding
           dispatch/drain loop the real router uses, against W echo workers, with a bounded in-flight window.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .routing import TAG_JOB, TAG_RESULT, LeastOutstanding


def pingpong(comm, sizes=(64, 512, 4096, 32768, 262144), reps=2000) -> list[dict] | None:
    rank, out = comm.Get_rank(), []
    for size in sizes:
        payload = {"data": b"x" * size}
        comm.Barrier()
        samples = []
        for _ in range(reps):
            if rank == 0:
                t = time.perf_counter()
                comm.send(payload, dest=1, tag=1)
                comm.recv(source=1, tag=2)
                samples.append((time.perf_counter() - t) * 1e6)
            elif rank == 1:
                comm.recv(source=0, tag=1)
                comm.send(payload, dest=0, tag=2)
        if rank == 0:
            s = np.array(samples[50:])   # drop warmup
            out.append({"bytes": size, "rtt_us_p50": float(np.percentile(s, 50)),
                        "rtt_us_p99": float(np.percentile(s, 99)),
                        "one_way_us_est": float(np.percentile(s, 50) / 2)})
    return out if rank == 0 else None


def dispatch(comm, jobs: int, payload_bytes: int, window: int) -> dict | None:
    from mpi4py import MPI

    rank, workers = comm.Get_rank(), comm.Get_size() - 1
    if rank != 0:                                   # echo worker
        while True:
            msg = comm.recv(source=0, tag=TAG_JOB)
            if msg is None:
                return None
            comm.send(msg, dest=0, tag=TAG_RESULT)

    picker, sent, done = LeastOutstanding(workers), 0, 0
    lat, t_sent = [], {}
    blob = b"x" * payload_bytes
    status = MPI.Status()
    t_start = time.perf_counter()
    while done < jobs:
        while sent < jobs and sent - done < window:
            idx = picker.pick()
            t_sent[sent] = time.perf_counter()
            comm.send({"id": sent, "worker": idx, "blob": blob}, dest=idx + 1, tag=TAG_JOB)
            sent += 1
        if comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            src = status.Get_source()
            msg = comm.recv(source=src, tag=TAG_RESULT)
            picker.finish(src - 1)
            lat.append((time.perf_counter() - t_sent.pop(msg["id"])) * 1e6)
            done += 1
    wall = time.perf_counter() - t_start
    for r in range(1, workers + 1):
        comm.send(None, dest=r, tag=TAG_JOB)
    s = np.array(lat)
    return {"workers": workers, "jobs": jobs, "payload_bytes": payload_bytes, "window": window,
            "jobs_per_s": jobs / wall, "rtt_us_p50": float(np.percentile(s, 50)),
            "rtt_us_p99": float(np.percentile(s, 99))}


def main() -> None:
    from mpi4py import MPI

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pingpong", "dispatch"])
    ap.add_argument("--jobs", type=int, default=20000)
    ap.add_argument("--payload", type=int, default=512)
    ap.add_argument("--window", type=int, default=64)
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    result = pingpong(comm) if args.mode == "pingpong" else dispatch(comm, args.jobs, args.payload, args.window)
    if comm.Get_rank() == 0 and result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
