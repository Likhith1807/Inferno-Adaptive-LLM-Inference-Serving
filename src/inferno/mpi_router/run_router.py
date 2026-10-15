"""
Entry point launched under mpirun:

    mpirun -n <replicas+1> python -m inferno.mpi_router.run_router \
        --requests reqs.json --ports 8000,8001 --model <name> --out out_dir

Rank 0 = dispatcher (writes results.jsonl + overhead.jsonl); ranks 1..N = replica workers (rank r talks to
ports[r-1]).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..workload.generator import Request


def main() -> None:
    from mpi4py import MPI

    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True, help="JSON list of Request dicts")
    ap.add_argument("--ports", required=True, help="comma-separated replica ports, one per worker rank")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    ports = [int(p) for p in args.ports.split(",")]
    if comm.Get_size() != len(ports) + 1:
        raise SystemExit(f"need {len(ports) + 1} ranks for {len(ports)} replicas, got {comm.Get_size()}")

    if comm.Get_rank() == 0:
        from .dispatcher import Dispatcher

        requests = [Request(**r) for r in json.loads(Path(args.requests).read_text())]
        results, overheads = Dispatcher(comm).run(requests)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.jsonl", "w") as f:
            f.writelines(json.dumps(r.to_row()) + "\n" for r in results)
        with open(out / "overhead.jsonl", "w") as f:
            f.writelines(json.dumps(o) + "\n" for o in overheads)
    else:
        from .replica_worker import ReplicaWorker

        ReplicaWorker(f"http://{args.host}:{ports[comm.Get_rank() - 1]}", args.model, comm).run()


if __name__ == "__main__":
    main()
