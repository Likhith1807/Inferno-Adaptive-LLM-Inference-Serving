"""Pure routing policies and the wire protocol. No MPI import, so this is unit-testable anywhere."""
from __future__ import annotations

TAG_JOB = 10
TAG_RESULT = 11


class LeastOutstanding:
    """Route to the replica with the fewest in-flight requests (ties -> lowest index).
    Replica ids are 0..n-1; the MPI dispatcher maps id i to rank i+1."""

    def __init__(self, n_replicas: int):
        if n_replicas < 1:
            raise ValueError("need at least one replica")
        self.outstanding = [0] * n_replicas

    def pick(self) -> int:
        idx = min(range(len(self.outstanding)), key=lambda i: self.outstanding[i])
        self.outstanding[idx] += 1
        return idx

    def finish(self, idx: int) -> None:
        if self.outstanding[idx] <= 0:
            raise RuntimeError(f"finish() on replica {idx} with nothing outstanding")
        self.outstanding[idx] -= 1


def overhead_us(row: dict) -> dict:
    """Split one request's coordination cost into its two MPI hops (microseconds).
    All timestamps come from time.monotonic(); on one Linux host that clock is shared across
    processes, so cross-rank differences are meaningful."""
    return {
        "mpi_dispatch_us": (row["t_worker_recv"] - row["t_dispatch"]) * 1e6,        # isend -> worker sees it
        "mpi_result_us": (row["t_dispatcher_recv"] - row["t_worker_send"]) * 1e6,   # worker isend -> dispatcher sees it
    }
