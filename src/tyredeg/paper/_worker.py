"""One shard of a parallel job, run as its own interpreter.

Started by ``model_checks._run_sharded``. Deliberately NOT a
``multiprocessing`` pool: on Windows, spawned pool workers occasionally die
with "the handle is invalid" when several processes spawn at once (several
analyses share this machine), taking the whole pool with them. A plain
subprocess that reads a pickle and writes a pickle has no inherited handles to
lose.

    python -m tyredeg.paper._worker JOB.pkl SHARD N_SHARDS OUT.pkl
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import warnings


def main(argv: list[str]) -> int:
    job_path, shard, n_shards, out_path = argv[0], int(argv[1]), int(argv[2]), argv[3]
    warnings.filterwarnings("ignore")
    with open(job_path, "rb") as fh:
        job = pickle.load(fh)

    from tyredeg.paper import model_checks as mc

    if job["df"] is not None:
        mc._init_worker_spec(job["df"], job["spec"], job["payload"])
    fn = getattr(mc, job["fn"])
    items = job["items"]
    # Progress is saved after every item, so a killed shard resumes rather
    # than restarting: items are pure functions of their seeded arguments.
    partial = out_path + ".partial"
    out: list = []
    if os.path.exists(partial):
        with open(partial, "rb") as fh:
            out = pickle.load(fh)
    done = {i for i, _ in out}
    for i in range(shard, len(items), n_shards):
        if i in done:
            continue
        out.append((i, fn(items[i])))
        with open(partial + ".tmp", "wb") as fh:
            pickle.dump(out, fh)
        _replace(partial + ".tmp", partial)
    with open(out_path + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    _replace(out_path + ".tmp", out_path)
    return 0


def _replace(src: str, dst: str) -> None:
    """``os.replace``, retried: Windows refuses it while another process reads ``dst``."""
    for _ in range(50):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(src, dst)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
