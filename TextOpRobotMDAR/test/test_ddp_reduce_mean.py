"""DDP eval-metrics reduce robustness (ddp_reduce_mean).

The eval extras dicts carry data-dependent per-class keys, so key sets can
differ across ranks. ddp_reduce_mean must reduce the gathered key union in
canonical order (identical collective sequence on every rank) with missing
keys contributing zeros — otherwise the NCCL streams diverge and the run
deadlocks with a watchdog timeout.
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TextOpRobotMDAR.robotmdar.train.manager import ddp_reduce_mean


def _worker(rank, world_size, port, out_queue):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    try:
        # Deliberately mismatched key sets: rank 0 has 'c', rank 1 has 'd'.
        d = {'a': torch.tensor(2.0), 'b': torch.tensor(1.0)}
        if rank == 0:
            d['c'] = torch.tensor(10.0)
        else:
            d['d'] = torch.tensor(4.0)
        out = ddp_reduce_mean(d)
        out_queue.put((rank, {k: v.item() for k, v in out.items()}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('port', [29540])
def test_reduce_union_of_mismatched_key_sets(port):
    ctx = mp.get_context('spawn')
    out_queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, out_queue))
             for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f'rank exited with {p.exitcode}'
    results = [out_queue.get(timeout=10) for _ in range(2)]
    results.sort(key=lambda item: item[0])
    _, rank0, rank1 = results[0][0], results[0][1], results[1][1]

    # Union of keys on both ranks, reduced in the same canonical order.
    assert set(rank0) == set(rank1) == {'a', 'b', 'c', 'd'}
    assert rank0 == rank1
    assert rank0['a'] == pytest.approx(2.0)
    assert rank0['b'] == pytest.approx(1.0)
    assert rank0['c'] == pytest.approx(5.0)  # (10 + 0) / 2
    assert rank0['d'] == pytest.approx(2.0)  # (0 + 4) / 2
