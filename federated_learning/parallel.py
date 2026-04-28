"""Parallel execution utilities for federated learning party training."""

import logging
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def get_available_gpus():
    """Return list of available CUDA device indices, or empty list if none."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_device_for_party(party_index, available_gpus):
    """Round-robin GPU assignment for a party. Falls back to CPU if no GPUs."""
    if not available_gpus:
        return torch.device("cpu")
    gpu_idx = available_gpus[party_index % len(available_gpus)]
    return torch.device(f"cuda:{gpu_idx}")


def parallel_party_execute(parties, task_fn, max_workers=None):
    """Run task_fn on each party in parallel using ThreadPoolExecutor.

    Args:
        parties: dict of {bank_id: party} to execute on.
        task_fn: callable(bank_id, party) -> result.
        max_workers: number of threads. None = len(parties). Set to 1 for sequential.

    Returns:
        dict of {bank_id: result} for each party.

    Raises:
        RuntimeError: if any party raises an exception, with bank_id context.
    """
    if max_workers is None:
        max_workers = len(parties)

    # Sequential fallback — no thread overhead
    if max_workers == 1:
        results = {}
        for bank_id, party in parties.items():
            results[bank_id] = task_fn(bank_id, party)
        return results

    results = {}
    errors = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(task_fn, bank_id, party): bank_id
            for bank_id, party in parties.items()
        }
        for future in as_completed(futures):
            bank_id = futures[future]
            try:
                results[bank_id] = future.result()
            except Exception as e:
                logger.error("Party %s failed: %s", bank_id, e)
                errors[bank_id] = e

    if errors:
        failed = ", ".join(str(bid) for bid in errors.keys())
        raise RuntimeError(f"Parties failed: {failed}. First error: {errors[next(iter(errors))]}")

    return results
