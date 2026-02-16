"""
ThreadPoolExecutor examples — run this to understand how it works.

Usage:
    python scripts/threadpool_examples.py
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# =============================================================================
# Example 1: Basic — sequential vs parallel
# =============================================================================

def slow_task(task_id, duration):
    """Simulates a party training for `duration` seconds."""
    print(f"  Task {task_id} started")
    time.sleep(duration)
    print(f"  Task {task_id} finished (took {duration}s)")
    return f"result_{task_id}"


def example_1_sequential():
    print("\n=== Example 1a: Sequential ===")
    start = time.time()

    results = {}
    for task_id in range(5):
        results[task_id] = slow_task(task_id, 1.0)

    elapsed = time.time() - start
    print(f"Sequential: {elapsed:.2f}s (5 tasks x 1s each)\n")
    return results


def example_1_parallel():
    print("=== Example 1b: Parallel with ThreadPoolExecutor ===")
    start = time.time()

    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks at once — they start running immediately
        futures = {
            executor.submit(slow_task, task_id, 1.0): task_id
            for task_id in range(5)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            task_id = futures[future]
            results[task_id] = future.result()

    elapsed = time.time() - start
    print(f"Parallel:   {elapsed:.2f}s (5 tasks x 1s, 5 workers)\n")
    return results


# =============================================================================
# Example 2: Simulated FL — parties writing to shared dict
# =============================================================================

def party_train(bank_id, shared_weights_dict, duration=0.5):
    """Simulates a party doing local training and writing weights to shared dict.

    This mirrors your FL pattern:
      party.update_local_weights()
      party.send_local_weights(manager)  # writes to manager.parties_weights[bank_id]
    """
    # Simulate training work
    time.sleep(duration)

    # Write to shared dict with unique key — thread-safe
    shared_weights_dict[bank_id] = {"weight": bank_id * 10, "bias": bank_id * 0.1}

    return bank_id


def example_2_fl_simulation():
    print("=== Example 2: Simulated FL round with shared state ===")

    # This is like manager.parties_weights = {}
    parties_weights = {}
    num_parties = 8

    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(party_train, bank_id, parties_weights): bank_id
            for bank_id in range(num_parties)
        }

        for future in as_completed(futures):
            bank_id = futures[future]
            future.result()  # raises exception if task failed

    elapsed = time.time() - start

    print(f"  {num_parties} parties trained in {elapsed:.2f}s (4 workers, 0.5s each)")
    print(f"  Expected sequential: {num_parties * 0.5:.1f}s")
    print(f"  Collected weights from {len(parties_weights)} parties:")
    for bank_id in sorted(parties_weights.keys()):
        print(f"    Bank {bank_id}: {parties_weights[bank_id]}")
    print()


# =============================================================================
# Example 3: Error handling — what happens when a task fails
# =============================================================================

def failing_task(task_id):
    time.sleep(0.2)
    if task_id == 3:
        raise ValueError(f"Task {task_id} failed!")
    return f"ok_{task_id}"


def example_3_error_handling():
    print("=== Example 3: Error handling ===")

    results = {}
    errors = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(failing_task, task_id): task_id
            for task_id in range(5)
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                results[task_id] = future.result()
            except Exception as e:
                errors[task_id] = str(e)
                print(f"  Task {task_id} raised: {e}")

    print(f"  Succeeded: {sorted(results.keys())}")
    print(f"  Failed:    {sorted(errors.keys())}")
    print()


# =============================================================================
# Example 4: CPU-bound work — Thread vs Process comparison
# =============================================================================

def cpu_work(n):
    """CPU-bound task: sum of squares."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def example_4_thread_vs_process():
    print("=== Example 4: Thread vs Process for CPU-bound work ===")
    print("  (This shows the GIL effect on pure Python code)\n")

    n = 5_000_000
    num_tasks = 4

    # Sequential
    start = time.time()
    for _ in range(num_tasks):
        cpu_work(n)
    seq_time = time.time() - start
    print(f"  Sequential:            {seq_time:.2f}s")

    # ThreadPool — won't help for pure Python (GIL)
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(cpu_work, n) for _ in range(num_tasks)]
        [f.result() for f in futures]
    thread_time = time.time() - start
    print(f"  ThreadPoolExecutor:    {thread_time:.2f}s (GIL prevents true parallelism)")

    # ProcessPool — real parallelism for CPU work
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(cpu_work, n) for _ in range(num_tasks)]
        [f.result() for f in futures]
    proc_time = time.time() - start
    print(f"  ProcessPoolExecutor:   {proc_time:.2f}s (true parallelism, no GIL)")

    print(f"\n  Key takeaway: For pure Python loops, ProcessPool wins.")
    print(f"  But XGBoost/PyTorch release the GIL, so ThreadPool works for those.")
    print()


# =============================================================================
# Example 5: Simulating your FL pattern more closely
# =============================================================================

def example_5_fl_pattern():
    """Mimics the actual FL training round from federated_manager.py"""
    print("=== Example 5: Full FL round simulation ===\n")

    # --- Setup (like manager initialization) ---
    num_parties = 15
    num_rounds = 3
    parties_weights = {}
    global_weights = {"w": 0.0}

    def local_train_and_send(bank_id, parties_weights, global_w):
        """What each party does in a round."""
        # Read global weights (read-only, safe)
        local_w = global_w["w"]

        # Simulate local training (the expensive part)
        time.sleep(0.3)  # stands in for model.update_weights()
        new_w = local_w + bank_id * 0.01

        # Send to manager (write to unique key, safe)
        parties_weights[bank_id] = {"w": new_w}

    def aggregate(parties_weights):
        """Manager aggregates — must happen after all parties finish."""
        avg_w = sum(pw["w"] for pw in parties_weights.values()) / len(parties_weights)
        return {"w": avg_w}

    # --- FL Training Loop ---
    total_start = time.time()

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        parties_weights = {}  # Clear each round

        # Parallel local training
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(local_train_and_send, bid, parties_weights, global_weights): bid
                for bid in range(num_parties)
            }
            for future in as_completed(futures):
                future.result()  # check for errors

        # Sequential aggregation (must wait for all parties)
        global_weights = aggregate(parties_weights)

        round_time = time.time() - round_start
        print(f"  Round {round_num}: {round_time:.2f}s | "
              f"global_w = {global_weights['w']:.4f} | "
              f"{num_parties} parties, 4 workers")

    total_time = time.time() - total_start
    seq_estimate = num_parties * 0.3 * num_rounds
    print(f"\n  Total: {total_time:.2f}s (sequential would be ~{seq_estimate:.1f}s)")
    print(f"  Speedup: ~{seq_estimate / total_time:.1f}x")
    print()


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ThreadPoolExecutor Tutorial")
    print("=" * 60)

    example_1_sequential()
    example_1_parallel()
    example_2_fl_simulation()
    example_3_error_handling()
    example_4_thread_vs_process()
    example_5_fl_pattern()

    print("=" * 60)
    print("Done! Key takeaways:")
    print("  1. ThreadPoolExecutor runs tasks concurrently in threads")
    print("  2. Shared dict writes with unique keys are thread-safe")
    print("  3. GIL limits pure Python, but C extensions (PyTorch, XGBoost) release it")
    print("  4. Error handling via try/except on future.result()")
    print("  5. max_workers controls concurrency (= num GPUs for GNN, or num CPU groups for XGBoost)")
    print("=" * 60)
