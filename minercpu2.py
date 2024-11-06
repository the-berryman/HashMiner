from multiprocessing import Process, Queue, Value
import time
import random
import string
import hashlib
from multiprocessing.queues import Empty
import ctypes

NUM_PROCESSES = 50
NONCE_LENGTH = 10


def generate_unique_nonce(process_id, counter, timestamp_bits):
    """
    Generate a unique nonce using:
    - process_id (2 hex chars = 8 bits)
    - timestamp_bits (4 hex chars = 16 bits)
    - counter (4 hex chars = 16 bits)
    """
    prefix = f"{process_id:02x}{timestamp_bits:04x}{counter:04x}"
    remaining_length = NONCE_LENGTH - len(prefix)
    random_chars = ''.join(random.choice(string.ascii_letters + string.digits + "!@#$%^&*")
                           for _ in range(remaining_length))
    return prefix + random_chars


def compute_sha256(names, nonce):
    """Compute a single SHA-256 hash"""
    input_string = f"{names} {nonce}".encode()
    hash_bytes = hashlib.sha256(input_string).digest()
    hex_hash = ''.join(f'{b:02x}' for b in hash_bytes)
    return hex_hash, len(hex_hash) - len(hex_hash.lstrip('0'))


def mine_worker(process_id, queue, team_names, total_hashes, stop_flag):
    """Worker process to mine and report best hashes"""
    names = " ".join(team_names)
    local_best_zeros = 0
    counter = 0
    local_hashes = 0
    start_time = int(time.time())

    print(f"Process {process_id} starting mining...")

    while not stop_flag.value:
        # Get current timestamp bits (16 bits, changes every second)
        timestamp_bits = (int(time.time()) - start_time) & 0xFFFF

        # Generate and test a nonce
        nonce = generate_unique_nonce(process_id, counter, timestamp_bits)
        counter = (counter + 1) & 0xFFFF  # Wrap at 16 bits

        # Compute the hash
        hex_hash, zeros = compute_sha256(names, nonce)
        local_hashes += 1

        # Update shared hash counter periodically
        if local_hashes % 1000 == 0:
            with total_hashes.get_lock():
                total_hashes.value += local_hashes
                local_hashes = 0

        # Send updates when we find a better local hash
        if zeros > local_best_zeros:
            local_best_zeros = zeros
            try:
                queue.put_nowait((process_id, zeros, nonce, hex_hash))
            except:
                pass  # Skip if queue is full


def display_progress(queue, start_time, total_hashes, stop_flag):
    """Handle the logging and display of progress from the worker processes."""
    last_print_time = start_time
    last_hash_count = 0
    global_best_zeros = 0
    global_best_nonce = None
    global_best_hash = None
    global_best_process = None

    while not stop_flag.value:
        try:
            # Non-blocking queue check
            while True:
                try:
                    process_id, zeros, nonce, hex_hash = queue.get_nowait()
                    if zeros > global_best_zeros:
                        global_best_zeros = zeros
                        global_best_nonce = nonce
                        global_best_hash = hex_hash
                        global_best_process = process_id
                        print(f"\nNew best from Process {process_id}! {zeros} leading zeros")
                        print(f"Nonce: {nonce}")
                        print(f"Hash: {hex_hash}")
                except Empty:
                    break

            # Update progress every 2 seconds
            current_time = time.time()
            elapsed = current_time - last_print_time
            if elapsed >= 2.0:
                current_total = total_hashes.value
                hashes_per_second = (current_total - last_hash_count) / elapsed
                print(f"\rTotal Processed: {current_total:,} | "
                      f"Rate: {hashes_per_second:,.0f}/s | Best: {global_best_zeros} leading zeros",
                      end="")
                last_print_time = current_time
                last_hash_count = current_total

        except Exception as e:
            print(f"Display error: {e}")

        time.sleep(0.1)  # Prevent tight loop


def mine(team_names):
    """Mine SHA-256 hashes using multiple processes."""
    queue = Queue(maxsize=100)  # Limit queue size
    total_hashes = Value(ctypes.c_uint64, 0)  # Shared counter
    stop_flag = Value(ctypes.c_bool, False)  # Shared stop flag

    start_time = time.time()

    display_process = Process(target=display_progress,
                              args=(queue, start_time, total_hashes, stop_flag))
    display_process.start()

    processes = []
    for i in range(NUM_PROCESSES):
        process = Process(target=mine_worker,
                          args=(i, queue, team_names, total_hashes, stop_flag))
        processes.append(process)
        process.start()

    try:
        # Wait for keyboard interrupt
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping miners...")
        stop_flag.value = True

    # Clean shutdown
    for process in processes:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()

    display_process.join(timeout=1)
    if display_process.is_alive():
        display_process.terminate()


if __name__ == "__main__":
    team_names = ["Reshanna", "Aasiyah", "Luis", "Gavin"]
    mine(team_names)