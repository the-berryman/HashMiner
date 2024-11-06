from multiprocessing import Process, Queue, Value
import time
import random
import string
import hashlib
from multiprocessing.queues import Empty
import ctypes
import signal
import sys

NUM_PROCESSES = 40
NONCE_LENGTH = 10


def generate_unique_nonce(process_id, counter, timestamp_bits):
    """Generate a unique nonce using process_id, timestamp, and counter"""
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
    try:
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

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Process {process_id} error: {e}")
    finally:
        # Make sure to add any remaining hashes to the total
        if local_hashes > 0:
            with total_hashes.get_lock():
                total_hashes.value += local_hashes


def display_progress(queue, start_time, total_hashes, stop_flag):
    """Handle the logging and display of progress"""
    try:
        last_print_time = start_time
        last_hash_count = 0
        global_best_zeros = 0

        while not stop_flag.value:
            try:
                # Non-blocking queue check
                while True:
                    try:
                        process_id, zeros, nonce, hex_hash = queue.get_nowait()
                        if zeros > global_best_zeros:
                            global_best_zeros = zeros
                            print(f"\nNew best from Process {process_id}! {zeros} leading zeros")
                            print(f"Nonce: {nonce}")
                            print(f"Hash: {hex_hash}")

                            # Save to file
                            with open('best_hashes.txt', 'a') as f:
                                f.write(f"Process {process_id} - {zeros} zeros\n")
                                f.write(f"Nonce: {nonce}\n")
                                f.write(f"Hash: {hex_hash}\n")
                                f.write("-" * 50 + "\n")
                    except Empty:
                        break

                # Update progress every 2 seconds
                current_time = time.time()
                elapsed = current_time - last_print_time
                if elapsed >= 2.0:
                    current_total = total_hashes.value
                    hashes_per_second = (current_total - last_hash_count) / elapsed
                    print(f"\rTotal Processed: {current_total:,} | "
                          f"Rate: {hashes_per_second:,.0f}/s | "
                          f"Best: {global_best_zeros} leading zeros",
                          end="", flush=True)
                    last_print_time = current_time
                    last_hash_count = current_total

            except Exception as e:
                print(f"Display error: {e}")

            time.sleep(0.1)  # Prevent tight loop

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Display process error: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutdown signal received. Cleaning up...")
    sys.exit(0)


def mine(team_names):
    """Mine SHA-256 hashes using multiple processes"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    queue = Queue(maxsize=100)
    total_hashes = Value(ctypes.c_uint64, 0)
    stop_flag = Value(ctypes.c_bool, False)
    start_time = time.time()

    # Start processes
    processes = []
    display_process = Process(target=display_progress,
                              args=(queue, start_time, total_hashes, stop_flag))
    display_process.daemon = True  # This will help with clean shutdown
    display_process.start()

    for i in range(NUM_PROCESSES):
        process = Process(target=mine_worker,
                          args=(i, queue, team_names, total_hashes, stop_flag))
        process.daemon = True  # This will help with clean shutdown
        processes.append(process)
        process.start()

    try:
        # Wait for keyboard interrupt
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping miners...")
    finally:
        # Clean shutdown
        stop_flag.value = True
        time.sleep(0.5)  # Give processes time to see the stop flag

        # Clean termination of processes
        for process in processes:
            process.terminate()
            process.join(timeout=1)

        display_process.terminate()
        display_process.join(timeout=1)

        # Final statistics
        total_time = time.time() - start_time
        print(f"\nFinal Results:")
        print(f"Total hashes: {total_hashes.value:,}")
        print(f"Time elapsed: {total_time:.2f} seconds")
        print(f"Average hash rate: {total_hashes.value / total_time:,.0f} hashes/second")


if __name__ == "__main__":
    team_names = ["Reshanna", "Aasiyah", "Luis", "Gavin"]
    mine(team_names)