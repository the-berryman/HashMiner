import os
import json
import random
import string
from datetime import datetime
import time
import numpy as np
import cupy as cp
import psutil
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler as cuda_compiler
import subprocess
import sys
from pathlib import Path
import concurrent.futures
from pycuda.compiler import SourceModule
from threading import Thread, Lock
from queue import Queue



def setup_msvc():
    """Setup Microsoft Visual C++ compiler environment"""
    # Common paths for Visual Studio 2019/2022
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
    ]

    msvc_paths = [
        r"VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64",  # 2019
        r"VC\Tools\MSVC\14.30.30705\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.31.31103\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.32.31326\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64",  # 2022
        r"VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64",  # 2022
    ]

    # Find Visual Studio installation
    vs_path = None
    cl_path = None

    for base_path in vs_paths:
        if os.path.exists(base_path):
            vs_path = base_path
            # Try to find cl.exe in possible MSVC paths
            for msvc_rel_path in msvc_paths:
                full_cl_path = os.path.join(base_path, msvc_rel_path, "cl.exe")
                if os.path.exists(full_cl_path):
                    cl_path = os.path.dirname(full_cl_path)
                    break
            if cl_path:
                break

    if not cl_path:
        raise RuntimeError("Could not find Visual C++ compiler (cl.exe)")

    # Add Visual Studio environment variables
    vs_tools_path = os.path.join(vs_path, "Common7", "Tools")
    if os.path.exists(vs_tools_path):
        # Run vsvars64.bat to set up environment
        vcvars_path = os.path.join(vs_tools_path, "VsDevCmd.bat")
        if os.path.exists(vcvars_path):
            # Run vcvars64.bat and capture environment
            process = subprocess.Popen(
                f'"{vcvars_path}" && set',
                stdout=subprocess.PIPE,
                shell=True
            )
            output = process.communicate()[0].decode('utf-8').split('\n')

            # Update current environment with Visual Studio variables
            for line in output:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

    # Add paths to environment
    os.environ['VS_PATH'] = vs_path

    # Update PATH to include cl.exe
    if cl_path not in os.environ['PATH']:
        os.environ['PATH'] = cl_path + os.pathsep + os.environ['PATH']

    # Set CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
    ]

    cuda_path = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_path = path
            break

    if cuda_path:
        os.environ['CUDA_PATH'] = cuda_path
        cuda_bin = os.path.join(cuda_path, 'bin')
        cuda_lib = os.path.join(cuda_path, 'lib64')

        # Add CUDA paths to PATH
        if cuda_bin not in os.environ['PATH']:
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
        if cuda_lib not in os.environ['PATH']:
            os.environ['PATH'] = cuda_lib + os.pathsep + os.environ['PATH']

    # Print diagnostic information
    print("Environment Setup:")
    print(f"Visual Studio Path: {vs_path}")
    print(f"MSVC Compiler Path: {cl_path}")
    print(f"CUDA Path: {cuda_path}")

    # Verify cl.exe is accessible
    try:
        subprocess.run(['cl'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print("Successfully located cl.exe")
    except Exception as e:
        print(f"Warning: Could not verify cl.exe: {e}")
        print("Current PATH:")
        print(os.environ['PATH'].replace(os.pathsep, '\n'))


def init_cuda_compilation():
    """Initialize environment for CUDA compilation"""
    try:
        setup_msvc()
    except Exception as e:
        print(f"Error setting up MSVC: {e}")
        sys.exit(1)

    # Verify CUDA compiler
    try:
        result = subprocess.run(['nvcc', '--version'],
                                capture_output=True,
                                text=True)
        print("\nCUDA Compiler Version:")
        print(result.stdout)
    except Exception as e:
        print(f"Error verifying CUDA compiler: {e}")
        sys.exit(1)


CUDA_KERNEL = '''
extern "C" {
    __device__ __constant__ unsigned int k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };

    __device__ __constant__ char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*";
    __device__ __constant__ int charset_size = 72;

    __device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
        return (x >> n) | (x << (32 - n));
    }

    __device__ __forceinline__ unsigned int ch(unsigned int x, unsigned int y, unsigned int z) {
        return (x & y) ^ (~x & z);
    }

    __device__ __forceinline__ unsigned int maj(unsigned int x, unsigned int y, unsigned int z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    __device__ __forceinline__ unsigned int ep0(unsigned int x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }

    __device__ __forceinline__ unsigned int ep1(unsigned int x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    __device__ __forceinline__ unsigned int sig0(unsigned int x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }

    __device__ __forceinline__ unsigned int sig1(unsigned int x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    __device__ __forceinline__ unsigned int xorshift(unsigned int* state) {
        unsigned int x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        return x;
    }

    __device__ __forceinline__ int count_leading_zeros(const unsigned char* hash) {
        int zeros = 0;
        for(int i = 0; i < 32; i++) {
            if(hash[i] == 0) {
                zeros += 2;
            } else {
                unsigned char h = hash[i];
                if((h & 0xF0) == 0) {
                    zeros++;
                    if((h & 0x0F) == 0) zeros++;
                }
                break;
            }
        }
        return zeros;
    }

    __global__ void sha256_kernel(
        const unsigned char* base_input,
        const int base_len,
        unsigned char* best_hash_out,
        char* best_nonce_out,
        int* best_zeros_out,
        const unsigned int seed,
        const int iterations_per_thread)
    {
        __shared__ int shared_best_zeros;
        __shared__ unsigned char shared_best_hash[32];
        __shared__ char shared_best_nonce[10];

        // Thread-local variables
        unsigned int rand_state = seed + (blockIdx.x * blockDim.x + threadIdx.x);
        unsigned char local_hash[32];
        char local_nonce[10];
        int local_best_zeros = 0;

        // Local buffers with explicit initialization
        unsigned char message[128] = {0};
        unsigned int w[64] = {0};
        unsigned int state[8] = {0};

        // Initialize shared memory
        if(threadIdx.x == 0) {
            shared_best_zeros = *best_zeros_out;
        }
        __syncthreads();

        // Copy base input to local buffer
        for(int i = 0; i < base_len; i++) {
            message[i] = base_input[i];
        }
        message[base_len] = ' ';

        // Main processing loop
        for(int iter = 0; iter < iterations_per_thread; iter++) {
            // Generate nonce
            for(int i = 0; i < 10; i++) {
                unsigned int rand_val = xorshift(&rand_state);
                message[base_len + 1 + i] = charset[rand_val % charset_size];
                local_nonce[i] = message[base_len + 1 + i];
            }

            // SHA-256 processing
            // Initialize state
            state[0] = 0x6a09e667;
            state[1] = 0xbb67ae85;
            state[2] = 0x3c6ef372;
            state[3] = 0xa54ff53a;
            state[4] = 0x510e527f;
            state[5] = 0x9b05688c;
            state[6] = 0x1f83d9ab;
            state[7] = 0x5be0cd19;

            const int total_len = base_len + 11;

            // Process message
            for(int i = 0; i < 16; i++) {
                w[i] = 0;
                for(int j = 0; j < 4 && (i * 4 + j) < total_len; j++) {
                    w[i] |= ((unsigned int)message[i * 4 + j] << (24 - j * 8));
                }
            }

            w[total_len / 4] |= 0x80U << (24 - (total_len % 4) * 8);
            w[15] = total_len * 8;

            for(int i = 16; i < 64; i++) {
                w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];
            }

            unsigned int a = state[0],
                         b = state[1],
                         c = state[2],
                         d = state[3],
                         e = state[4],
                         f = state[5],
                         g = state[6],
                         h = state[7];

            for(int i = 0; i < 64; i++) {
                unsigned int t1 = h + ep1(e) + ch(e,f,g) + k[i] + w[i];
                unsigned int t2 = ep0(a) + maj(a,b,c);
                h = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }

            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
            state[4] += e;
            state[5] += f;
            state[6] += g;
            state[7] += h;

            for(int i = 0; i < 8; i++) {
                local_hash[i*4] = (state[i] >> 24) & 0xFF;
                local_hash[i*4 + 1] = (state[i] >> 16) & 0xFF;
                local_hash[i*4 + 2] = (state[i] >> 8) & 0xFF;
                local_hash[i*4 + 3] = state[i] & 0xFF;
            }

            int zeros = count_leading_zeros(local_hash);
            if(zeros > local_best_zeros) {
                local_best_zeros = zeros;
                if(zeros > shared_best_zeros) {
                    atomicMax(&shared_best_zeros, zeros);
                    if(zeros == shared_best_zeros) {
                        for(int i = 0; i < 32; i++) {
                            shared_best_hash[i] = local_hash[i];
                        }
                        for(int i = 0; i < 10; i++) {
                            shared_best_nonce[i] = local_nonce[i];
                        }
                    }
                }
            }
        }

        __syncthreads();

        if(threadIdx.x == 0 && shared_best_zeros > *best_zeros_out) {
            *best_zeros_out = shared_best_zeros;
            for(int i = 0; i < 32; i++) {
                best_hash_out[i] = shared_best_hash[i];
            }
            for(int i = 0; i < 10; i++) {
                best_nonce_out[i] = shared_best_nonce[i];
            }
        }
    }
}
''';


class ResultManager:
    def __init__(self, team_names):
        self.team_names = team_names
        self.results_file = 'mining_results.json'
        self.save_queue = Queue()
        self.save_lock = Lock()
        self.known_results = set()  # For fast duplicate checking
        self._load_previous_results()

        # Start background save thread
        self.running = True
        self.save_thread = Thread(target=self._background_save, daemon=True)
        self.save_thread.start()

    def _load_previous_results(self):
        """Load previous results without blocking mining"""
        team_key = '_'.join(self.team_names)
        self.data = {
            'best_zeros': 0,
            'results': [],
            'total_hashes': 0,
            'last_seed': int(time.time())
        }

        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    saved_data = json.load(f)
                    if team_key in saved_data:
                        self.data = saved_data[team_key]
                        # Build set of known hashes for fast duplicate checking
                        self.known_results = {r['hash'] for r in self.data['results']}
            except:
                pass  # Use default data if loading fails

        return self.data

    def _background_save(self):
        """Background thread for saving results"""
        while self.running or not self.save_queue.empty():
            try:
                result = self.save_queue.get(timeout=1.0)
                if result:
                    team_key = '_'.join(self.team_names)
                    with self.save_lock:
                        if os.path.exists(self.results_file):
                            try:
                                with open(self.results_file, 'r') as f:
                                    all_data = json.load(f)
                            except:
                                all_data = {}
                        else:
                            all_data = {}

                        all_data[team_key] = result

                        with open(self.results_file, 'w') as f:
                            json.dump(all_data, f, indent=2)
            except:
                continue

    def add_result(self, zeros, nonce, hash_hex, total_hashes, seed):
        """Queue a result for background saving"""
        if hash_hex not in self.known_results:
            self.known_results.add(hash_hex)
            result_entry = {
                'zeros': zeros,
                'nonce': nonce,
                'hash': hash_hex,
                'timestamp': datetime.now().isoformat()
            }
            self.data['results'].append(result_entry)
            self.data['total_hashes'] = total_hashes
            self.data['last_seed'] = seed
            if zeros > self.data['best_zeros']:
                self.data['best_zeros'] = zeros

            # Queue for background saving
            self.save_queue.put(self.data)

    def get_stats(self):
        """Get current statistics"""
        return {
            'best_zeros': self.data['best_zeros'],
            'total_results': len(self.data['results']),
            'total_hashes': self.data['total_hashes'],
            'last_seed': self.data['last_seed']
        }

    def cleanup(self):
        """Clean up background thread"""
        self.running = False
        self.save_thread.join()

def mine_hashes(team_names, batch_size=50000000):
    """Mine hashes with optimized GPU implementation"""
    """Mine hashes with optimized GPU implementation and zero-impact result saving"""
    names = " ".join(team_names)
    print(f"\nStarting optimized GPU mining...")
    print(f"Team Names: {names}")

    # Initialize result manager
    result_manager = ResultManager(team_names)
    stats = result_manager.get_stats()

    print(f"\nLoaded previous results:")
    print(f"Best zeros found: {stats['best_zeros']}")
    print(f"Known good nonces: {stats['total_results']}")
    print(f"Total previous hashes: {stats['total_hashes']:,}")

    # RTX 3090 specific optimizations
    options = [
        '-O3',
        '--use_fast_math',
        '--gpu-architecture=sm_86',
        '--ptxas-options=-v',
        '--maxrregcount=64',  # Optimize register usage
        '-DFULL_UNROLL'  # Enable full loop unrolling
    ]

    mod = SourceModule(CUDA_KERNEL, no_extern_c=True, options=options)
    sha256_kernel = mod.get_function("sha256_kernel")

    # Enable L1 cache
    sha256_kernel.set_cache_config(cuda.func_cache.PREFER_L1)

    # Prepare input data with proper alignment
    base_input = np.frombuffer(names.encode(), dtype=np.uint8)
    base_len = len(base_input)

    # Use page-locked memory for faster transfers
    base_input_host = cuda.pagelocked_empty(base_len, np.uint8)
    best_hash_host = cuda.pagelocked_empty(32, np.uint8)
    best_nonce_host = cuda.pagelocked_empty(16, np.dtype('S1'))  # Aligned to 16
    best_zeros_host = cuda.pagelocked_empty(1, np.int32)

    base_input_host[:] = base_input[:]

    # Allocate GPU memory with proper alignment
    base_input_gpu = cuda.mem_alloc(((base_len + 255) // 256) * 256)
    best_hash_gpu = cuda.mem_alloc(32)
    best_nonce_gpu = cuda.mem_alloc(16)
    best_zeros_gpu = cuda.mem_alloc(4)

    cuda.memcpy_htod(base_input_gpu, base_input_host)
    best_zeros_host[0] = 0
    cuda.memcpy_htod(best_zeros_gpu, best_zeros_host)

    # Get device properties
    device = cuda.Device(0)
    attributes = device.get_attributes()
    num_multiprocessors = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

    # RTX 3090 optimized configuration
    block_size = 256
    warps_per_block = block_size // 32
    blocks_per_sm = 32  # Increased for better occupancy
    grid_size = num_multiprocessors * blocks_per_sm
    iterations_per_thread = 2000  # Increased for better amortization

    print(f"\nGPU Configuration:")
    print(f"Device: {device.name()}")
    print(f"Number of SMs: {num_multiprocessors}")
    print(f"Warps per Block: {warps_per_block}")
    print(f"Blocks per SM: {blocks_per_sm}")
    print(f"Block Size: {block_size}")
    print(f"Grid Size: {grid_size}")
    print(f"Iterations per Thread: {iterations_per_thread}")
    print(f"Total Parallel Threads: {block_size * grid_size:,}")
    print(f"Effective Batch Size: {block_size * grid_size * iterations_per_thread:,}")

    # Create multiple CUDA streams for better overlap
    num_streams = 2
    streams = [cuda.Stream() for _ in range(num_streams)]
    current_stream = 0

    start_time = time.time()
    last_print_time = start_time
    total_hashes = 0
    seed = int(time.time())
    best_zeros = 0

    try:
        while True:
            stream = streams[current_stream]

            # Launch kernel
            sha256_kernel(
                base_input_gpu,
                np.int32(base_len),
                best_hash_gpu,
                best_nonce_gpu,
                best_zeros_gpu,
                np.uint32(seed),
                np.int32(iterations_per_thread),
                block=(block_size, 1, 1),
                grid=(grid_size, 1, 1),
                stream=stream
            )

            current_batch = block_size * grid_size * iterations_per_thread
            total_hashes += current_batch
            seed = (seed * 1664525 + 1013904223) & 0xffffffff

            # Check results asynchronously
            cuda.memcpy_dtoh_async(best_zeros_host, best_zeros_gpu, stream)
            stream.synchronize()

            if best_zeros_host[0] > best_zeros:
                best_zeros = best_zeros_host[0]
                cuda.memcpy_dtoh(best_hash_host, best_hash_gpu)
                cuda.memcpy_dtoh(best_nonce_host, best_nonce_gpu)

                hex_hash = ''.join(f'{b:02x}' for b in best_hash_host)
                nonce_str = best_nonce_host.tobytes().decode('ascii').strip('\x00')

                print(f"\nNew best! {best_zeros} leading zeros!")
                print(f"Input: {names} {nonce_str}")
                print(f"Hash: {hex_hash}")

                # Save the best result to a file
                with open('best_hash.txt', 'a') as f:
                    f.write(f"Team Names: {names}\n")
                    f.write(f"Nonce: {nonce_str}\n")
                    f.write(f"Hash: {hex_hash}\n")
                    f.write(f"Leading Zeros: {best_zeros}\n")
                    f.write("-" * 50 + "\n")

            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - last_print_time
                hash_rate = current_batch / elapsed
                gpu_util, gpu_power, gpu_temp = get_gpu_stats()

                print(f"\rHashes: {total_hashes:,} | "
                      f"Rate: {hash_rate:,.0f}/s | "
                      f"Power: {gpu_power:.1f}W | "
                      f"Temp: {gpu_temp}°C | "
                      f"GPU: {gpu_util}%", end="", flush=True)

                last_print_time = current_time

            # Rotate streams
            current_stream = (current_stream + 1) % num_streams

    except KeyboardInterrupt:
        for stream in streams:
            stream.synchronize()
        total_time = time.time() - start_time
        print("\n\nStopping...")
        print(f"\nFinal Results:")
        print(f"Total hashes: {total_hashes:,}")
        print(f"Time elapsed: {total_time:.2f} seconds")
        print(f"Average hash rate: {total_hashes / total_time:,.0f} hashes/second")
        print(f"Best leading zeros found: {best_zeros}")

    finally:
        base_input_gpu.free()
        best_hash_gpu.free()
        best_nonce_gpu.free()
        best_zeros_gpu.free()

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        import subprocess
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return int(output.strip())
    except:
        return 0

def get_gpu_stats():
    """Get GPU utilization, power, and temperature"""
    try:
        import subprocess
        # Get GPU stats
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        util, power, temp = map(float, result.strip().split(', '))
        return util, power, temp
    except:
        return 0, 0, 0


if __name__ == "__main__":
    # Initialize CUDA compilation environment first
    init_cuda_compilation()

    # Then proceed with the rest of your code
    team_names = ["Reshanna", "Aasiyah", "Luis", "Gavin"]
    mine_hashes(team_names)