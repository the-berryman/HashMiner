```markdown
# CPU SHA-256 Hash Miner (see below for readme on GPU Hash Miner)

A high-performance CPU-based SHA-256 hash mining tool optimized for the Intel i9-10900 processor. Features multi-process mining, unique nonce generation, and clean process management.

## Features

### Core Features
- Multi-process architecture utilizing all CPU cores
- Unique nonce generation combining:
  - Process ID (2 hex chars)
  - Timestamp (4 hex chars)
  - Counter (4 hex chars)
  - Random suffix
- Real-time performance monitoring
- Clean process shutdown
- Automatic hash result logging

### Performance
- Optimized for Intel i9-10900 (10 cores/20 threads)
- Typical performance: ~170,000-190,000 hashes/second
- Auto-scales to CPU core count
- Non-blocking result reporting

### Nonce Format
```
[Process ID][Timestamp][Counter][Random]
    2 hex     4 hex     4 hex    Random
```

## Requirements

```
Python 3.7+
multiprocessing
hashlib
psutil (optional, for CPU monitoring)
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cpu-miner.git
cd cpu-miner
```

2. Run the miner:
```bash
python minercpu2.py
```

## Configuration

Key parameters can be adjusted at the top of the script:

```python
NUM_PROCESSES = 40  # Number of mining processes (default: 2x CPU threads)
NONCE_LENGTH = 10   # Total length of nonce string
```

## Output

### Console Output
```
Process 0 starting mining...
Process 1 starting mining...
[...]

New best from Process 2! 5 leading zeros
Nonce: 0200000009
Hash: 00000f931c4ce4edff24b046833e272c6e989ce84dfe2d81d453f011b660e9c

Total Processed: 19,697,000 | Rate: 183,582/s | Best: 5 leading zeros
```

### File Output
Results are automatically saved to `best_hashes.txt`:
```
Process 2 - 5 zeros
Nonce: 0200000009
Hash: 00000f931c4ce4edff24b046833e272c6e989ce84dfe2d81d453f011b660e9c
--------------------------------------------------
```

## Performance Tuning

- `NUM_PROCESSES`: Set to 2x your CPU thread count for optimal performance
- For i9-10900:
  - Default: 40 processes (2 * 20 threads)
  - Adjust based on other system load
  - Monitor CPU usage to find optimal setting

## Clean Shutdown

The miner handles Ctrl+C (SIGINT) gracefully:
- Saves final statistics
- Cleanly terminates all processes
- Reports total hashes and average rate

## Nonce Generation Strategy

Each nonce is uniquely generated using:
1. Process ID (2 hex chars) - Ensures unique nonces across processes
2. Timestamp (4 hex chars) - Changes every second
3. Counter (4 hex chars) - Sequential within each process
4. Random suffix - Remaining characters

This ensures:
- No nonce collisions between processes
- Time-based variation
- Complete search space coverage

## Files

- `minercpu2.py` - Main mining script
- `best_hashes.txt` - Log of best hashes found

## Notes

- Designed for Windows systems with Python 3.7+
- Optimized for Intel i9-10900 architecture
- Process count can be adjusted for other CPUs
- Clean shutdown supported via Ctrl+C
- In the future would like to explore adding and handling multiple nonce generators

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
```

```markdown
# GPU SHA-256 Hash Miner

A high-performance CUDA-based SHA-256 hash mining tool optimized for NVIDIA RTX 3090. Features multi-stream processing, automatic compiler setup, and comprehensive GPU monitoring.

## System Requirements

### Hardware
- NVIDIA RTX 3090 GPU (or compatible NVIDIA GPU)
- 16GB+ System RAM recommended
- x64 processor

### Software Prerequisites
- Windows 10/11 64-bit
- Python 3.7+ (64-bit)
- CUDA Toolkit 11.8 or newer
- Visual Studio 2019 or 2022 with C++ development tools
- NVIDIA GPU Drivers (Latest Game Ready Driver recommended)

## Installation

1. Install Visual Studio (if not installed):
   ```bash
   # Download from:
   https://visualstudio.microsoft.com/downloads/
   # During installation, select:
   - Desktop development with C++
   - Windows 10/11 SDK
   ```

2. Install CUDA Toolkit:
   ```bash
   # Download from:
   https://developer.nvidia.com/cuda-11-8-0-download-archive
   # Select:
   - Windows
   - x86_64
   - Version 11.8 or newer
   ```

3. Create Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install Python dependencies:
   ```bash
   pip install numpy
   pip install cupy-cuda11x  # Replace '11x' with your CUDA version
   pip install pycuda
   pip install psutil
   ```

5. Clone repository:
   ```bash
   git clone https://github.com/yourusername/gpu-miner.git
   cd gpu-miner
   ```

## Configuration

### GPU Settings
```python
# RTX 3090 optimized settings in mine_hashes():
block_size = 256
blocks_per_sm = 32
iterations_per_thread = 2000
num_streams = 2
```

### CUDA Compiler Options
```python
options = [
    '-O3',
    '--use_fast_math',
    '--gpu-architecture=sm_86',  # RTX 3090 architecture
    '--ptxas-options=-v',
    '--maxrregcount=64',
    '-DFULL_UNROLL'
]
```

## Usage

1. Activate virtual environment:
```bash
.venv\Scripts\activate
```

2. Run the miner:
```bash
python gpu_miner.py
```

## Features

### Hardware Optimization
- CUDA stream overlapping
- L1 cache optimization
- Page-locked memory transfers
- Optimized register usage
- Multi-stream processing

### Performance Monitoring
- Real-time hash rate
- GPU utilization
- Power consumption
- Temperature monitoring
- Automatic result logging

### Compiler Setup
- Automatic Visual Studio detection
- CUDA toolkit integration
- Environment variable management
- Compiler validation

## Output Format

### Console Output
```
Starting optimized GPU mining...
Team Names: Reshanna Aasiyah Luis Gavin

GPU Configuration:
Device: NVIDIA GeForce RTX 3090
Number of SMs: 82
Warps per Block: 8
Blocks per SM: 32
Block Size: 256
Grid Size: 2624
Iterations per Thread: 2000

Hashes: 1,944,027,136,000 | Rate: 629,268,738/s | Power: 202.0W | Temp: 74.0°C | GPU: 100.0%
```

### File Output
Results are saved to:
- `best_hash.txt` - Best hashes found
- `mining_results.json` - Detailed mining statistics

## Troubleshooting

### Common Issues

1. CUDA Compiler Not Found:
```bash
# Ensure CUDA is in PATH:
echo %CUDA_PATH%
# Should show CUDA installation directory
```

2. Visual Studio Not Found:
```bash
# Verify Visual Studio installation:
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -property installationPath
```

3. GPU Not Detected:
```bash
# Check NVIDIA driver:
nvidia-smi
# Should show GPU information
```

### Performance Optimization

1. Temperature Management:
- Keep GPU below 80°C
- Ensure good airflow
- Monitor power usage

2. Memory Usage:
- Close other GPU applications
- Monitor system RAM
- Use performance power plan

## Files

- `gpu_miner.py` - Main mining script
- `best_hash.txt` - Hash results log
- `mining_results.json` - Mining statistics

## Performance Expectations

RTX 3090 typical performance:
- Hash Rate: 600-700 million hashes/second
- Power Usage: 200-300W
- Temperature: 70-80°C

## Contributing

Contributions welcome! Please ensure:
1. Code follows CUDA best practices
2. Memory management is optimized
3. Error handling is robust

## License

MIT License - See LICENSE file for details

## Acknowledgments

- NVIDIA CUDA Documentation
- PyCUDA Project
- Visual Studio Build Tools
```