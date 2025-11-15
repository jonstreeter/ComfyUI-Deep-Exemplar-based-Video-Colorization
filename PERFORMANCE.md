# Performance Optimization Guide

This guide covers all available performance optimizations for ColorMNet and Deep Exemplar nodes.

## Quick Summary

| Optimization | Speedup | Nodes | Installation |
|--------------|---------|-------|--------------|
| **torch.compile** | 15-25% | All 4 nodes | Built-in (PyTorch 2.0+) |
| **SageAttention** | 20-30% | Deep Exemplar only | `pip install sageattention` |
| **CUDA Correlation** | 2-5x attention | ColorMNet only | Auto-install (requires CUDA toolkit) |
| **FP16** | ~2x | ColorMNet only | Built-in parameter |

---

## torch.compile (PyTorch 2.0+)

**Available for:** All 4 nodes (ColorMNet Video, ColorMNet Image, Deep Exemplar Video, Deep Exemplar Image)

### What is it?

PyTorch 2.0's JIT compiler that optimizes model execution through graph fusion, kernel optimization, and memory layout improvements.

### Performance Impact

- **Expected speedup:** 15-25% after warmup
- **First run:** Slower due to compilation overhead (30-60s additional time)
- **Subsequent runs:** Full speedup benefit
- **Quality:** Identical to non-compiled version

### Usage

Enable via node parameter:
```
use_torch_compile = True
```

### Requirements

- PyTorch 2.0 or later
- No additional installation needed
- Works on both CPU and GPU

### Troubleshooting

If compilation fails, the nodes automatically fall back to standard mode with a warning message. Check console output for details.

---

## SageAttention

**Available for:** Deep Exemplar Video and Image nodes only

### What is it?

INT8-quantized attention implementation that provides significant speedup for self-attention operations with minimal quality impact.

### Performance Impact

- **Expected speedup:** 20-30% on attention operations
- **Overall speedup:** ~15-20% for full pipeline
- **Quality:** Negligible difference due to INT8 quantization
- **VRAM:** Slightly reduced usage

### Installation

```bash
pip install sageattention
```

### Usage

Enable via node parameter:
```
use_sage_attention = True
```

### Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- `sageattention` package installed

### How it Works

SageAttention integrates into the `Self_Attn` layer in NonlocalNet:
- Automatically converts standard attention to INT8-quantized version
- Falls back to standard attention if unavailable or if errors occur
- No model retraining required

### Troubleshooting

If SageAttention fails to load:
1. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check package installation: `pip show sageattention`
3. The node will automatically fall back to standard attention with equivalent quality

---

## Automatic Performance Checks

ColorMNet automatically checks for and attempts to install performance optimizations on first run.

## CUDA Correlation Sampler

The **spatial_correlation_sampler** CUDA extension provides **2-5x faster** short-term attention processing.

### Requirements Check

On first run, ColorMNet will automatically check for:
- ✅ CUDA Toolkit installed and CUDA_HOME set
- ✅ NVIDIA CUDA compiler (nvcc)
- ✅ Visual Studio C++ compiler (cl.exe)
- ✅ PyTorch CUDA availability

### Automatic Installation

If all requirements are met, ColorMNet will:
1. Automatically compile and install spatial_correlation_sampler
2. Takes 2-5 minutes on first run
3. Enable optimized mode automatically

### Performance Modes

| Mode | Speed | Requirements |
|------|-------|--------------|
| **Optimized** | 100% (fastest) | CUDA extension installed |
| **Fallback** | 60-70% speed | No extra requirements |

**Note:** Both modes produce identical quality - the difference is only speed!

## Installation Instructions

If automatic installation fails, you can install manually:

### 1. Install CUDA Toolkit

Download from: https://developer.nvidia.com/cuda-toolkit

**Important:** Match your PyTorch CUDA version
- Check with: `python -c "import torch; print(torch.version.cuda)"`
- Install matching CUDA Toolkit (e.g., 12.8 for PyTorch CUDA 12.8)

### 2. Set CUDA_HOME Environment Variable

**Windows:**
```powershell
# System Properties → Advanced → Environment Variables → New
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

**Linux/Mac:**
```bash
export CUDA_HOME=/usr/local/cuda
```

### 3. Install Visual Studio Build Tools (Windows Only)

Download: https://visualstudio.microsoft.com/downloads/

**Select:** Desktop development with C++

### 4. Manual Installation

After setting up requirements:
```bash
pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git
```

### 5. Verify Installation

Restart ComfyUI and check the console output:
```
[ColorMNet] ✓ spatial_correlation_sampler already installed
[ColorMNet] 🚀 Optimized CUDA mode enabled (2-5x faster)
Correlation sampler: enabled (CUDA optimized)
```

## Troubleshooting

### "CUDA_HOME not set"

Set the environment variable to your CUDA installation directory.

### "Visual Studio C++ compiler not found"

Install Visual Studio Build Tools with C++ workload.

### "Compilation error"

Ensure CUDA Toolkit version matches your PyTorch CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
```

### Still not working?

The fallback mode works perfectly! You'll get:
- Same quality results
- 20-40% slower processing
- No additional setup required

## Performance Comparison

**Example: 196 frames @ 480x640 on RTX 5090**

| Mode | Time | FPS |
|------|------|-----|
| Optimized | ~45s | 4.4 |
| Fallback | ~65s | 3.0 |

The speed difference is noticeable but not critical for most use cases.

---

## Combining Optimizations

You can enable multiple optimizations simultaneously for maximum performance:

### ColorMNet Maximum Performance

```
use_fp16 = True
use_torch_compile = True
memory_mode = "low_memory"  # or "balanced"
```

**Expected combined speedup:** 2.5-3x vs baseline (FP32, no compile)

### Deep Exemplar Maximum Performance

```
use_torch_compile = True
use_sage_attention = True
use_half_resolution = True
frame_propagate = True
```

**Expected combined speedup:** 1.4-1.6x vs baseline (no optimizations)

### Performance Benchmark Example

**Test:** 240 frames @ 768x432 on RTX 5090

| Configuration | ColorMNet | Deep Exemplar |
|--------------|-----------|---------------|
| Baseline (no opts) | ~75s (3.2 FPS) | ~68s (3.5 FPS) |
| FP16 only | ~45s (5.3 FPS) | N/A |
| torch.compile only | ~58s (4.1 FPS) | ~55s (4.4 FPS) |
| All optimizations | ~36s (6.7 FPS) | ~47s (5.1 FPS) |

**Note:** Actual performance varies by GPU, resolution, and content.

---

## Optimization Recommendations by GPU

### Low-end (6-8GB VRAM)
**ColorMNet:**
- `use_fp16 = True`
- `use_torch_compile = True`
- `memory_mode = "low_memory"`

**Deep Exemplar:**
- `use_half_resolution = True`
- `use_torch_compile = True`
- `use_sage_attention = True`

### Mid-range (8-16GB VRAM)
**ColorMNet:**
- `use_fp16 = True`
- `use_torch_compile = True`
- `memory_mode = "balanced"`

**Deep Exemplar:**
- `use_half_resolution = True` (optional)
- `use_torch_compile = True`
- `use_sage_attention = True`
- `wls_filter_on = True`

### High-end (16GB+ VRAM)
**ColorMNet:**
- `use_fp16 = False` (best quality)
- `use_torch_compile = True`
- `memory_mode = "high_quality"`

**Deep Exemplar:**
- `use_half_resolution = False`
- `use_torch_compile = True`
- `use_sage_attention = True`
- `wls_filter_on = True`

---

## Quality Impact Summary

| Optimization | Quality Impact |
|--------------|----------------|
| torch.compile | **None** - Identical output |
| SageAttention | **Negligible** - INT8 quantization barely visible |
| FP16 | **Minimal** - Slight numerical differences |
| CUDA Correlation | **None** - Identical output |

**Recommendation:** Enable all applicable optimizations for your hardware. The quality impact is minimal to none while providing significant speed improvements.
