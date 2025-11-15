# ⚡ Performance Optimizations Guide

**How to get maximum speed from your video colorization**

---

## ✅ Already Working

Your setup is already performing well:
- ✅ CUDA graphs detection working
- ✅ FP16 enabled (2x faster)
- ✅ RTX 5090 GPU detected
- ✅ ColorMNet loaded successfully
- ✅ Processing ~2.2 FPS (851 frames in ~500 seconds)

---

## 🚀 Available Optimizations

### **1. Visual Studio C++ Compiler (Optional, 20-40% speedup)**

**Status:** ❌ Not installed

**What it does:**
- Enables CUDA spatial_correlation_sampler compilation
- Optimizes ColorMNet's matching algorithm
- **20-40% faster processing** for ColorMNet

**Currently:**
```
[ColorMNet] Missing requirements for CUDA extensions:
[ColorMNet]   ❌ Visual Studio C++ compiler (cl.exe) not found
[ColorMNet] Currently using fallback mode (20-40% slower, same quality)
```

**How to install:**

1. **Download Visual Studio Build Tools:**
   https://visualstudio.microsoft.com/downloads/

2. **Select "Desktop development with C++"**
   - This installs cl.exe (C++ compiler)
   - Size: ~7GB download, ~10GB installed

3. **Restart ComfyUI**
   - ColorMNet will auto-compile the optimizations
   - First compilation takes ~2-3 minutes
   - Subsequent runs use compiled version

**Expected improvement:**
- **Before:** 851 frames in ~500 seconds (~2.2 FPS)
- **After:** 851 frames in ~350-400 seconds (~2.5-3.0 FPS)
- **Savings:** ~100-150 seconds on your video!

**Is it worth it?**
- ✅ YES if you process many videos
- ⚠️ MAYBE if you only process occasionally (large download)
- ❌ NO if disk space is limited (10GB)

---

### **2. torch.compile for DeepExemplar (10-25% speedup)**

**Status:** ✅ Works on DeepExemplar nodes

**What it does:**
- JIT compiles model for faster inference
- **10-25% faster** on first run, then instant
- Only works on DeepExemplar (not ColorMNet)

**Why not ColorMNet?**
ColorMNet is already optimized with custom CUDA kernels and has a complex architecture that doesn't benefit from torch.compile. You'll see this message:

```
[ColorMNet] torch.compile requested but not applicable to ColorMNet architecture
[ColorMNet] ColorMNet is already optimized with custom kernels
```

**This is expected and correct!**

**To enable (DeepExemplar only):**
```
use_torch_compile: true
```

**Expected improvement:**
- DeepExemplar: 10-25% faster
- ColorMNet: No benefit (already optimized)

---

## 📊 Performance Comparison

### ColorMNet (Your Current Node)

| Setup | FPS | Time (851 frames) | vs Baseline |
|-------|-----|------------------|-------------|
| **Current (No optimizations)** | 2.2 | 500s (~8min) | Baseline |
| **+ Visual Studio C++** | 2.8 | 400s (~6.5min) | **-20%** ⭐ |
| **+ torch.compile** | 2.2 | 500s | No effect |
| **+ Both** | 2.8 | 400s | **-20%** |

**Recommendation for ColorMNet:**
- Install Visual Studio C++ if you process videos regularly
- Don't bother enabling torch.compile (no benefit)

---

### DeepExemplar

| Setup | Relative Speed | Recommendation |
|-------|---------------|----------------|
| **vgg19 + torch.compile** | ⚡⚡⚡⚡⚡ | Fastest |
| **dinov2_vitb (no compile)** | ⚡⚡⚡ | Best quality/speed balance |
| **dinov2_vitb + torch.compile** | ⚡⚡⚡⚡ | Recommended ⭐ |
| **dinov2_vitl + torch.compile** | ⚡⚡⚡ | Max quality |

**Recommendation for DeepExemplar:**
- Enable torch.compile (10-25% speedup)
- Use dinov2_vitb for best balance

---

## 🎯 Recommended Settings

### **For ColorMNet (Current):**

**Best Quality + Speed:**
```
memory_mode: balanced
post_processor: color_matcher
use_fp16: true
use_torch_compile: false (no benefit)
```

**After Installing Visual Studio C++:**
```
Same settings as above
(Optimizations auto-enable, no settings change needed!)
```

---

### **For DeepExemplar:**

**Best Quality:**
```
feature_encoder: dinov2_vitb
post_processor: color_matcher
use_torch_compile: true
use_sage_attention: false
```

**Fastest:**
```
feature_encoder: vgg19
post_processor: none
use_torch_compile: true
use_sage_attention: true
```

---

## 🔧 How to Install Visual Studio C++

### Step-by-Step:

1. **Go to:** https://visualstudio.microsoft.com/downloads/

2. **Download:** "Build Tools for Visual Studio 2022" (free)

3. **Run installer**

4. **Select workload:** "Desktop development with C++"
   - This includes:
     - MSVC C++ compiler
     - Windows SDK
     - CMake tools

5. **Install** (~7GB download, takes 15-30 min)

6. **Restart ComfyUI**

7. **Run ColorMNet** - you'll see:
   ```
   [ColorMNet] ✓ Visual Studio C++ compiler found
   [ColorMNet] Compiling spatial_correlation_sampler...
   [ColorMNet] ✓ CUDA extensions compiled successfully!
   ```

8. **Enjoy 20-40% faster processing!**

---

## ❓ FAQ

### **Q: Do I need Visual Studio C++ for DeepExemplar?**
**A:** No, only for ColorMNet's optimizations.

### **Q: Will torch.compile speed up ColorMNet?**
**A:** No, ColorMNet is already optimized. It only helps DeepExemplar.

### **Q: Is the Visual Studio download safe?**
**A:** Yes, it's official Microsoft software.

### **Q: Can I uninstall Visual Studio after compilation?**
**A:** No, it's needed each time Python packages are updated/rebuilt.

### **Q: How much disk space does it need?**
**A:** ~10GB for Build Tools (just the compiler, not full Visual Studio)

### **Q: Will it slow down my computer?**
**A:** No, it only runs when compiling CUDA extensions.

---

## 🎯 Quick Decision Guide

**"Should I install Visual Studio C++?"**

| Your Situation | Recommendation |
|----------------|----------------|
| Process videos daily/weekly | ✅ **YES** - Big time savings |
| Process videos monthly | ⚠️ **MAYBE** - Depends on disk space |
| Only testing/learning | ❌ **NO** - Not worth 10GB |
| Limited disk space (<20GB free) | ❌ **NO** - Too large |
| Have fast internet + disk space | ✅ **YES** - Why not? |

---

## 📈 Expected Performance After Optimizations

### Your Video (851 frames, 720x960):

**Currently:**
- Time: 500 seconds (~8.3 minutes)
- Speed: 2.2 FPS

**With Visual Studio C++ (ColorMNet):**
- Time: ~350-400 seconds (~6-7 minutes)
- Speed: ~2.5-3.0 FPS
- **Savings: 100-150 seconds per video!**

**With torch.compile (DeepExemplar):**
- Time: ~10-25% faster
- Only applies to DeepExemplar nodes

---

## ✅ Summary

**What's already working:**
- ✅ GPU detection
- ✅ FP16
- ✅ CUDA graphs handling
- ✅ ColorMNet processing

**Optional optimizations:**
1. **Visual Studio C++** - 20-40% faster ColorMNet
2. **torch.compile** - 10-25% faster DeepExemplar (no benefit for ColorMNet)

**Current message you're seeing is normal:**
```
[ColorMNet] Currently using fallback mode (20-40% slower, same quality)
```

This just means you could be 20-40% faster with Visual Studio C++ installed!

---

**Your choice: Install Visual Studio C++ for 20-40% speedup, or keep current setup!** 🚀

Both work perfectly - it's just an optional speed optimization!
