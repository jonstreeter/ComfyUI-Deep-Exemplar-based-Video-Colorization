# ⚠️ Repository Moved

**This repository has been renamed and relocated.**

---

## 🔗 New Repository Location

**Old Name:** `ComfyUI-Deep-Exemplar-based-Video-Colorization`

**New Name:** `ComfyUI-Reference-Based-Video-Colorization`

### ➡️ Please visit the new repository:

# https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization

---

## 🎯 Why Was It Renamed?

The old name "Deep-Exemplar-based" was misleading because this toolkit now includes:

✅ **Multiple Colorization Methods:**
- ColorMNet (2024) - Modern DINOv2-based approach
- Deep Exemplar (2019) - Classic CVPR method

✅ **Multiple Feature Encoders:**
- VGG19 (original)
- DINOv2 (ViT-S/B/L)
- CLIP (ViT-B)

✅ **Advanced Post-Processing:**
- color-matcher
- WLS filter
- Guided filter
- Bilateral filter

The new name **"Reference-Based Video Colorization"** better reflects this comprehensive toolkit.

---

## 📦 Installation

**Use the new repository:**

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git
cd ComfyUI-Reference-Based-Video-Colorization/
pip install -r requirements.txt
```

---

## ⚙️ If You Already Cloned the Old Repository

Update your local repository to point to the new location:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Deep-Exemplar-based-Video-Colorization/

# Update the remote URL
git remote set-url origin https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git

# Pull the latest changes
git pull
```

Or start fresh:

```bash
# Remove old directory
cd ComfyUI/custom_nodes/
rm -rf ComfyUI-Deep-Exemplar-based-Video-Colorization/

# Clone new repository
git clone https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git
```

---

## 🔗 Links

- **New Repository:** https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization
- **Issues:** https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization/issues
- **Documentation:** https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization#readme

---

## ✨ What's New in the Latest Release

- **4 colorization nodes** (2 video, 2 image)
- **Auto-installer** for dependencies
- **Modern encoders** (DINOv2, CLIP)
- **Advanced post-processing** pipeline
- **Performance optimizations** (torch.compile, FP16)
- **Comprehensive documentation**

---

**⭐ Please update your bookmarks and star the new repository!**

https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization
