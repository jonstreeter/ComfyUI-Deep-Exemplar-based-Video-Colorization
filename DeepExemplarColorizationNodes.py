"""
DeepExemplarColorizationNodes.py

This module provides two ComfyUI nodes for Deep Exemplar-based Colorization:
  - DeepExColorImageNode: Colorizes a grayscale image using a reference color image.
  - DeepExColorVideoNode: Colorizes a sequence of grayscale video frames using a reference color image.
  
Both nodes follow the test.py logic from the base Deep Exemplar project.
"""

import sys
import subprocess
import importlib
import cv2

# Check for cv2.ximgproc.createFastGlobalSmootherFilter and attempt installation if needed.
if not hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter'):
    print("cv2.ximgproc.createFastGlobalSmootherFilter not found. Attempting to install opencv-contrib-python...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "opencv-contrib-python"])
        importlib.reload(cv2)
        if hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter'):
            print("Installation successful and function is now available.")
        else:
            print("Installation attempted, but function is still unavailable. Please install or upgrade opencv-contrib-python (e.g., 'pip install --upgrade opencv-contrib-python') and restart ComfyUI.")
    except Exception as e:
        print(f"Failed to install opencv-contrib-python: {e}\nPlease install opencv-contrib-python manually and restart ComfyUI.")

WLS_FILTER_AVAILABLE = hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter')

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as Tlib
from tqdm import tqdm as console_tqdm

# Import ComfyUI's internal ProgressBar for GUI progress updates.
from comfy.utils import ProgressBar

from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor
from utils.util import batch_lab2rgb_transpose_mc, uncenter_l, tensor_lab2rgb
from models.FrameColor import frame_colorization

MODELS_LOADED = False
NONLOCAL_NET = None
COLOR_NET = None
VGG_NET = None

def load_models_if_needed():
    global MODELS_LOADED, NONLOCAL_NET, COLOR_NET, VGG_NET
    if MODELS_LOADED:
        return
    print("[DeepExemplar] Loading model checkpoints...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nonlocal_ckpt = os.path.join(script_dir, "checkpoints", "video_moredata_l1", "nonlocal_net_iter_76000.pth")
    color_ckpt = os.path.join(script_dir, "checkpoints", "video_moredata_l1", "colornet_iter_76000.pth")
    vgg_ckpt = os.path.join(script_dir, "data", "vgg19_conv.pth")
    from models.NonlocalNet import WarpNet, VGG19_pytorch
    from models.ColorVidNet import ColorVidNet
    NONLOCAL_NET = WarpNet(1).cuda().eval()
    COLOR_NET = ColorVidNet(7).cuda().eval()
    VGG_NET = VGG19_pytorch().cuda().eval()
    NONLOCAL_NET.load_state_dict(torch.load(nonlocal_ckpt, map_location="cuda"))
    COLOR_NET.load_state_dict(torch.load(color_ckpt, map_location="cuda"))
    VGG_NET.load_state_dict(torch.load(vgg_ckpt, map_location="cuda"))
    MODELS_LOADED = True
    print("[DeepExemplar] Models loaded successfully.")

def adjust_target_size(input_h: int, input_w: int, min_h: int = 64, min_w: int = 64, downsample_multiple: int = 32) -> (int, int):
    final_h = max(input_h, min_h)
    final_w = max(input_w, min_w)
    remainder_h = final_h % downsample_multiple
    if remainder_h != 0:
        down_dist = remainder_h
        up_dist = downsample_multiple - remainder_h
        if up_dist < down_dist:
            final_h += up_dist
        else:
            if final_h - down_dist >= min_h:
                final_h -= down_dist
            else:
                final_h += up_dist
    remainder_w = final_w % downsample_multiple
    if remainder_w != 0:
        down_dist = remainder_w
        up_dist = downsample_multiple - remainder_w
        if up_dist < down_dist:
            final_w += up_dist
        else:
            if final_w - down_dist >= min_w:
                final_w -= down_dist
            else:
                final_w += up_dist
    return final_h, final_w

def build_test_py_transform(image_size=(432,768)):
    return Tlib.Compose([
        CenterPad(image_size),
        Tlib.CenterCrop(image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ])

def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
    if tensor_img.dim() == 4 and tensor_img.size(0) == 1:
        tensor_img = tensor_img.squeeze(0)
    if tensor_img.dim() == 3 and tensor_img.shape[-1] in (1, 3):
        tensor_img = tensor_img.permute(2, 0, 1)
    tensor_img = tensor_img.detach().cpu().clamp(0,1)
    np_img = (tensor_img.numpy()*255).astype(np.uint8)
    np_img = np.transpose(np_img, (1,2,0))
    if np_img.shape[2] == 1:
        return Image.fromarray(np_img[:,:,0], mode="L")
    else:
        return Image.fromarray(np_img, mode="RGB")

class DeepExColorImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_to_colorize": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "target_width": ("INT", {"default":768, "min":16, "max":4096}),
                "target_height": ("INT", {"default":432, "min":16, "max":4096}),
                "wls_filter_on": ("BOOLEAN", {"default":True}),
                "lambda_value": ("FLOAT", {"default":500.0, "min":0, "max":2000}),
                "sigma_color": ("FLOAT", {"default":4.0, "min":0, "max":50}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_image"
    CATEGORY = "DeepExemplar/Image"

    def colorize_image(self, image_to_colorize: torch.Tensor, reference_image: torch.Tensor,
                         target_width: int, target_height: int, wls_filter_on: bool,
                         lambda_value: float, sigma_color: float) -> (list,):
        load_models_if_needed()
        # Note: adjust_target_size expects (height, width); here we pass target_height, target_width.
        final_h, final_w = adjust_target_size(target_height, target_width, 64, 64, 32)
        print(f"[DeepExColorImageNode] Adjusted size: ({target_width}, {target_height}) -> ({final_w}, {final_h})")
        transform = build_test_py_transform((final_h, final_w))
        src_pil = tensor_to_pil(image_to_colorize)
        ref_pil = tensor_to_pil(reference_image)
        src_lab_full = transform(src_pil).unsqueeze(0).cuda()
        ref_lab_full = transform(ref_pil).unsqueeze(0).cuda()
        src_half = F.interpolate(src_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        ref_half = F.interpolate(ref_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        I_ref_l = ref_half[:, 0:1, :, :]
        I_ref_ab = ref_half[:, 1:3, :, :]
        I_ref_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_ref_l), I_ref_ab), dim=1)).to(ref_half.device)
        with torch.no_grad():
            features_B = VGG_NET(I_ref_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            I_ab_predict, _, _ = frame_colorization(
                src_half, ref_half, torch.zeros_like(src_half),
                features_B, VGG_NET, NONLOCAL_NET, COLOR_NET,
                feature_noise=0, temperature=1e-10
            )
        ab_up = F.interpolate(I_ab_predict, scale_factor=2.0, mode="bilinear", align_corners=False) * 1.25
        if wls_filter_on and WLS_FILTER_AVAILABLE:
            guide_l = uncenter_l(src_lab_full[:, 0:1, :, :]) * (255.0/100.0)
            guide_np = guide_l[0, 0].detach().cpu().numpy().astype(np.uint8)
            wlsf = cv2.ximgproc.createFastGlobalSmootherFilter(guide_np, lambda_value, sigma_color)
            a_np = ab_up[0, 0].detach().cpu().numpy()
            b_np = ab_up[0, 1].detach().cpu().numpy()
            a_filt = wlsf.filter(a_np)
            b_filt = wlsf.filter(b_np)
            a_t = torch.from_numpy(a_filt).unsqueeze(0).unsqueeze(0).to(ab_up.device)
            b_t = torch.from_numpy(b_filt).unsqueeze(0).unsqueeze(0).to(ab_up.device)
            ab_up = torch.cat([a_t, b_t], dim=1)
        else:
            if wls_filter_on and not WLS_FILTER_AVAILABLE:
                print("[DeepExColorImageNode] Warning: 'createFastGlobalSmootherFilter' not available; skipping filtering. To fix, install or upgrade opencv-contrib-python (e.g., 'pip install --upgrade opencv-contrib-python') and restart ComfyUI.")
        out_np = batch_lab2rgb_transpose_mc(src_lab_full[:, 0:1, :, :], ab_up)
        out_tensor = torch.from_numpy(out_np).float().div(255.0)
        return ([out_tensor],)

class DeepExColorVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "frame_propagate": ("BOOLEAN", {"default":True}),
                "use_half_resolution": ("BOOLEAN", {"default":True}),
                "target_width": ("INT", {"default":768, "min":16, "max":4096}),
                "target_height": ("INT", {"default":432, "min":16, "max":4096}),
                "wls_filter_on": ("BOOLEAN", {"default":True}),
                "lambda_value": ("FLOAT", {"default":500.0, "min":0, "max":2000}),
                "sigma_color": ("FLOAT", {"default":4.0, "min":0, "max":50}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_video"
    CATEGORY = "DeepExemplar/Video"

    def colorize_video(self, video_frames: any, reference_image: torch.Tensor,
                       frame_propagate: bool, use_half_resolution: bool,
                       target_width: int, target_height: int, wls_filter_on: bool,
                       lambda_value: float, sigma_color: float) -> (list,):
        load_models_if_needed()
        final_h, final_w = adjust_target_size(target_height, target_width, 64, 64, 32)
        print(f"[DeepExColorVideoNode] Adjusted size: ({target_width}, {target_height}) -> ({final_w}, {final_h})")
        if wls_filter_on and not WLS_FILTER_AVAILABLE:
            print("[DeepExColorVideoNode] Warning: 'createFastGlobalSmootherFilter' not available; skipping filtering. To fix, install or upgrade opencv-contrib-python (e.g., 'pip install --upgrade opencv-contrib-python') and restart ComfyUI.")
        if isinstance(video_frames, list):
            frames_list = video_frames
        elif isinstance(video_frames, torch.Tensor):
            if video_frames.dim() == 4:
                frames_list = [video_frames[i] for i in range(video_frames.size(0))]
            else:
                frames_list = [video_frames]
        else:
            frames_list = [video_frames]
        print(f"[DeepExColorVideoNode] Number of frames: {len(frames_list)}")
        transform = build_test_py_transform((final_h, final_w))
        ref_pil = tensor_to_pil(reference_image)
        ref_lab_full = transform(ref_pil).unsqueeze(0).cuda()
        if use_half_resolution:
            ref_half = F.interpolate(ref_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        else:
            ref_half = ref_lab_full.clone()
        I_ref_l = ref_half[:, 0:1, :, :]
        I_ref_ab = ref_half[:, 1:3, :, :]
        I_ref_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_ref_l), I_ref_ab), dim=1)).to(ref_half.device)
        with torch.no_grad():
            features_B = VGG_NET(I_ref_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        out_frames = []
        last_lab = None
        num_frames = len(frames_list)
        gui_pbar = ProgressBar(num_frames)
        console_pbar = console_tqdm(total=num_frames, desc="[DeepExColorVideoNode] Console Progress")
        for i, frm in enumerate(frames_list):
            frame_pil = tensor_to_pil(frm)
            frm_lab_full = transform(frame_pil).unsqueeze(0).cuda()
            if use_half_resolution:
                frm_half = F.interpolate(frm_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
            else:
                frm_half = frm_lab_full.clone()
            if i == 0:
                with torch.no_grad():
                    I_ab, _, _ = frame_colorization(
                        frm_half, ref_half, torch.zeros_like(frm_half),
                        features_B, VGG_NET, NONLOCAL_NET, COLOR_NET,
                        feature_noise=0, temperature=1e-10
                    )
                if frame_propagate:
                    last_lab = torch.cat([frm_half[:, 0:1, :, :], I_ab], dim=1)
            else:
                if frame_propagate and last_lab is not None:
                    with torch.no_grad():
                        I_ab, _, _ = frame_colorization(
                            frm_half, ref_half, last_lab,
                            features_B, VGG_NET, NONLOCAL_NET, COLOR_NET,
                            feature_noise=0, temperature=1e-10
                        )
                    last_lab = torch.cat([frm_half[:, 0:1, :, :], I_ab], dim=1)
                else:
                    with torch.no_grad():
                        I_ab, _, _ = frame_colorization(
                            frm_half, ref_half, torch.zeros_like(frm_half),
                            features_B, VGG_NET, NONLOCAL_NET, COLOR_NET,
                            feature_noise=0, temperature=1e-10
                        )
                    last_lab = None
            if use_half_resolution:
                ab_up = F.interpolate(I_ab, scale_factor=2.0, mode="bilinear", align_corners=False) * 1.25
            else:
                ab_up = I_ab.clone()
            if wls_filter_on and WLS_FILTER_AVAILABLE:
                guide_l = uncenter_l(frm_lab_full[:, 0:1, :, :]) * (255.0/100.0)
                guide_np = guide_l[0, 0].detach().cpu().numpy().astype(np.uint8)
                wlsf = cv2.ximgproc.createFastGlobalSmootherFilter(guide_np, lambda_value, sigma_color)
                a_np = ab_up[0, 0].detach().cpu().numpy()
                b_np = ab_up[0, 1].detach().cpu().numpy()
                a_filt = wlsf.filter(a_np)
                b_filt = wlsf.filter(b_np)
                a_t = torch.from_numpy(a_filt).unsqueeze(0).unsqueeze(0).to(ab_up.device)
                b_t = torch.from_numpy(b_filt).unsqueeze(0).unsqueeze(0).to(ab_up.device)
                ab_up = torch.cat([a_t, b_t], dim=1)
            out_np = batch_lab2rgb_transpose_mc(frm_lab_full[:, 0:1, :, :], ab_up)
            out_t = torch.from_numpy(out_np).float().div(255.0)
            out_frames.append(out_t)
            gui_pbar.update_absolute(i+1, num_frames)
            console_pbar.update(1)
            if hasattr(self, 'set_progress'):
                self.set_progress(gui_pbar.progress)
        console_pbar.close()
        return (out_frames,)

NODE_CLASS_MAPPINGS = {
    "DeepExColorImageNode": DeepExColorImageNode,
    "DeepExColorVideoNode": DeepExColorVideoNode
}
