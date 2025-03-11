import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List
from tqdm import tqdm

import cv2
from PIL import Image

###############################################################################
# Global references for the colorization nets
###############################################################################
MODELS_LOADED = False
NONLOCAL_NET  = None
COLOR_NET     = None
VGG_NET       = None

###############################################################################
# Paths from install.py
###############################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
nonlocal_ckpt = os.path.join(current_dir, "checkpoints", "video_moredata_l1", "nonlocal_net_iter_76000.pth")
color_ckpt    = os.path.join(current_dir, "checkpoints", "video_moredata_l1", "colornet_iter_76000.pth")
vgg_ckpt      = os.path.join(current_dir, "data", "vgg19_conv.pth")

###############################################################################
# Helper: get height,width from PIL or Torch
###############################################################################
def get_hw(image_or_tensor):
    """Return (height, width) from a PIL image or Torch [C,H,W] or [N,C,H,W]."""
    if isinstance(image_or_tensor, Image.Image):
        return image_or_tensor.height, image_or_tensor.width
    elif isinstance(image_or_tensor, torch.Tensor):
        # shape [C,H,W] => (H,W)
        # shape [N,C,H,W] => pick the first
        if image_or_tensor.dim()==4:
            if image_or_tensor.size(0)==0:
                return 0,0
            return image_or_tensor.shape[-2], image_or_tensor.shape[-1]
        elif image_or_tensor.dim()==3:
            return image_or_tensor.shape[-2], image_or_tensor.shape[-1]
        else:
            return 0,0
    else:
        return 0,0

###############################################################################
# CenterPadToRef: Pad or Crop to (ref_h, ref_w)
###############################################################################
import numpy as np
import torchvision.transforms as Tlib

class CenterPadToRef:
    """
    If the image is smaller in a dimension => pad
    If bigger => center-crop
    Expects a PIL image as input.
    """
    def __init__(self, ref_h, ref_w):
        self.ref_h = ref_h
        self.ref_w = ref_w
    def __call__(self, pil_img):
        arr = np.array(pil_img)
        h_old, w_old = arr.shape[:2]
        out_h, out_w = self.ref_h, self.ref_w
        # If smaller => pad, if bigger => crop
        # We'll do a naive approach
        if h_old==out_h and w_old==out_w:
            return pil_img
        # create an output
        c_old = 1 if arr.ndim==3 and arr.shape[2]==1 else 3
        out_arr = np.zeros((out_h, out_w, c_old), dtype=arr.dtype)
        # if bigger => crop
        if h_old>out_h:
            start_h = (h_old - out_h)//2
            arr = arr[start_h:start_h+out_h,:,:]
            h_old = out_h
        if w_old>out_w:
            start_w = (w_old - out_w)//2
            arr = arr[:, start_w:start_w+out_w, :]
            w_old = out_w
        # if smaller => pad
        off_h = (out_h - h_old)//2 if out_h>h_old else 0
        off_w = (out_w - w_old)//2 if out_w>w_old else 0
        h_end = off_h + h_old
        w_end = off_w + w_old
        out_arr[off_h:h_end, off_w:w_end] = arr
        return Image.fromarray(out_arr)

###############################################################################
# Convert PIL or Torch to Lab tensor (L => L-50)
###############################################################################
from skimage import color
def convert_any_image_to_lab_tensor(item):
    """
    If PIL => do np.array => rgb2lab => (L-50)
    If Torch => shape [C,H,W] or [N,C,H,W], convert to CPU np => rgb2lab => (L-50)
    """
    if isinstance(item, Image.Image):
        arr = np.array(item).astype(np.float32)
        if arr.ndim==2:
            arr = np.stack([arr,arr,arr], axis=-1)
        arr /= 255.0
        lab = color.rgb2lab(arr)
        lab_t = torch.from_numpy(lab.transpose(2,0,1))
        lab_t[0:1,:,:] = lab_t[0:1,:,:]-50.0
        return lab_t
    elif isinstance(item, torch.Tensor):
        # if shape [N,C,H,W], pick first
        if item.dim()==4:
            if item.size(0)==0:
                return None
            item = item[0]
        # now [C,H,W]
        arr = item.detach().cpu().float()
        # guess scale
        if arr.max()>2.0:
            arr = arr/255.0
        np_arr = arr.numpy()
        np_arr = np_arr.transpose(1,2,0)  # [H,W,C]
        lab = color.rgb2lab(np_arr)
        lab_t = torch.from_numpy(lab.transpose(2,0,1))
        lab_t[0:1,:,:] = lab_t[0:1,:,:]-50.0
        return lab_t
    else:
        return None

###############################################################################
# final step: batch_lab2rgb_transpose_mc, optional WLS
###############################################################################
from utils.util import batch_lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l
from models.NonlocalNet import WarpNet, VGG19_pytorch
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization

###############################################################################
# The Node
###############################################################################
NODE_CLASS_MAPPINGS = {}

class DeepExemplarVideoColorNodeTestPyLike:
    """
    Node that:
      - checks if reference_image is PIL or Torch
      - obtains (ref_h, ref_w)
      - optional center pad/crop each frame to (ref_h, ref_w)
      - convert to Lab => downsample => colorize => upsample => optional WLS => final batch_lab2rgb
      - frame_propagate => first frame is reference
      - WLS defaults: lambda=500, sigma=4
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "frame_propagate": ("BOOLEAN", {
                    "default": False,
                    "widget": "checkbox"
                }),
                "use_center_pad": ("BOOLEAN", {
                    "default": False,
                    "widget": "checkbox"
                }),
                "use_center_crop": ("BOOLEAN", {
                    "default": False,
                    "widget": "checkbox"
                }),
                "wls_filter_on": ("BOOLEAN", {
                    "default": True,
                    "widget": "checkbox"
                }),
                "lambda_value": ("FLOAT", {
                    "default": 500.0,
                    "min": 0,
                    "max": 2000
                }),
                "sigma_color": ("FLOAT", {
                    "default": 4.0,
                    "min": 0,
                    "max": 50
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_video"
    CATEGORY = "Video/DeepExemplar"

    def load_models_if_needed(self):
        global MODELS_LOADED, NONLOCAL_NET, COLOR_NET, VGG_NET
        if MODELS_LOADED:
            return
        print("[DeepExemplarVideoColorNodeTestPyLike] Loading models from local checkpoints...")

        nonlocal_net = WarpNet(1)
        color_net    = ColorVidNet(7)
        vgg_net      = VGG19_pytorch()

        nonlocal_net.load_state_dict(torch.load(nonlocal_ckpt, map_location="cuda"))
        color_net.load_state_dict(torch.load(color_ckpt, map_location="cuda"))
        vgg_net.load_state_dict(torch.load(vgg_ckpt, map_location="cuda"))

        nonlocal_net.cuda().eval()
        color_net.cuda().eval()
        vgg_net.cuda().eval()

        NONLOCAL_NET = nonlocal_net
        COLOR_NET    = color_net
        VGG_NET      = vgg_net
        MODELS_LOADED = True
        print("[DeepExemplarVideoColorNodeTestPyLike] Models loaded.")

    def colorize_video(self,
                       video_frames: Any,
                       reference_image: Any,
                       frame_propagate: bool,
                       use_center_pad: bool,
                       use_center_crop: bool,
                       wls_filter_on: bool,
                       lambda_value: float,
                       sigma_color: float
                       ) -> (List[np.ndarray],):

        self.load_models_if_needed()

        # if frame_propagate => first frame is reference
        # unify frames => list
        frames_list = []
        if isinstance(video_frames, list):
            frames_list = video_frames
        elif isinstance(video_frames, torch.Tensor):
            # shape [N,C,H,W] or [C,H,W]
            if video_frames.dim()==4:
                frames_list = [video_frames[i] for i in range(video_frames.size(0))]
            else:
                frames_list = [video_frames]
        elif isinstance(video_frames, Image.Image):
            frames_list = [video_frames]

        if frame_propagate and len(frames_list)>0:
            reference_image = frames_list[0]

        # get (ref_h, ref_w)
        ref_h, ref_w = get_hw(reference_image)
        if ref_h<1 or ref_w<1:
            print("[DeepExemplarVideoColorNodeTestPyLike] reference_image is zero dimension.")
            return ([],)

        # build optional pad/crop
        pad_transform = CenterPadToRef(ref_h, ref_w) if use_center_pad else None
        import torchvision.transforms as Tlib
        crop_transform = Tlib.CenterCrop((ref_h, ref_w)) if use_center_crop else None

        # Convert reference => lab
        # if it's PIL or Torch
        # if pad/crop => must be PIL
        # so if reference_image is Torch, we convert to PIL if needed
        ref_pil = reference_image
        if isinstance(reference_image, torch.Tensor):
            # convert to PIL
            ref_pil = self.torch_to_pil(reference_image)

        if pad_transform:
            ref_pil = pad_transform(ref_pil)
        if crop_transform:
            ref_pil = crop_transform(ref_pil)

        ref_lab_large = convert_any_image_to_lab_tensor(ref_pil).unsqueeze(0).cuda()
        ref_lab_small = F.interpolate(ref_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)

        with torch.no_grad():
            ref_l = ref_lab_small[:,0:1,:,:]+50.0
            ref_ab = ref_lab_small[:,1:3,:,:]
            ref_unnorm = torch.cat([ref_l, ref_ab], dim=1)
            ref_rgb = tensor_lab2rgb(ref_unnorm)
            features_B = VGG_NET(ref_rgb, ["r12","r22","r32","r42","r52"], preprocess=True)

        last_lab_predict = ref_lab_small.clone() if frame_propagate else None

        out_frames = []
        for idx, item in enumerate(tqdm(frames_list, desc="[DeepExemplar Node]")):
            # unify item => PIL
            if isinstance(item, torch.Tensor):
                pil_frame = self.torch_to_pil(item)
            else:
                pil_frame = item

            # pad/crop if toggles
            if pad_transform:
                pil_frame = pad_transform(pil_frame)
            if crop_transform:
                pil_frame = crop_transform(pil_frame)

            # convert to lab
            lab_large = convert_any_image_to_lab_tensor(pil_frame)
            if lab_large is None:
                print(f"[DeepExemplar] skipping frame {idx}, invalid.")
                continue
            lab_large = lab_large.unsqueeze(0).cuda()
            lab_small = F.interpolate(lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)

            IA_l = lab_small[:,0:1,:,:]
            if last_lab_predict is None:
                last_lab_predict = torch.zeros_like(lab_small)

            with torch.no_grad():
                I_current_ab_predict, _, _ = frame_colorization(
                    lab_small,
                    ref_lab_small,
                    last_lab_predict,
                    features_B,
                    VGG_NET,
                    NONLOCAL_NET,
                    COLOR_NET,
                    False,
                    1e-10
                )
            last_lab_predict = torch.cat([IA_l, I_current_ab_predict], dim=1)

            ab_up = F.interpolate(I_current_ab_predict, scale_factor=2.0, mode="bilinear", align_corners=False)*1.25

            if wls_filter_on:
                guide_l = (lab_large[:,0:1,:,:]+50.0).clamp(0,100)*255.0/100.0
                guide_np = guide_l[0,0].detach().cpu().numpy().astype(np.uint8)
                wlsf = cv2.ximgproc.createFastGlobalSmootherFilter(guide_np, lambda_value, sigma_color)
                a_np = ab_up[0,0].detach().cpu().numpy()
                b_np = ab_up[0,1].detach().cpu().numpy()
                a_filt = wlsf.filter(a_np)
                b_filt = wlsf.filter(b_np)
                a_t = torch.from_numpy(a_filt).unsqueeze(0).unsqueeze(0).cuda()
                b_t = torch.from_numpy(b_filt).unsqueeze(0).unsqueeze(0).cuda()
                ab_up = torch.cat([a_t,b_t], dim=1)

            out_rgb = batch_lab2rgb_transpose_mc(lab_large[:,0:1,:,:]+50.0, ab_up)
            np_rgb = out_rgb[0].astype(np.uint8)
            out_frames.append(np_rgb)

        return (out_frames,)

    def torch_to_pil(self, tensor: torch.Tensor):
        """
        Convert a Torch [C,H,W] or [N,C,H,W] in 0..1 or 0..255 to a PIL image.
        We'll just pick the first if [N,C,H,W].
        """
        if tensor.dim()==4:
            if tensor.size(0)==0:
                # return blank
                arr = np.zeros((1,1,3), dtype=np.uint8)
                return Image.fromarray(arr)
            tensor = tensor[0]
        arr = tensor.detach().cpu().float()
        if arr.max()>2.0:
            arr = arr/255.0
        arr_np = arr.numpy()  # shape [C,H,W]
        arr_np = arr_np.transpose(1,2,0)  # [H,W,C]
        arr_np = (arr_np*255).clip(0,255).astype(np.uint8)
        pil_img = Image.fromarray(arr_np)
        return pil_img


NODE_CLASS_MAPPINGS["DeepExemplarVideoColorNodeTestPyLike"] = DeepExemplarVideoColorNodeTestPyLike
