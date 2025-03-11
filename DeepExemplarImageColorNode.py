import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Any, List

###############################################################################
# Optional: fix relative paths for "data/vgg19_gray.pth" if used
###############################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
orig_torch_load = torch.load
def custom_torch_load(path, *args, **kwargs):
    if path == "data/vgg19_gray.pth":
        path = os.path.join(current_dir, "data", "vgg19_gray.pth")
    return orig_torch_load(path, *args, **kwargs)
torch.load = custom_torch_load
###############################################################################

# Import from your local code
import importlib.abc
import importlib.machinery

class CustomImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, base_path):
        self.base_path = base_path
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('utils.') or fullname == 'utils':
            parts = fullname.split('.')
            rel_path = os.path.join(*parts) + ".py"
            full_path = os.path.join(self.base_path, rel_path)
            if os.path.exists(full_path):
                return importlib.machinery.ModuleSpec(
                    name=fullname,
                    loader=importlib.machinery.SourceFileLoader(fullname, full_path),
                    origin=full_path
                )
        return None

sys.meta_path.insert(0, CustomImportFinder(current_dir))

from models.NonlocalNet import WarpNet, VGG19_pytorch
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from utils.util import batch_lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

import cv2
try:
    _ = cv2.ximgproc
except AttributeError:
    print("[DeepExemplarImageColorNode] Warning: ximgproc not found. WLS filter may be unavailable.")

# ComfyUI node registry
NODE_CLASS_MAPPINGS = {}

# Paths to your model checkpoints
ROOT_DIR = os.path.dirname(__file__)
NONLOCAL_TEST_PATH = os.path.join(ROOT_DIR, "checkpoints", "video_moredata_l1", "nonlocal_net_iter_76000.pth")
COLOR_TEST_PATH    = os.path.join(ROOT_DIR, "checkpoints", "video_moredata_l1", "colornet_iter_76000.pth")
VGG_PATH           = os.path.join(ROOT_DIR, "data", "vgg19_conv.pth")

# Global references
MODELS_LOADED = False
NONLOCAL_NET  = None
COLOR_NET     = None
VGG_NET       = None

def load_models_if_needed():
    global MODELS_LOADED, NONLOCAL_NET, COLOR_NET, VGG_NET
    if MODELS_LOADED:
        return
    print("[DeepExemplarImageColorNode] Loading models...")
    nonlocal_net = WarpNet(1)
    color_net    = ColorVidNet(7)
    vgg_net      = VGG19_pytorch()

    nonlocal_net.load_state_dict(torch.load(NONLOCAL_TEST_PATH, map_location="cuda"))
    color_net.load_state_dict(torch.load(COLOR_TEST_PATH, map_location="cuda"))
    vgg_net.load_state_dict(torch.load(VGG_PATH, map_location="cuda"))

    nonlocal_net.cuda().eval()
    color_net.cuda().eval()
    vgg_net.cuda().eval()

    MODELS_LOADED = True
    NONLOCAL_NET  = nonlocal_net
    COLOR_NET     = color_net
    VGG_NET       = vgg_net
    print("[DeepExemplarImageColorNode] Models loaded.")

###############################################################################
# Utilities to convert Torch Tensors <-> PIL
###############################################################################
def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Expects shape [C,H,W]. Converts to PIL (RGB).
    Clamps to [0..1], then scales to [0..255].
    """
    image_tensor = image_tensor.detach().cpu().clamp(0,1)
    np_img = (image_tensor.numpy() * 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1,2,0))  # [H,W,C]
    pil_img = Image.fromarray(np_img, mode="RGB")
    return pil_img

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Converts a PIL (RGB) to Torch float tensor [C,H,W], scaled 0..1.
    """
    return T.ToTensor()(pil_img)

###############################################################################
# Transforms from test.py
###############################################################################
class ComposeTransforms:
    def __init__(self, steps):
        self.steps = steps
    def __call__(self, img):
        for step in self.steps:
            img = step(img)
        return img

def build_transform(image_size=(432,768)):
    return ComposeTransforms([
        CenterPad(image_size),
        T.CenterCrop(image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ])

###############################################################################
# The Node
###############################################################################
class DeepExemplarImageColorNode:
    """
    Accepts a single image (Torch Tensor or PIL) + optional boolean WLS, etc.
    Outputs a single colorized image (Torch Tensor).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "wls_filter_on": ("BOOLEAN", {
                    "default": True,
                    "forceInput": False,
                    "widget": "checkbox"
                }),
                "lambda_value": ("FLOAT", {"default": 500.0, "min":0, "max":2000}),
                "sigma_color": ("FLOAT", {"default":4.0, "min":0, "max":50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_image"
    CATEGORY = "DeepExemplar"

    def colorize_image(self,
                       input_image: Any,
                       wls_filter_on: bool,
                       lambda_value: float,
                       sigma_color: float) -> (torch.Tensor,):

        load_models_if_needed()

        # 1) Convert input to PIL
        pil_img = None
        if isinstance(input_image, torch.Tensor):
            # shape [C,H,W]
            pil_img = tensor_to_pil(input_image)
        elif isinstance(input_image, Image.Image):
            pil_img = input_image
        else:
            print("[DeepExemplarImageColorNode] Unsupported image type. Returning empty.")
            return (torch.zeros((3,64,64)),)

        # 2) Prepare transforms
        transform = build_transform((432,768))
        input_lab = transform(pil_img).unsqueeze(0).cuda()  # shape [1,3,H,W]
        input_small = F.interpolate(input_lab, scale_factor=0.5, mode="bilinear")

        # We'll treat the reference as itself for a single image (no external ref).
        ref_tensor_small = input_small.clone()

        # Precompute VGG features
        with torch.no_grad():
            I_l  = input_small[:,0:1,:,:]
            I_ab = input_small[:,1:3,:,:]
            I_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_l), I_ab), dim=1))
            features_B = VGG_NET(I_rgb, ["r12","r22","r32","r42","r52"], preprocess=True)

        # colorize
        with torch.no_grad():
            I_current_ab_predict, _, _ = frame_colorization(
                I_current_lab=input_small,
                I_reference_lab=ref_tensor_small,
                I_last_lab=torch.zeros_like(input_small),
                features_B=features_B,
                vggnet=VGG_NET,
                nonlocal_net=NONLOCAL_NET,
                color_net=COLOR_NET,
                feature_noise=0,
                temperature=1e-10,
            )

        # Upscale ab
        ab_up = F.interpolate(I_current_ab_predict, scale_factor=2.0, mode="bilinear") * 1.25

        # optional WLS
        if wls_filter_on:
            guide_image = uncenter_l(input_lab[:,0:1,:,:]) * 255/100.0
            guide_numpy = guide_image[0,0].detach().cpu().numpy().astype(np.uint8)
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_numpy, lambda_value, sigma_color
            )
            a_channel = ab_up[0,0].detach().cpu().numpy()
            b_channel = ab_up[0,1].detach().cpu().numpy()
            a_filtered = wls_filter.filter(a_channel)
            b_filtered = wls_filter.filter(b_channel)
            a_filt_t = torch.from_numpy(a_filtered).unsqueeze(0).unsqueeze(0)
            b_filt_t = torch.from_numpy(b_filtered).unsqueeze(0).unsqueeze(0)
            ab_up = torch.cat([a_filt_t, b_filt_t], dim=1).to(ab_up.device).unsqueeze(0)

        # Convert final to RGB
        out_rgb = batch_lab2rgb_transpose_mc(input_lab[:,0:1,:,:], ab_up[:1])
        np_rgb  = out_rgb[0].astype(np.uint8)
        color_pil = Image.fromarray(np_rgb, mode="RGB")

        # Convert back to Torch
        output_tensor = pil_to_tensor(color_pil)  # shape [C,H,W], float 0..1
        return (output_tensor,)


NODE_CLASS_MAPPINGS["DeepExemplarImageColorNode"] = DeepExemplarImageColorNode
