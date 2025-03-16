import os
import sys
import torch
import types

###############################################################################
# 1) Patch torch.load for data/<filename>.pth so base code's relative paths 
#    are mapped to absolute paths in your local "data" folder.
###############################################################################
orig_torch_load = torch.load
REMAP_PATHS = {
    "data/vgg19_gray.pth": os.path.join(os.path.dirname(__file__), "data", "vgg19_gray.pth"),
}
def custom_torch_load(path, *args, **kwargs):
    if path in REMAP_PATHS:
        path = REMAP_PATHS[path]
    return orig_torch_load(path, *args, **kwargs)
torch.load = custom_torch_load

###############################################################################
# 2) Insert this custom node root directory into sys.path if not already present
###############################################################################
custom_node_root = os.path.dirname(os.path.abspath(__file__))
if custom_node_root not in sys.path:
    sys.path.insert(0, custom_node_root)

###############################################################################
# 3) Remove any pre-existing "utils" or "models" entries from sys.modules 
###############################################################################
for modname in ["utils", "models"]:
    if modname in sys.modules:
        del sys.modules[modname]

###############################################################################
# 4) Dynamically create "utils" and "models" packages, pointing them to our local directories.
###############################################################################
utils_dir = os.path.join(custom_node_root, "utils")
models_dir = os.path.join(custom_node_root, "models")
if os.path.isdir(utils_dir):
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = [utils_dir]
    sys.modules["utils"] = utils_module
if os.path.isdir(models_dir):
    models_module = types.ModuleType("models")
    models_module.__path__ = [models_dir]
    sys.modules["models"] = models_module

###############################################################################
# 5) Import and register node mappings.
###############################################################################
from .DeepExemplarColorizationNodes import NODE_CLASS_MAPPINGS
