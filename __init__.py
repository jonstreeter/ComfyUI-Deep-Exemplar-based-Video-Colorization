import os
import sys

# Add the parent directory of 'models' and 'utils' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .DeepExemplarImageColorNode import NODE_CLASS_MAPPINGS as IMG_MAP
from .DeepExemplarVideoColorNode import NODE_CLASS_MAPPINGS as VID_MAP

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(IMG_MAP)
NODE_CLASS_MAPPINGS.update(VID_MAP)