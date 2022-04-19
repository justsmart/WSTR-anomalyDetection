import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d, resnet26, resnet50
from timm.models.registry import register_model
from torchvision.ops import roi_align
import math
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


