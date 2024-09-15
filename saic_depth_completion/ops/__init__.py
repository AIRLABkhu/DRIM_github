from functools import partial

import torch

from .batch_norm import FrozenBatchNorm2d
from .spade import SPADE, SelfSPADE
from .sean import SEAN