import numpy as np
import pandas as pd
# pd.options.plotting.backend = "plotly"
import random

import os
from tqdm import tqdm
tqdm.pandas()
import time
import copy
# import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
# from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision.transforms.functional as TF

#torchio
import torchio as tio


# class indices

LARGE_BOWEL = 0
SMALL_BOWEL = 1
STOMACH = 2
MASK_INDICES = {'large_bowel': LARGE_BOWEL, 'small_bowel':SMALL_BOWEL, 'stomach':STOMACH}
DATA_SIZE = 274