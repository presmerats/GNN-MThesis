import importlib
import time
import pickle
import traceback
import random
import os
import re
import sys
import json
from pprint import pprint
import numpy as np
from numpy.random import choice
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset


# separate the dataset into directories 


dataset = FunctionsDataset(root='./tmp/symbols_dataset_part1')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part2')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part3')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part4')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part5')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part6')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part7')
dataset = FunctionsDataset(root='./tmp/symbols_dataset_part8')