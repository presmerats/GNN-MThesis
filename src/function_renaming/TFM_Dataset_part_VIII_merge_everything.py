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
from shutil import copyfile

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset


# read the folders and cp the files to another place but renaming them accordingly
# function_renaming/tmp/to_merge
# function_renaming/tmp/symbols_dataset_2


last_file = 0

folder = '/home/pau/Projectes/GNN-MThesis/src/function_renaming/tmp/to_merge'

destination_folder = '/home/pau/Projectes/GNN-MThesis/src/function_renaming/tmp/symbols_dataset_2/processed/'

last_root_folder = ''

# walk folders
for root, dirs, files in os.walk(folder, topdown=False):
    #print(root, dirs, files)
    for name in files:
        if root.find('processed')<0:
            #print("break loop on ",root)
            break

        if last_root_folder != root:
            print(root)
            last_root_folder = root 


        filename = os.path.join(root, name)
        

        # get last index of copied file 
        # generate new filename
        dst = os.path.join(destination_folder,"data_"+str(last_file+1)+".pt")
        
        # on each folder move file to localtion, changing index value

        copyfile(filename,dst)

        last_file+=1

