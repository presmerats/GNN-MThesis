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
from TFM_Dataset_part_VIII_Symbols_Dataset_labelling_config import *


# read the folders and cp the files to another place but renaming them accordingly
# function_renaming/tmp/to_merge
# function_renaming/tmp/symbols_dataset_2


def print_function_names(dataset):
    """
        This func is used to read fuctiona names
        and manually gather some keywors

    """



    progress = 0
    try:
        progress = pickle.load(open('labelling_progress.pickle','rb'))
    except:
        pass 

    for i in range(progress,len(dataset)):
        f = dataset[i].filename
        j = len('tmp/symbols_dataset_part1/raw/graphs01/unknown/')
        f = f[j:]
        k = f.find('/')
        l = f.find('_edges.txt')
        f2 = f[(k+1):l]
        f3 = f[:k]
        # remove _edges.txt 
        print(f3,f2)
        progress += 1
        pickle.dump(progress,open('labelling_progress.pickle','wb'))
        if i % 10 == 0 :
            pressed_key = input("press any key..")
            if pressed_key.lower()=="x":
                break



    

def find_max_match(match_dict):
    maxcount = -1
    maxcount2 = -1
    maxcount_key = ''
    for k,v in match_dict.items():
        if v > maxcount:
            maxcount_key=k
            maxcount = v 
        elif v > maxcount2:
            maxcount2_key = k
            maxcount2 = v
            
    if maxcount2>1 and maxcount - maxcount2 < 2:
        return maxcount_key, maxcount2_key
    else:
        return maxcount_key, ''

def find_label(file_name,topics_dict,top2=True):
    """
        sets label depending on the executable filename
    """
    # using f2
    label = ''

    matches = {}
    for k,v in topics_dict.items():
        keywords1 = v
        matches[k]=0
        # check topic is present
        for kw in keywords1:
            if file_name.find(kw)>-1:
                matches[k]+=1                


    label,label2 = find_max_match(matches)

    if top2:     
        return label+"_"+label2 
    else:
        return label


def find_specific_label_v1(function_name):
    # using f2
    label = ''

    for comb in combinations:
        

        topic_name = comb[0]
        

        # topic keywords
        keywords1 = topics[topic_name]

        # check topic is present
        topic_found = False
        for kw in keywords1:
            if function_name.find(kw)>-1:
                topic_found = True
                break 

        if not topic_found:
            continue 


        # task keywords (if any)
        keywords2 = []
        if len(comb)==1:
            label = topic_name
            break
        elif len(comb)==2:
            task_name = comb[1]
            keywords2 = tasks[task_name]
            task_found = False
            for kw in keywords2:
                if function_name.find(kw)>-1:
                    task_found = True
                    label = topic_name+'_'+task_name
                    break 

        if topic_found and task_found:
            break

    return label 




def set_labels(dataset):

    """

    ok- prepare:
        - topics (high-level)
        - tasks

    ok- read filenames and add synonyms manually

    ok- reduce tasks to the predefined combinations topic-tasks

    - run:
      
        - classes
            - get keywords of each of the subclass tuple
            - if keywords in both -> class assigned
            - assign many classes ? or just first one?

            
        ok- inspect matches of folder and filename
        ok- add each class of each match 

    - count matches of each topic and each task.
        add the topic-task with most matches

    """

    labels = []
    for i in range(len(dataset)): #range(100): #
        
        f = dataset[i].filename
        
        j = len('tmp/symbols_dataset_part1/raw/graphs01/unknown/')
        f = f[j:]
        k = f.find('/')
        l = f.find('_edges.txt')
        f2 = f[(k+1):l].lower()
        f3 = f[:k].lower()
        
        
        #label = find_label(f2)
        label1 = find_label(f2,topics)
        label2 = find_label(f2,tasks,top2=False)
        label  = label1+"__"+label2


        if label=='':
            label = find_label(f3,filename_topics)


        labels.append((i,f3,f2,label))

        #print(i,f3,f2, label)

        if i % 1000 == 0 :
            print(i,f3, f2, label)

    with open('labels.txt','w') as f:
        for l in labels:

            f.write(str(l[0])+" "+\
                    str(l[1])+" "+\
                    str(l[2])+" "+\
                    str(l[3])+" "+\
                    "\n")
        





if __name__ == '__main__':


    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)

    #print_function_names(dataset)
    set_labels(dataset)