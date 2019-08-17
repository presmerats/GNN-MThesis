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





def modify_data_y_and_label(idx,y,thelabel):
    dataobj = dataset[idx]
    dataobj.__setattr__('label',thelabel)
    dataobj.y = y
    dataset.save_changes( idx, dataobj )

def read_labels_from_csv(csv_file, version=1):
    """
        returns a dict of function_names and label in string
        version=1
            return onlythe first topic

        version=2
            returns the topic and the task 
            (more classes, more granularity)
    """

    labels = {}
    with open(csv_file,'r') as f:
        for line in f.readlines():
            elems = line.split(',')

            if version==1:
                elems[3] = elems[3].split('_')[0]
            else:

                try:
                    topic = elems[3].split('_')[0]
                    elems[3] = elems[3].replace('___','__')
                    task = elems[3].split('__')[1]
                    elems[3] = topic +'_'+task

                except:
                    elems[3] = elems[3].replace('___','_')
                    elems[3] = elems[3].replace('__','_')

            if elems[1] not in labels.keys():
                labels[elems[1]]={}
            labels[elems[1]][elems[2]]= elems[3].replace('\n','')

    return labels 


def get_file_func_names(filename):
    f = filename 
    j = len('tmp/symbols_dataset_part1/raw/graphs01/unknown/')
    f = f[j:]
    k = f.find('/')
    l = f.find('_edges.txt')
    f2 = f[(k+1):l].lower()
    f3 = f[:k].lower()
    return f3,f2

def set_labels_on_dataset(dataset, labels):

    unique_labels = list(set([ v2 for k,v in labels.items() for k2,v2 in v.items()]))
    #print("how many unique labels: ",len(unique_labels))

    int_id = 0
    labels_to_int = { }
    for l in unique_labels:
        labels_to_int[l]=int_id
        int_id+=1
    #print("Last id of unique labels: ",int_id)

    json.dump(labels_to_int,open('labels_id_v1','w'))

    errors = 0
    for i in range(len(dataset)):
        filename = dataset[i].filename
        # get filename
        # get function name
        filename,funcname = get_file_func_names(filename)

        try:
            label = labels[filename][funcname]
            #print(filename, funcname, label, labels_to_int[label])
        except:
            errors+=1
            #print(filename,funcname,'')
            # label as unknown?
            label = ''

        #modify_data_label(i,label)            
        modify_data_y_and_label(i,labels_to_int[label],label)
        

    print("how many unique labels: ",len(unique_labels))
    print("Last id of unique labels: ",int_id)
    print("num errors ",errors)

if __name__ == '__main__':

    # v1 labels--dataset----------------------------
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    #dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)

    #print_function_names(dataset)
    csv_file ='./labels-mod3-verified-10000.csv'
    labels = read_labels_from_csv( csv_file, version=1)
    #labels = read_labels_from_csv( csv_file, version=2)
    #pprint(labels)

    set_labels_on_dataset(dataset, labels)



    # v2 labels--dataset----------------------------
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)

    #print_function_names(dataset)
    csv_file ='./labels-mod3-verified-10000.csv'
    labels = read_labels_from_csv( csv_file, version=2)
    #pprint(labels)


    set_labels_on_dataset(dataset, labels)

    for  i in range(len(dataset)):
        data = dataset[i]
        print(data.y, data.label)


    # verify 
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(dataset[0].label,dataset[0].y)

    dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    print(dataset[0].label,dataset[0].y)