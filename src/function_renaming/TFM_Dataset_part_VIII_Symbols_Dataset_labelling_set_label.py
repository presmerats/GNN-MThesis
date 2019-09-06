"""
This script reads a csv with function_names and classses, 
then traverses an imported dataset of functions and modifies the class according to it.



"""

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





def modify_data_y_and_label(idx,y,thelabel, dataset):
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


def read_labels_from_csv_v2(csv_file, version=1):
    """
        returns a dict of function_names and label in string
        version=1
            return onlythe first topic

        version=2
            returns the topic and the task 
            (more classes, more granularity)

        In this version2, the separator is not important,
        could be '___', '__' or '_'
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
                    elems[3] = elems[3].replace('___','_')
                    elems[3] = elems[3].replace('__','_')
                    task = elems[3].split('_')[1]
                    elems[3] = topic +'_'+task

                except:
                    elems[3] = elems[3].replace('___','_')
                    elems[3] = elems[3].replace('__','_')

            if elems[1] not in labels.keys():
                labels[elems[1]]={}
            labels[elems[1]][elems[2]]= elems[3].replace('\n','')

    return labels 



def label_tuple_to_string(label_tuple):
    k = label_tuple
    if isinstance(k,tuple) and len(k) == 2:        
        return k[0]+'_'+k[1]

    elif isinstance(k,tuple) and len(k)==1:
        return k[0]

    else:
        #print(type(k),k,k[0],k[1])
        return k
        

def find_tuple_as_key(t, label_dict):
    """
        find if the tuple t 
        is one of the key in the label_dict,
        and return it,

        return None otherwise
    """
    #print("find_tuple_as_key",t)
    for k in label_dict.keys():
        #print("     ->:",k)
        # if len(k) == len(t):
        #     print(k,t)
        if len(k) == 2 \
            and len(k) == len(t) \
            and k[0].lower() == t[0].lower() \
            and k[1].lower() == t[1].lower():
            
            return label_tuple_to_string(k)
        elif len(k)==1 \
            and len(k) == len(t) \
            and k[0].lower() == t[0].lower():
                
                return label_tuple_to_string(k)

    return None

def find_tuple_as_synonym_match(t, label_dict):
    """
        find if the tuple t 
        is one of the synonyms of the keys in the label_dict,
        and return the key,

        return None otherwise
    """
    #print("find_tuple_as_synonym_match")
    for tuple_key,synonyms in label_dict.items():
        for k in synonyms:
            if len(k) == 2 \
                and len(k) == len(t) \
                and k[0].lower() == t[0].lower() \
                and k[1].lower() == t[1].lower():
                
                return label_tuple_to_string(tuple_key)

            elif len(k)==1 \
                and len(k) == len(t) \
                and k[0].lower() == t[0].lower():
                    
                    return label_tuple_to_string(tuple_key)

    return None



def iterative_read_labels_from_csv(csv_file, version=2):
    """
        Recreate read_labels_from_csv
        but:
            - replace('__','_')
            - replace('___','_')
            - apply the synonyms dict to reduce the number of labels
            - count how many labels don't have a synonym match
            - show that list
            - return labels dict
    """

    # parse labels
    labels_dict = read_labels_from_csv_v2(csv_file, version=version)

    # unmatched list
    unmatched=[]

    # synonym match
    for modulekey,v in labels_dict.items():
        for filekey,v2 in v.items():
            # transform into tuple of (topic,task)
            ts = v2.split('_')
            topic = ts[0]
            topic_task_tuple = (topic,)
            if len(ts)>1:
                task = ts[1]
                topic_task_tuple = (topic,task)

            # find it as key in synonyms dict
            label_tuple = find_tuple_as_key(topic_task_tuple, synonyms)


            # or match if as synonym
            if label_tuple is None:
                label_tuple = find_tuple_as_synonym_match(topic_task_tuple, synonyms)



            if label_tuple is None:
                if topic_task_tuple[0] != '':
                    unmatched.append(topic_task_tuple)
                labels_dict[modulekey][filekey] = ''
            else:
                # save label_tuple into the labels dict
                labels_dict[modulekey][filekey] = label_tuple_to_string(label_tuple)


            # if len(unmatched)>10:
            #     break

    pprint(unmatched)
    print("total unmatched ", len(unmatched))
    return labels_dict



def get_file_func_names(filename):
    f = filename 
    j = len('tmp/symbols_dataset_part1/raw/graphs01/unknown/')
    f = f[j:]
    k = f.find('/')
    l = f.find('_edges.txt')
    f2 = f[(k+1):l].lower()
    f3 = f[:k].lower()
    return f3,f2



def set_labels_on_dataset(dataset, labels, file_on_disk='labels_id_v3'):
    """
    given a dict of func-name: label

    translatees labels to integer 
        file: labels_id_v1

    for each graph in the dataset
        get_file_func_names()
        label = labels[filename][funcname]
        modify_data_y_and_label(i, labels_to_int[label], label, dataset)
    """

    print("\nverifying labels dict:")
    ll = 0
    for k,v in labels.items():
        
        for k2,v2 in v.items():
            print(k,k2,v2)
            ll+=1
            if ll>10:
                break

    print()

    unique_labels = list(set([ v2 for k,v in labels.items() for k2,v2 in v.items()]))
    print("how many unique labels: ",len(unique_labels))

    int_id = 0
    labels_to_int = { }
    for l in unique_labels:
        labels_to_int[l]=int_id
        int_id+=1
    #print("Last id of unique labels: ",int_id)

    json.dump(labels_to_int,open(file_on_disk,'w'))

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
        modify_data_y_and_label(i,labels_to_int[label],label, dataset)
        

    print("how many unique labels: ",len(unique_labels))
    print("Last id of unique labels: ",int_id)
    print("num errors ",errors)

if __name__ == '__main__':


    # v3 labels--dataset----------------------------
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_3')
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)
    csv_file ='./labels-mod3-verified-20000-recovered2.csv'
    labels = iterative_read_labels_from_csv(csv_file, version=2)

    # print("\n computed labels:")
    # pprint(labels)
    # print()

    set_labels_on_dataset(dataset, labels, file_on_disk='labels_id_v3')

    # print("result of labeling the dataset")
    # for  i in range(len(dataset)):
    #     data = dataset[i]
    #     print(data.y, data.label)



    exit()

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

    set_labels_on_dataset(dataset, labels,file_on_disk='labels_id_v1')



    # v2 labels--dataset----------------------------
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)

    #print_function_names(dataset)
    csv_file ='./labels-mod3-verified-10000.csv'
    labels = read_labels_from_csv( csv_file, version=2)
    #pprint(labels)


    set_labels_on_dataset(dataset, labels,file_on_disk='labels_id_v2')

    for  i in range(len(dataset)):
        data = dataset[i]
        print(data.y, data.label)


    # verify 
    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    print(dataset[0].label,dataset[0].y)

    dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    print(dataset[0].label,dataset[0].y)