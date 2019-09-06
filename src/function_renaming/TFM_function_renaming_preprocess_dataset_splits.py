import pandas as pd 
import numpy as np 
import os 
from os import environ, path 
from pprint import pprint 
import copy
import scipy
from datetime import datetime
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/graph_classification') 

from TFM_graph_classification import *
from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset
from TFM_function_renaming_baseline_models import *






def precompute_dataset_split(dataset_folder,
                             dataset_version, 
                             min_count,
                             features,
                             split_type_name,
                             destination_folder,
                             ):
    """
        precompute datset splits and savign them to a folder.
        Actual work, all parametrized.
    """
    dataset = FunctionsDataset(root=dataset_folder)
    dataset_version=dataset_version
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_features)

    if split_type_name == 'unchanged' or split_type_name == 'remove_min':
        X_train, X_test, y_train, y_test, nclasses, train_dataset, test_dataset = dataset_split_shared_splits_graph_version(dataset, features=features, min_count=min_count)
    elif split_type_name == 'undersample_max':
        X_train, X_test, y_train, y_test, nclasses, train_dataset, test_dataset = dataset_split_balanced_major_classes_graph_version(dataset, features=features, max_count=min_count)

    data_split_type=split_type_name
    suffix = dataset_version + '-' + data_split_type
    folder = destination_folder

    pickle.dump(X_train,open(
        os.path.join(folder,'X_train.pickle'),'wb+'))
    pickle.dump(X_test,open(
        os.path.join(folder,'X_test.pickle'),'wb+'))
    pickle.dump(y_train,open(
        os.path.join(folder,'y_train.pickle'),'wb+'))
    pickle.dump(y_test,open(
        os.path.join(folder,'y_test.pickle'),'wb+'))
    pickle.dump(nclasses,open(
        os.path.join(folder,'nclasses.pickle'),'wb+'))

    # save dataset graph symlinks for train and test

    # verify dataset contains root folder
    print("Root folder verification: ", train_dataset.root)
    print("Destination: ",os.path.join(destination_folder[4:],'training_set'))

    # symlinks copies of the processed files
    save_partial_dataset_symlinks(
        train_dataset, 
        new_name=os.path.join(destination_folder[4:],'training_set'))
    save_partial_dataset_symlinks(
        test_dataset, 
        new_name=os.path.join(destination_folder[4:],'test_set'))







def precompute_dataset_splits_unchanged_classes():
    """
        loads a dataset, performs the train/test split with  shared splits,
        writes back to pickle files

        dataset: v1,v2,v3
        split types:
            - preserve number of clases: not purging min classes (mincount=0)
            - removing very minimal classes(mincount>0)
            - soft balance: weight minor- balance majorclasses(maxcount>0)

    """
    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_1',
                             dataset_version='v1', 
                             min_count=0,
                             features='all',
                             split_type_name='unchanged',
                             destination_folder= 'tmp/symbols_dataset_1_precomp_split_unchanged',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_2',
                             dataset_version='v2', 
                             min_count=0,
                             features='all',
                             split_type_name='unchanged',
                             destination_folder= 'tmp/symbols_dataset_2_precomp_split_unchanged',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_3',
                             dataset_version='v3', 
                             min_count=0,
                             features='all',
                             split_type_name='unchanged',
                             destination_folder= 'tmp/symbols_dataset_3_precomp_split_unchanged',
                             )


def precompute_dataset_splits_remove_min_classes():
    """
        loads a dataset, performs the train/test split with  shared splits,
        writes back to pickle files

        dataset: v1,v2,v3
        split types:
            - preserve number of clases: not purging min classes (mincount=0)
            - removing very minimal classes(mincount>0)
            - soft balance: weight minor- balance majorclasses(maxcount>0)

    """
    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_1',
                             dataset_version='v1', 
                             min_count=0,
                             features='all',
                             split_type_name='remove_min',
                             destination_folder= 'tmp/symbols_dataset_1_precomp_split_remove_min',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_2',
                             dataset_version='v2', 
                             min_count=0,
                             features='all',
                             split_type_name='remove_min',
                             destination_folder= 'tmp/symbols_dataset_2_precomp_split_remove_min',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_3',
                             dataset_version='v3', 
                             min_count=0,
                             features='all',
                             split_type_name='remove_min',
                             destination_folder= 'tmp/symbols_dataset_3_precomp_split_remove_min',
                             )


def precompute_dataset_splits_undersample_max_classes():
    """
        loads a dataset, performs the train/test split with  shared splits,
        writes back to pickle files

        dataset: v1,v2,v3
        split types:
            - preserve number of clases: not purging min classes (mincount=0)
            - removing very minimal classes(mincount>0)
            - soft balance: weight minor- balance majorclasses(maxcount>0)

    """
    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_1',
                             dataset_version='v1', 
                             min_count=4580,
                             features='all',
                             split_type_name='undersample_max',
                             destination_folder= 'tmp/symbols_dataset_1_precomp_split_undersample_max',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_2',
                             dataset_version='v2', 
                             min_count=2484,
                             features='all',
                             split_type_name='undersample_max',
                             destination_folder= 'tmp/symbols_dataset_2_precomp_split_undersample_max',
                             )


    precompute_dataset_split(dataset_folder='./tmp/symbols_dataset_3',
                             dataset_version='v3', 
                             min_count=2009,
                             features='all',
                             split_type_name='undersample_max',
                             destination_folder= 'tmp/symbols_dataset_3_precomp_split_undersample_max',
                             )

def count_samples_per_class(dataset, class_ids):

    class_counts = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
        if cl not in class_counts.keys():
            class_counts[cl]=0
        class_counts[cl]+=1

    class_counts2 = {}
    for k,v in class_counts.items():
        class_counts2[str(k)+'_'+class_ids[k]]=v

    return class_counts2



    

def get_class_ids(dataset):
    
    class_ids = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
        if cl not in class_ids.keys():
            class_ids[cl]=dataset[i].label
        else:
            if class_ids[cl] != dataset[i].label:
                print("class ",cl," label ",class_ids[cl]," other label ",dataset[i].label)
        #class_counts[cl]+=1

    return class_ids


def inspect_dataset(dataset_folder):
    print("Inspecting", dataset_folder)
    dataset = FunctionsDataset(root=dataset_folder)
    print("num samples:",len(dataset))
    print("num classes:",dataset.num_classes)
    print("num features:",dataset.num_features)

    class_ids = get_class_ids(dataset)
    # print(" Class id - name:")
    # pprint(class_ids)

    samples_per_class = count_samples_per_class(dataset, class_ids)
    print(" Samples per class:")
    pprint(samples_per_class)




if __name__ == '__main__':

    # inspect_dataset('./tmp/symbols_dataset_1')
    # inspect_dataset('./tmp/symbols_dataset_2')
    #inspect_dataset('./tmp/symbols_dataset_3')
    
    # selected max counts for each dataset version
    # are written(hardcoded) inside each corresponding function

    precompute_dataset_splits_unchanged_classes()
    precompute_dataset_splits_remove_min_classes()
    precompute_dataset_splits_undersample_max_classes()