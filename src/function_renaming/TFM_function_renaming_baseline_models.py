import pandas as pd 
import numpy as np 
import os 
from os import environ, path 
from datetime import datetime

from pprint import pprint 

import pickle


import itertools 

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid


import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F

import torch
import torch.utils.data 
from torch.utils.data import Dataset, TensorDataset

from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset
#from .graph_classification.TFM_graph_classification_baseline_models import *



import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/graph_classification') 

from TFM_graph_classification import *
#from TFM_graph_classification_models import *



def save_results(results_dict, features, dataset_version, results_folder='baseline', models_folder='models/'):
    """
    # append results to results_dict on disk
    append results to resutls_dict on disk(json format)
                 serializes model object as a pickle file
                 writes the filen path to 'model_instance' key

    it assumes it is executed from the root folder src/function_renaming
    and so will save results json files to results/name.json
    """

    if dataset_version=='v1' and results_folder=='baseline':
        filepath = 'results/training_results.json'
    elif dataset_version=='v2' and results_folder=='baseline':
        filepath = 'results/training_results_v2.json'
    elif dataset_version=='v1' and results_folder=='nlp':
        filepath = 'results/nlp_training_results.json'
    elif dataset_version=='v2' and results_folder=='nlp':
        filepath = 'results/nlp_training_results_v2.json'
    else:
        filepath = results_folder
        if filepath.find('results/')!= 0:
            filepath = 'results/'+filepath
        
    try:
        if not os.path.exists('./results'):
            os.mkdir('./results')

        if not os.path.exists(filepath):
            f = open(filepath,'w+')
            f.write('{}')
            f.close()
        else:
            f = open(filepath,'r')
            content = f.read()
            f.close()
            if content[0]!='{' or content[-1]!='}':
                f = open(filepath,'w+')
                f.write('{}')
                f.close()


        r = json.load(open(filepath,'r'))

        # how to merge? , just append? or merge for each model?
        for model_name in results_dict.keys():
            if model_name not in r.keys():
                r[model_name]={}

            for score,score_val in results_dict[model_name].items():
                #print(score)
                score_val['features'] = features

                datetime_str=datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                model_filename = os.path.join(models_folder,model_name+'_'+datetime_str)
                    
                # remove model_instance and pickle it to disk
                try:
                    #print(score_val.keys())
                    model_instance = score_val.pop('model_instance')
                    pickle.dump(model_instance,open(model_filename,'wb'))

                except Exception as err:
                    print("need to implement model_instance saving to disk")
                    traceback.print_exc()
                    
                score_val['model_instance'] = model_filename

                r[model_name][score + datetime_str] = score_val 
        
        json.dump(r, open(filepath,'w'))
        #pprint(results_dict)
        #return results_dict
    except Exception as err:
        print("Error with "+model_name+" and "+score)
        traceback.print_exc()


def param_to_string(params_dict):
    """
        Model params dictionary to a string
    """
    result = ''
    if params_dict is not None:
        for k,v in params_dict.items():
            result += '__'+str(k)+':'+str(v)

    if result.find('__')==0:
        result=result[2:]
    return result


def params_to_string(params_list):
    result =[]

    for params in params_list:
        result.append(param_to_string(params))
    return result


def get_params(param, list_dicts):
    result = []

    #print("get_params")
    for d in list_dicts:
        #pprint(d)
        if isinstance(d, dict):
            result.append(d[param])
        else:
            if len(d)>0 and isinstance(d[0],dict):
                result.append(d[0][param])
            else:
                result.append(d)

    return result



def get_filepath(dataset_version, results_folder, results_file=None):
    if results_file is not None:
        filepath = results_file
    elif dataset_version=='v1' and results_folder=='baseline':
        filepath = 'results/training_results.json'
    elif dataset_version=='v2' and results_folder=='baseline':
        filepath = 'results/training_results_v2.json'
    elif dataset_version=='v1' and results_folder=='nlp':
        filepath = 'results/nlp_training_results.json'
    elif dataset_version=='v2' and results_folder=='nlp':
        filepath = 'results/nlp_training_results_v2.json'
    else:
        filepath=results_folder

    return filepath


def print_training_stats(dataset_version='v1', results_folder='baseline', results_file=None):
    """
    read json from disk and print the best model of each model type.
    print also it's scores obviously
    """

    filepath = get_filepath(dataset_version, results_folder, results_file)
    
    r = json.load(open(filepath,'r'))
    
    info_res = [ 
        (model,score,score_res['cv_score'], 
         score_res['params'], 
         score_res['features'], 
         score_res['weighted avg']) 
        if 'weighted avg' in score_res.keys() 
        else  
        (model,score,score_res['cv_score'], 
         score_res['params'], 
         score_res['features'],
         '')
        for model,model_res in r.items() 
            for score,score_res in model_res.items() 
             ]

    # for each modell print the best result
    models = list(set([ a[0] for a in info_res]))

    best_models_list = []    
    for model in models:

        scores = [ a[2] for a in info_res if a[0]==model]
        params = [ a[3] for a in info_res if a[0]==model]
        score_names = [ a[1] for a in info_res if a[0]==model]
        features = [ a[4] for a in info_res if a[0]==model]
        weighted_scores = [ a[5] for a in info_res if a[0]==model]
        max_score = max(scores)
        max_model = scores.index(max_score)
        max_model_params = params[max_model] # later this will be params
        score_name = score_names[max_model]
        best_models_list.append((model, score_name, max_score, max_model_params,features[max_model],weighted_scores))


    # print all results for all models in a flat table
    best_models = pd.DataFrame(data={
        'model': [a[0] for a in best_models_list],
        'parameters': params_to_string([a[3] for a in best_models_list]),
        'data features': [a[4] for a in best_models_list],
        'optimized score': [a[1] for a in best_models_list],
        'avg score in cv': [a[2] for a in best_models_list],
        'micro-precision': get_params('precision',[a[5] for a in best_models_list]),
        'micro-recall': get_params('recall',[a[5] for a in best_models_list]),
        'micro-f1': get_params('f1-score',[a[5] for a in best_models_list]),
        'support': get_params('support',[a[5] for a in best_models_list]),
        }
        )


    return best_models



def print_all_training_stats(dataset_version='v1', results_folder='baseline', results_file=None):
    """
    read json from disk and print the best model of each model type.
    print also it's scores obviously
    """

    filepath = get_filepath(dataset_version, results_folder, results_file)
    
    
    r = json.load(open(filepath,'r'))
    
    info_res = [ 
        (model,score,score_res['cv_score'], 
         score_res['params'], 
         score_res['features'], 
         score_res['weighted avg']) 
        if 'weighted avg' in score_res.keys() 
        else  
        (model,score,score_res['cv_score'], 
         score_res['params'], 
         score_res['features'],
         '')
        for model,model_res in r.items() 
            for score,score_res in model_res.items() 
             ]

    # print all results for all models in a flat table
    all_models = pd.DataFrame(data={
        'model': [a[0] for a in info_res],
        'parameters': params_to_string([a[3] for a in info_res]),
        'data features': [a[4] for a in info_res],
        'optimized score': [a[1] for a in info_res],
        'avg score in cv': [a[2] for a in info_res],
        'micro-precision': get_params('precision',[a[5] for a in info_res]),
        'micro-recall': get_params('recall',[a[5] for a in info_res]),
        'micro-f1': get_params('f1-score',[a[5] for a in info_res]),
        'support': get_params('support',[a[5] for a in info_res]),
        }
        )

   
    return all_models
    




def balance_maximum_classes(dataset, max_count=-1,min_remove=-1, remove_classes=[]):
    """
        shuffle and then count items by class,
            use a threshold to remove classes with more items than the threshold

    """

    dataset.shuffle()

    class_counts = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
        if cl not in class_counts.keys():
            class_counts[cl]=0
        class_counts[cl]+=1

    pprint(class_counts)

    undersample_classes = []
    remove_mins= remove_classes
    for k,v in class_counts.items():
        if max_count>-1 and v>max_count:
            undersample_classes.append(k)
            print(" classes to undersample ",k)
        elif min_remove > -1 and v<min_remove:
            remove_mins.append(k)        


    #max_class_count = min(class_counts.values())
    # deactivated feature by using now the max
    max_class_count = max_count if max_count > -1 else max(class_counts.values())
    print("max class count set to ", max_class_count)


    purged_list = []
    final_class_counts = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
    
        if cl not in final_class_counts.keys():
            final_class_counts[cl]=0
        
        if final_class_counts[cl]>max_class_count:
            continue
        
        if cl in remove_mins:
            continue

        final_class_counts[cl]+=1
        purged_list.append(i)

    print("final class counts after trimming max and min classes")
    pprint(final_class_counts)

    #return purged_list 
    return purged_list , len(list(final_class_counts.keys())) 


def purge_minimum_classes(dataset, min_count=40):
    """
    shuffle and then count items by class,
                use a threshold to remove classes with less items than the threshold
                -> could be useful , but it's a bit misleading
                -> it is better to do this approach on the labeling phase(like remove labels from minority classes
                    and so they are no longer present in the dataset)

                20190905 currently used with a threshold of 0
    """

    dataset.shuffle()

    class_counts = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
        if cl not in class_counts.keys():
            class_counts[cl]=0
        class_counts[cl]+=1

    pprint(class_counts)

    remove_classes = []
    for k,v in class_counts.items():
        if v<min_count:
            remove_classes.append(k)
            print(" to remove class ",k)
            
    for k in remove_classes:
        class_counts.pop(k)

    #max_class_count = min(class_counts.values())
    # deactivated feature by using now the max
    max_class_count = max(class_counts.values())
    print("max class count set to ", max_class_count)

    purged_list = []
    final_class_counts = {}
    for i in range(len(dataset)):
        cl = dataset[i].y 
        if cl not in remove_classes:
            

            if cl not in final_class_counts.keys():
                final_class_counts[cl]=0
            
            if final_class_counts[cl]>max_class_count:
                continue

            final_class_counts[cl]+=1
            purged_list.append(i)

    pprint(final_class_counts)

    #return purged_list 
    return purged_list , len(list(final_class_counts.keys())) 


def pack_features(dataset, samples_list):
    """

    for before dataset split, gets all features from topo, code and document
    and puts them into a np.ndarray()

    get all columns of topological features matrix, code features matrix , document(code bag of words)
                 and return a numpy array/matrix

    # IN:purged_list, features, dataset, 
    # OUT: X

    """
    X = []
    for j in samples_list:
        d = dataset[j].__getattribute__('code_feats')
        topos = dataset[j].__getattribute__('x_topo_feats')
            

        # if features == 'x_topo_feats':
        #     cfeats.extend(topos)
            
        # elif features =='code_feats' or features=='code feats':
        #     cfeats =[ d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
        # # document, topological and code features
        # elif features =='topo and code feats' or features=='feats_topo_code':
        #     cfeats =[d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
        #     cfeats.extend(topos)

        # elif features == 'document':
        #     cfeats = d['document']

        # elif features == 'document_simplified':
        #     cfeats = d['document_simplified']

        # elif features =='document and list funcs':
        #     cfeats =[d['document'], d['list_funcs']]

        # elif features =='document_simplified and list funcs':
        #     cfeats =[d['document_simplified'], d['list_funcs']]

        # # document and topological features
        # elif features =='document and topo feats':
        #     cfeats =[d['document'], d['list_funcs']]
        #     cfeats.extend(topos)

        # elif features =='document_simplified and topo feats':
        #     cfeats =[d['document_simplified'], ]
        #     cfeats.extend(topos)

        # elif features =='document_simplified and list_funcs and topo feats':
        #     cfeats =[d['document_simplified'], d['list_funcs']]
        #     cfeats.extend(topos)

        # # document and code features
        # #code_feats = {
        # #         'nregs': num_regs,
        # #         'num_distinct_regs': num_distinct_regs,
        # #         'ninstrs': num_instrs,
        # #         'ndispls': num_displs,
        # #         'nimms': num_imms,
        # #         'nmaddrs': num_memaddrs,
        # #         'num_funcs': num_funcs,
        # #         'document': doc,              -> removed
        # #         'document_simplified': doc2,  -> removed
        # #         'list_regs': list_regs,       -> removed
        # #         'list_funcs': list_funcs      -> removed
        # #     }
        # elif features =='document and code feats':
        #     cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]

        # elif features =='document_simplified and code feats':
        #     cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
        # elif features =='document_simplified and list_funcs and code feats':
        #     cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
        # # document, topological and code features
        # elif features =='document and topo and code feats':
        #     cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
        #     cfeats.extend(topos)

        # elif features =='document_simplified and topo and code feats':
        #     cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        #     cfeats.extend(topos)
            
        # elif features =='document_simplified and list_funcs and topo and code feats':
        #     cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        #     cfeats.extend(topos)

        # elif features =='all':
        #     cfeats =[d['document'],d['document_simplified'],d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        #     cfeats.extend(topos)


        cfeats =[d['document'],d['document_simplified'],d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        cfeats.extend(topos)

        X.append(cfeats)
            

    X = np.array(X)

    return X



def pack_features_graph_version(dataset, samples_list):
    """
        This is applied before the dataset split, and usually will contain all the features,
        the feature selection will be performed later when training specific model with specific features

        Remake of filter_features, 
        but this time a column containing the index of the row in the original dataset is added at the end 
        X[:,-1] contains the num of the row it comes from.
        This will be used it further functions to split the dataset
    """

    
    X = []
    for j in samples_list:
        d = dataset[j].__getattribute__('code_feats')
        topos = dataset[j].__getattribute__('x_topo_feats')

        # get all the features
        cfeats =[d['document'],d['document_simplified'],d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        cfeats.extend(topos)

        # add the row number to the last column
        cfeats.append(j)

        X.append(cfeats)
            

    X = np.array(X)

    return X



def filter_features_new(X_train, features):
    """
    # IN:purged_list, features, dataset, 
    # OUT: X, X_doc
        get selected columns of topological features matrix, code features matrix , document(code bag of words)
        and return a numpy array/matrix for numeric features , 
            and another list for documents (only code as one big string possibly concatenated with the list of present func calls)
    """
    X_tr = []
    X_tr_doc = []
    for j in range(len(X_train)):
           
        cfeats_train = []
        doc_feats = []
        
        # when features =='all':
        #     cfeats =[
        #         d['document'],
        #         d['document_simplified'],
        #         d['list_funcs'], 
        #         d['nregs'], 
        #         d['num_distinct_regs'], 
        #         d['ninstrs'], 
        #         d['ndispls'], 
        #         d['nimms'], 
        #         d['nmaddrs'], 
        #         d['num_funcs'],]
        #     cfeats.extend(topos)
        

        l = X_train.shape[1]

        if features == 'x_topo_feats':
            cfeats_train.extend(X_train[j,10:l])
            
            
        elif features =='code_feats' or features=='code feats':
            # cfeats =[ d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
            cfeats_train.extend(X_train[j,3:10])
            

        # document, topological and code features
        elif features =='topo and code feats' or features=='feats_topo_code':
            # cfeats =[d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
            # cfeats.extend(topos)

            list_cols = list(range(3,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features == 'document':
            #cfeats = d['document']
            doc_feats = ''.join(''.join(X_train[j,0]))
            

        elif features == 'document_simplified':
            #cfeats = d['document_simplified']
            doc_feats = ''.join(''.join(X_train[j,1]))
            

        elif features =='document and list funcs':
            #cfeats =[d['document'], d['list_funcs']]
            doc_feats = ''.join(''.join(X_train[j,0] + ' ' + X_train[j,2]))
            

        elif features =='document_simplified and list funcs':
            #cfeats =[d['document_simplified'], d['list_funcs']]
            doc_feats = ''.join(''.join(X_train[j,1] + ' ' + ' '.join(X_train[j,2]) ))
            
            

        # document and topological features
        elif features =='document and topo feats':
            #cfeats =[d['document'], d['list_funcs']]
            #cfeats.extend(topos)

            doc_feats = ''.join(''.join(X_train[j,0]))

            list_cols = list(range(10,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='document_simplified and topo feats':
            # cfeats =[d['document_simplified'], ]
            # cfeats.extend(topos)
            doc_feats = ''.join(''.join(X_train[j,1]))

            list_cols = list(range(10,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='document_simplified and list_funcs and topo feats':
            # cfeats =[d['document_simplified'], d['list_funcs']]
            # cfeats.extend(topos)

            doc_feats = ''.join(''.join(X_train[j,1] + ' ' + ' '.join(X_train[j,2]) ))

            list_cols = list(range(10,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        # document and code features
        #code_feats = {
        #         'nregs': num_regs,
        #         'num_distinct_regs': num_distinct_regs,
        #         'ninstrs': num_instrs,
        #         'ndispls': num_displs,
        #         'nimms': num_imms,
        #         'nmaddrs': num_memaddrs,
        #         'num_funcs': num_funcs,
        #         'document': doc,              -> removed
        #         'document_simplified': doc2,  -> removed
        #         'list_regs': list_regs,       -> removed
        #         'list_funcs': list_funcs      -> removed
        #     }
        elif features =='document and code feats':
            #cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
            
            doc_feats = ''.join(''.join(X_train[j,0]))

            list_cols = list(range(3,10))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='document_simplified and code feats':
            #cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            doc_feats = ''.join(''.join(X_train[j,1]))

            list_cols = list(range(3,10))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='document_simplified and list_funcs and code feats':
            #cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            

            doc_feats = ''.join(''.join(X_train[j,1] + ' ' + ' '.join(X_train[j,2]) ))

            list_cols = list(range(3,10))
            cfeats_train.extend(X_train[j,list_cols])
            

        # document, topological and code features
        elif features =='document and topo and code feats':
            #cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
            #cfeats.extend(topos)

            doc_feats = ''.join(''.join(X_train[j,0]))

            list_cols = list(range(3,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='document_simplified and topo and code feats':
            #cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            #cfeats.extend(topos)
            doc_feats = ''.join(''.join(X_train[j,1]))

            list_cols = list(range(3,l))
            cfeats_train.extend(X_train[j,list_cols])
            
            
        elif features =='document_simplified and list_funcs and topo and code feats':
            # cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            # cfeats.extend(topos)
            doc_feats = ''.join(''.join(X_train[j,1] + ' ' + ' '.join(X_train[j,2]) ))

            list_cols = list(range(3,l))
            cfeats_train.extend(X_train[j,list_cols])
            

        elif features =='all':
            doc_feats = ''.join(''.join(X_train[j,0] + ' ' + X_train[j,2] ))

            list_cols = list(range(3,l))
            cfeats_train.extend(X_train[j,list_cols])
            
        # print("0",X_train[j,0])
        # print("2",X_train[j,2])
        # doc_feats = ''.join(''.join(X_train[j,0] + ' ' + X_train[j,2] ))

        # list_cols = list(range(3,l))
        # cfeats_train.extend(X_train[j,list_cols])
        

        X_tr.append(cfeats_train)
        X_tr_doc.append(doc_feats)
        
            

    X_train_numeric_features  = np.array(X_tr)
    X_train_doc = X_tr_doc 
   
    return  X_train_numeric_features,X_train_doc


def filter_features_new_v2(X_train, features):
    """
    # IN:purged_list, features, dataset, 
    # OUT: X, X_doc
        get selected columns of topological features matrix, code features matrix , document(code bag of words)
        and return a numpy array/matrix for numeric features , 
            and another list for documents (only code as one big string possibly concatenated with the list of present func calls)

        this time a dataframe is returned

        transform the columns of the matrix in a dataframe
    """
    # X_tr = []
    # X_tr_doc = []
    # for j in range(len(X_train)):
           
    #     cfeats_train = []
    #     doc_feats = []
        
    #     # when features =='all':
    #     #     cfeats =[
    #     #         d['document'],
    #     #         d['document_simplified'],
    #     #         d['list_funcs'], 
    #     #         d['nregs'], 
    #     #         d['num_distinct_regs'], 
    #     #         d['ninstrs'], 
    #     #         d['ndispls'], 
    #     #         d['nimms'], 
    #     #         d['nmaddrs'], 
    #     #         d['num_funcs'],]
    #     #     cfeats.extend(topos)
        

    #     l = X_train.shape[1]

    #     if features == 'x_topo_feats':
    #         cfeats_train.extend(X_train[j,10:l])
            
            
    #     elif features =='code_feats' or features=='code feats':
    #         # cfeats =[ d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
    #         cfeats_train.extend(X_train[j,3:10])
            

    #     # document, topological and code features
    #     elif features =='topo and code feats' or features=='feats_topo_code':
    #         # cfeats =[d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
    #         # cfeats.extend(topos)

    #         list_cols = list(range(3,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features == 'document':
    #         #cfeats = d['document']
    #         doc_feats = ''.join(''.join(X_train[j,0]))
            

    #     elif features == 'document_simplified':
    #         #cfeats = d['document_simplified']
    #         doc_feats = ''.join(''.join(X_train[j,1]))
            

    #     elif features =='document and list funcs':
    #         #cfeats =[d['document'], d['list_funcs']]
    #         doc_feats = ''.join(''.join(X_train[j,0] + ' ' + X_train[j,2]))
            

    #     elif features =='document_simplified and list funcs':
    #         #cfeats =[d['document_simplified'], d['list_funcs']]
    #         doc_feats = ''.join(''.join(X_train[j,1] + ' ' + X_train[j,2] ))
            
            

    #     # document and topological features
    #     elif features =='document and topo feats':
    #         #cfeats =[d['document'], d['list_funcs']]
    #         #cfeats.extend(topos)

    #         doc_feats = ''.join(''.join(X_train[j,0]))

    #         list_cols = list(range(10,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='document_simplified and topo feats':
    #         # cfeats =[d['document_simplified'], ]
    #         # cfeats.extend(topos)
    #         doc_feats = ''.join(''.join(X_train[j,1]))

    #         list_cols = list(range(10,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='document_simplified and list_funcs and topo feats':
    #         # cfeats =[d['document_simplified'], d['list_funcs']]
    #         # cfeats.extend(topos)

    #         doc_feats = ''.join(''.join(X_train[j,1] + ' ' + X_train[j,2] ))

    #         list_cols = list(range(10,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     # document and code features
    #     #code_feats = {
    #     #         'nregs': num_regs,
    #     #         'num_distinct_regs': num_distinct_regs,
    #     #         'ninstrs': num_instrs,
    #     #         'ndispls': num_displs,
    #     #         'nimms': num_imms,
    #     #         'nmaddrs': num_memaddrs,
    #     #         'num_funcs': num_funcs,
    #     #         'document': doc,              -> removed
    #     #         'document_simplified': doc2,  -> removed
    #     #         'list_regs': list_regs,       -> removed
    #     #         'list_funcs': list_funcs      -> removed
    #     #     }
    #     elif features =='document and code feats':
    #         #cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
            
    #         doc_feats = ''.join(''.join(X_train[j,0]))

    #         list_cols = list(range(3,10))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='document_simplified and code feats':
    #         #cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
    #         doc_feats = ''.join(''.join(X_train[j,1]))

    #         list_cols = list(range(3,10))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='document_simplified and list_funcs and code feats':
    #         #cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            

    #         doc_feats = ''.join(''.join(X_train[j,1] + ' ' + X_train[j,2] ))

    #         list_cols = list(range(3,10))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     # document, topological and code features
    #     elif features =='document and topo and code feats':
    #         #cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
    #         #cfeats.extend(topos)

    #         doc_feats = ''.join(''.join(X_train[j,0]))

    #         list_cols = list(range(3,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='document_simplified and topo and code feats':
    #         #cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
    #         #cfeats.extend(topos)
    #         doc_feats = ''.join(''.join(X_train[j,1]))

    #         list_cols = list(range(3,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            
            
    #     elif features =='document_simplified and list_funcs and topo and code feats':
    #         # cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
    #         # cfeats.extend(topos)
    #         doc_feats = ''.join(''.join(X_train[j,1] + ' ' + X_train[j,2] ))

    #         list_cols = list(range(3,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            

    #     elif features =='all':
    #         doc_feats = ''.join(''.join(X_train[j,0] + ' ' + X_train[j,2] ))

    #         list_cols = list(range(3,l))
    #         cfeats_train.extend(X_train[j,list_cols])
            
    #     X_tr.append(cfeats_train)
    #     X_tr_doc.append(doc_feats)
        
            
    # X_train_numeric_features  = np.array(X_tr)
    # X_train_doc = X_tr_doc 

    X_train_numeric_features, X_train_doc = filter_features_new(X_train, features)

    X_train_all = pd.DataFrame(
                data=X_train_numeric_features,
                )

    X_train_all['document'] = X_train_doc
   
    # transform all back to a dataframe
    # with first columns for the numeric features
    # last column for the document feature
    num_numeric_cols = X_train_numeric_features.shape[1]
    train_numeric_cols = list(range(0,num_numeric_cols))
    train_nlp_cols = list(range(num_numeric_cols, num_numeric_cols+1))


    return  X_train_all,train_numeric_cols,train_nlp_cols

def filter_features_new_v3(X_train,X_tfidf, features):
    """
    # IN:purged_list, features, dataset, 
    # OUT: X, X_doc
        get selected columns of topological features matrix, code features matrix , document(code bag of words)
        and return a numpy array/matrix for numeric features , 
            and another list for documents (only code as one big string possibly concatenated with the list of present func calls)

        this time a dataframe is returned

        transform the columns of the matrix in a dataframe
    """
 
    X_train_numeric_features, X_train_doc = filter_features_new(X_train, features.replace('tfidf','document'))

    X_train_all = pd.DataFrame(
                    data=np.concatenate(
                        (X_train_numeric_features,X_tfidf), 
                        axis=1
                    )
                )

    
    # transform all back to a dataframe
    # with first columns for the numeric features
    # last column for the document feature
    num_numeric_cols = X_train_numeric_features.shape[1]
    train_numeric_cols = list(range(0,num_numeric_cols))
    train_nlp_cols = list(range(num_numeric_cols,X_train_all.shape[1]))

    # print("numeric_cols list ",train_numeric_cols)
    # print("tfidf_cols list ",train_nlp_cols)

    return  X_train_all,train_numeric_cols,train_nlp_cols

       


def load_dataset_split(folder='tmp/symbols_dataset_3_precomp_split_unchanged'):
    X_train = pickle.load(open(os.path.join(folder,'X_train.pickle'),'rb'))
    X_test = pickle.load(open(os.path.join(folder,'X_test.pickle'),'rb'))
    y_train = pickle.load(open(os.path.join(folder,'y_train.pickle'),'rb'))
    y_test = pickle.load(open(os.path.join(folder,'y_test.pickle'),'rb'))
    nclasses = pickle.load(open(os.path.join(folder,'nclasses.pickle'),'rb'))

    return X_train, X_test, y_train, y_test, nclasses


def dataset_split_core(dataset, purged_list, nclasses):
    n = len(purged_list)
    y = [dataset[j].y for j in purged_list]
    y = np.array(y)

    X = pack_features(dataset,purged_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        #stratify=y,
        random_state = 7)

    return X_train, X_test, y_train, y_test, nclasses


def dataset_split_core_graph_version(dataset, purged_list, nclasses):

    n = len(purged_list)
    y = [dataset[j].y for j in purged_list]
    y = np.array(y)

    X = pack_features_graph_version(dataset,purged_list)

    # make X have a last column with the dataset index

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        #stratify=y,
        random_state = 7)

    # Build the train and test dataset with X last columns which contains the dataset index
    train_list = X_train[:,-1].tolist()
    test_list = X_test[:,-1].tolist()
    train_dataset = dataset[torch.LongTensor(train_list)]
    test_dataset = dataset[torch.LongTensor(test_list)]

    # remove X last columns which contains the dataset index
    X_train = X_train[:,:-1]
    X_test  = X_test[:,:-1]

    return X_train, X_test, y_train, y_test, nclasses, train_dataset, test_dataset


def dataset_split_shared_splits(dataset, features='all', min_count=0):
    """
    purge minimum classes, prepare X and y and then do train test splits

    change it , set all featues topo + code
    then if the alg is just for topo then just remove code features
    """

    purged_list, nclasses = purge_minimum_classes(dataset,min_count)

    return dataset_split_core(dataset, purged_list, nclasses)
    # n = len(purged_list)
    # y = [dataset[j].y for j in purged_list]
    # y = np.array(y)

    # X = filter_features(dataset,purged_list, features)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, 
    #     y,
    #     #stratify=y,
    #     random_state = 7)

    # return X_train, X_test, y_train, y_test, nclasses



def dataset_split_shared_splits_graph_version(dataset, features='all', min_count=0):
    """
    purge minimum classes, prepare X and y and then do train test splits

    change it , set all featues topo + code
    then if the alg is just for topo then just remove code features
    """

    purged_list, nclasses = purge_minimum_classes(dataset,min_count)

    return dataset_split_core_graph_version(dataset, purged_list, nclasses)
   
    

def dataset_split_balanced_major_classes(dataset, features='all', max_count=-1):
    """
    undersample majority classes, minority classes intact
    """

    purged_list, nclasses = balance_maximum_classes(dataset,max_count)
    return dataset_split_core(dataset, purged_list, nclasses)
    
    # n = len(purged_list)
    # y = [dataset[j].y for j in purged_list]
    # y = np.array(y)

    # X = filter_features(dataset,purged_list, features)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, 
    #     y,
    #     #stratify=y,
    #     random_state = 7)

    # return X_train, X_test, y_train, y_test, nclasses


def dataset_split_balanced_major_classes_graph_version(dataset, features='all', max_count=-1,min_remove=-1, remove_classes=[]):
    """
    undersmaple major classes, min classes intact
    """

    purged_list, nclasses = balance_maximum_classes(dataset,max_count,min_remove, remove_classes=remove_classes)
    return dataset_split_core_graph_version(dataset, purged_list, nclasses)
    


class mlp1(torch.nn.Module):
    def __init__(self, d1=10,d2=20,num_classes=3):
        super(mlp1, self).__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_classes)
        
    def forward(self, data):
        x = data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

class mlp2(torch.nn.Module):
    def __init__(self, d1=10,d2=20,d3=30,num_classes=3):
        super(mlp2, self).__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2,d3)
        self.fc3 = nn.Linear(d3, num_classes)
        
    def forward(self, data):
        x = data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x


class mlp3(torch.nn.Module):
    def __init__(self, d1=10,d2=20,d3=30,d4=20,num_classes=3):
        super(mlp3, self).__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2,d3)
        self.fc3 = nn.Linear(d3, d4)
        self.fc4 = nn.Linear(d4, num_classes)
        
    def forward(self, data):
        x = data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        
        return x



class mlp4(torch.nn.Module):
    def __init__(self, d1=10,d2=20,d3=30,d4=20,d5=24,num_classes=3):
        super(mlp4, self).__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2,d3)
        self.fc3 = nn.Linear(d3, d4)
        self.fc4 = nn.Linear(d4, d5)
        self.fc5 = nn.Linear(d5, num_classes)
        
    def forward(self, data):
        x = data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        return x




def prepare_nn_models_quick():
    models_params = {

        mlp1.__name__: {
            'model': mlp1,
            'params_set': [
                # {
                #  'd2': [50,100,300],
                #  'num_epochs': [100,300,],
                #  },
                 {
                 'd1': [75],
                 'd2': [10],
                 'num_epochs': [100],
                 }
                 
            ]
        },
    }

    return models_params



def prepare_nn_models():
    models_params = {

        mlp1.__name__: {
            'model': mlp1,
            'params_set': [
                # {
                #  'd2': [50,100,300],
                #  'num_epochs': [100,300,],
                #  },
                 {
                 'd1': [20,50,100],
                 'd2': [10,20],
                 'num_epochs': [150],
                 }
                 
            ]
        },
        mlp2.__name__: {
            'model': mlp2,
            'params_set': [
                {
                 'd2': [50,100,150],
                 'd3': [5,50,200],
                 'num_epochs': [100,200],
                 },                 
            ]
        },

    }

    return models_params


def prepare_models_quick():
    """
    dictionary of model classes and all their possible parameter values to train
    """
    models_params = {

        LogisticRegression.__name__: {
            'model': LogisticRegression,
            'params_set': [
                {
                 'multi_class': ['ovr'],
                 #'penalty': ['l1','l2'],
                 'penalty': ['l2'],
                 'solver': ['newton-cg'],
                 'max_iter': [100,], #'max_iter': [200],
                 #'C':[1,10,100,1000],
                 'C':[1],
                 },
            ]
        },
        RandomForestClassifier.__name__: {
            'model': RandomForestClassifier,
            'params_set': [
                {
                 # 'n_estimators': [4,16,50,100,500],
                 # 'max_depth': [2,4,8,16],
                 'n_estimators': [16],
                 'max_depth': [8],
                 },
            ]
        },
    }
    return models_params



def prepare_models():
    models_params = {

        LogisticRegression.__name__: {
            'model': LogisticRegression,
            'params_set': [
                {
                 'multi_class': ['ovr'],
                 #'penalty': ['l1','l2'],
                 'penalty': ['l2'],
                 'solver': ['newton-cg'],
                 'max_iter': [200],
                 #'C':[1,10,100,1000],
                 'C':[1,10],

                 },
                 
            ]
        },
        DecisionTreeClassifier.__name__: {
            'model': DecisionTreeClassifier,
            'params_set': [
                {
                 #'max_depth': [2,4,8,16],
                 'max_depth': [2,4],
                 },
            ]
        },
        RandomForestClassifier.__name__: {
            'model': RandomForestClassifier,
            'params_set': [
                {
                 'n_estimators': [4,16,50,100,500],
                 'max_depth': [2,4,8,16],
                 },
            ]
        },
        XGBClassifier.__name__: {
            'model': XGBClassifier,
            'params_set': [
                {
                 'booster': ['gbtree','dart'],
                 'max_depth': [2,4,8,16],
                 },
            ]
        },
        XGBClassifier.__name__+'linear': {
            'model': XGBClassifier,
            'params_set': [
                {
                 'lambda': [0,0.1,1,10],
                 },
            ]
        },
        # SVC.__name__: {
        #     'model': SVC, 
        #     'params_set': [
        #         {'kernel': ['rbf'],
        #          #'gamma': [1e-3,1e-4],
        #          'C':[1,10,100,1000]},

        #         {'kernel': ['linear'],
        #          'C':[1,10,100,1000]}
        #     ]
        # },


    }


    return models_params



def cv_train_models(X_train, y_train, X_test, y_test, models_params, scores, features='', dataset_version='',results_folder='baseline', models_folder='models/'):
    """
    for each model type and score, do CV, retrain and testing , then save to results_dict
    """

    results_dict={}

    for model_name in models_params.keys():
        print("Training ", model_name)
        model_dict = models_params[model_name]
        model = model_dict['model']()
        parameters = model_dict['params_set']
        
        # now each model_name is trained with all params and then saved
        results_dict = {}
        results_dict[model_name]={}

        for score in scores:
            try:
                start = time.time()
        
                print("GridseachCV for ", score)
                clf = GridSearchCV(model, parameters, cv=3, scoring=score, n_jobs=2)
                clf.fit(X_train, y_train)
                y_true, y_pred = y_test, clf.predict(X_test)

                end = time.time()

                
                results_dict[model_name][score] = classification_report(y_true, y_pred, output_dict=True)
                
                results_dict[model_name][score]['params'] = clf.best_params_
                results_dict[model_name][score]['cv_score'] = clf.best_score_
                results_dict[model_name][score]['time'] = round(end-start)

                results_dict[model_name][score]['model_instance'] = clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

        save_results(results_dict, features,dataset_version, results_folder, models_folder=models_folder)


    return results_dict








class CustomDataset(Dataset):
    """
        Used for creating a data loader in torch.
        If you just need this code, you can use TensorDataset

    """
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


# def test_nn_model(X_test, y_test, model_instance,scores, nclasses, results_dict ):
#     global device 
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
#     results_dict['test_nn']={}
#     start = time.time()

#     model = model_instance
#     model = model.to(device)
    
#     test_data = CustomDataset(
#         torch.from_numpy(X_test).float(), 
#         torch.from_numpy(y_test).long())

    
#     # now test it
#     model.eval()
#     X_test_gpu = test_data.x.to(device)
#     y_test_gpu = test_data.y.to(device)
#     out = model(X_test_gpu)
#     # now select the classes by the max softmax value
#     _, y_test_pred = out.max(dim=1)

#     # pprint(y_test_gpu[:3])
#     # pprint(y_test_pred[:3])
    
#     error_measures = F1Score(y_test_pred.flatten().tolist(), 
#                        y_test_gpu.flatten().tolist(),
#                        nclasses)

#     #test_loss = F.nll_loss(out, y_test_gpu)
#     #pprint(test_loss.to('cpu').detach().numpy().item())
#     #test_error = test_loss.to('cpu').detach().numpy().item()

#     end = time.time()
    
    
#     score = scores[0]
#     results_dict['test_nn'][score] = error_measures[score]
#     results_dict['test_nn']['time'] = round(end-start)

#     return results_dict



def train_nn_model_one_fold (X_train, y_train, X_test, y_test, nn_model_params, scores, nclasses, extra_params=None):
    """
        This function is called for each k-fold cv run.
        So here X_test and y_test correspond to the cv testing fold and X_train and y_train to the num folds for training 
    """
    # print("in train_nn_model_one_fold")
    # pprint(nn_model_params)
    global device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dict={} 

    
    #results_dict['test_nn']={}
    start = time.time()

    model_class = nn_model_params['model_class']
    model = model_class(
            **nn_model_params['model_kwargs']
            )
    
    model_name = model.__class__.__name__


    results_dict[model_name]={}
    model = model.to(device)
    model.train()

    loss_history = []
    cv_avg_f1 = 0

    num_epochs = nn_model_params['num_epochs']
    train_data = CustomDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long())
    test_data = CustomDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long())

    loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    
    total_num_graphs = len(X_train)

    optimizer = torch.optim.Adam(model.parameters())


    for e in range(num_epochs):
        loss_train = 0.0

        cv_avg_f1 = 0
        
        total_num_graphs==0
        for X,y in loader:
            
            
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            out = model(X)
            target = y 


            #---correct-way----------
            target = torch.squeeze(y)
            
            # print(out.shape)
            # pprint(out)
            # #print(out)
            # print(target.shape)
            # pprint(target)
            #print(target)
            #loss = F.nll_loss(nn.LogSoftmax(out), target)
            m = nn.LogSoftmax(dim=1)
            loss_func = nn.NLLLoss()
            #loss = loss_func(m(out), target)
            loss = loss_func(out, target)
            #----old-way-------
            # print(out.shape)
            # print(target.shape)
            # loss = F.nll_loss(out, target)
            

            loss_train +=loss
    
            loss.backward()
            optimizer.step()
            total_num_graphs+=X.shape[0]

            cv_avg_f1 += f1_score(
                        target.flatten().tolist(),
                        torch.argmax(out, dim=1).flatten().tolist(),
                        average='micro' )

        cv_avg_f1 = cv_avg_f1 / len(loader)
        loss_train = loss_train /total_num_graphs
        loss_history.append(loss_train)
    


    # now test it
    model.eval()
    X_test_gpu = test_data.x.to(device)
    y_test_gpu = test_data.y.to(device)
    out = model(X_test_gpu)
    # now select the classes by the max softmax value
    _, y_test_pred = out.max(dim=1)

    # pprint(y_test_gpu[:3])
    # pprint(y_test_pred[:3])
    
    # error_measures = F1Score(y_test_pred.flatten().tolist(), 
    #                    y_test_gpu.flatten().tolist(),
    #                    nclasses)

    # error_measures = {
    #     'f1_micro':f1_score(
    #                     y_test_gpu.flatten().tolist(),
    #                     y_test_pred.flatten().tolist(),
    #                     average='micro' )
    #     }

    error_measures = classification_report(
                        y_test_gpu.flatten().tolist(),
                        y_test_pred.flatten().tolist(),
                        output_dict=True)

    #test_loss = F.nll_loss(out, y_test_gpu)
    #pprint(test_loss.to('cpu').detach().numpy().item())
    #test_error = test_loss.to('cpu').detach().numpy().item()

    end = time.time()
    
    
    # since the classes are balanced, we can use macro average
    score = scores[0]
    results_dict[model_name][score] = error_measures
    results_dict[model_name][score]['cv_score'] = cv_avg_f1

    if extra_params is not None:
        nn_model_params['model_kwargs'].update(extra_params)
    results_dict[model_name][score]['params'] = nn_model_params['model_kwargs']
    
    results_dict[model_name][score]['num_epochs'] = num_epochs

    results_dict[model_name][score]['time'] = round(end-start)


    return results_dict, model


def train_nn_model_cv(X_train, y_train, X_test, y_test, nn_model_params, scores, nclasses, numfolds=3 ):

    """
        train each model given a param combination doing cross-validation and testing the best model

        - given a param combination instanciate a model
            - for each cvfold do training
                - for each epoch do training
                    - for each batch do sgd
                - save cv_errors
            - average cv_error 
            - select best performing model
        - retrain best model on all X_train 
        - test model  


    """

    # print("in train_nn_model_cv")
    # pprint(nn_model_params)

    if isinstance(X_train, np.ndarray):
        X_train2 = X_train 
    else:
        X_train2 = X_train.toarray()

    if isinstance(X_test, np.ndarray):
        X_test2 = X_test
    else:
        X_test2 = X_test.toarray()

    kfld = KFold(n_splits=numfolds)
    cv_error_score_history = []
    num_fold = 0
    for kf in kfld.split(X_train) :
        """
            for each split
                train num_epochs
                save loss_history
                save error_history
            
            average the cv_error_score

            select the best model

            retrain 

            test 
        """

        cv_fold_results_dict, _ =  train_nn_model_one_fold(X_train2, y_train, X_test2, y_test, nn_model_params, scores, nclasses )

        model_name = ''
        for k,v in cv_fold_results_dict.items():
            model_name=k

        score = scores[0]
        # only key at first level is the score ()
        # for k,v in cv_fold_results_dict[model_name].items():
        #     score=k

        cv_error_score_history.append(cv_fold_results_dict[model_name][score]['cv_score'])

        # print(" kold ",num_fold," error measure(score)=",cv_fold_results_dict[model_name][score]['cv_score'])
        num_fold+=1



    # average it's error measure (or take it already averaged)
    cv_avg_error_score = 0
    for e in cv_error_score_history:
        cv_avg_error_score+= e 
    cv_avg_error_score = cv_avg_error_score / numfolds 

    results_dict = cv_fold_results_dict 
    results_dict['cv_avg_error'] = cv_avg_error_score 

    # print("\n after cv, error history is")
    # pprint(cv_error_score_history)
    # print(" cv avg error i ",cv_avg_error_score)

    return results_dict


def unroll_all_possible_pipeline_model_combos( nn_models_params, nclasses, tfvec_params):

    """
        1)update each dict with tfvec_params
        
        for model_class_key,v in nn_models_params.items():
                v.update(copy.deepcopy(tfvec_params))

        2) call  ParameterGrid for each model 
        from sklearn.model_selection import ParameterGrid
        param_grid = {'a': [1, 2], 'b': [True, False]}
        list(ParameterGrid(param_grid))

    """
    all_combos = []

    v = nn_models_params
   
    #pprint(v)
    list_params_sets = ParameterGrid(v)
    all_combos.extend(list(list_params_sets))

    return all_combos


def unroll_all_possible_model_combos_with_tfidf( nn_models_params, nclasses, tfvec_params):

    """
        1)update each dict with tfvec_params
        
        for model_class_key,v in nn_models_params.items():
                v.update(copy.deepcopy(tfvec_params))

        2) call  ParameterGrid for each model 
        from sklearn.model_selection import ParameterGrid
        param_grid = {'a': [1, 2], 'b': [True, False]}
        list(ParameterGrid(param_grid))

    """
    all_combos = []
    for model_class_key,v in nn_models_params.items():
        pprint(model_class_key)
        pprint(v)
        for m in v['params_set']:
            m['model_class'] = [v['model']]
            m.update(copy.deepcopy(tfvec_params))
            if 'num_epochs' not in m.keys():
                m['num_epochs'] = [50]
            if 'num_classes' not in m.keys():
                m['num_classes'] = [nclasses]

        pprint(v)

        

        list_params_sets = ParameterGrid(v['params_set'])
        all_combos.extend(list(list_params_sets))

    return all_combos



def unroll_all_possible_model_combos(X_train, nn_models_params, nclasses):
    """
    using parameterGrid to unfold(combinatorial explosion) of models and their param combinations
    """


    # get X number of columns
    n_X_cols = X_train.shape[1]
    print("n_X_cols for the nn: ", n_X_cols, " and X shape ",  X_train.shape)

    all_combos = []

    nn_models_params_unrolled = []
    for model_class_key,v in nn_models_params.items():
        # exclude model_class
        # unfold loop for each parameter
        # for each k get all values
        # print("model", model_class_key)
        # pprint(v)

        all_model_combos = []
        for params_set in v['params_set']:

            # each params_set is a dict with k, v_lists 
            # all_lists must be 
            # [[(k1,v11),(k1,v12),(k1,v13)],
            #  [(k2,v21),(k2,v22),(k2,v23)],
            #  [(k3,v31),(k3,v32),(k3,v33)],
            # ]

            print("params_set")
            pprint(params_set)

            all_list = []
            for kp,vp in params_set.items():
                all_list.append([(kp,vpi) for vpi in vp])

            # print("all_list")
            # pprint(all_list)

            all_combinations_of_params = list(itertools.product(*all_list)) 
            # [ ((k1,v11),(k2,v21),(k3,v31)),
            #   ((k1,v12),(k2,v21),(k3,v31)),
            #   ...
            #]

            # print("all_permutaions")
            # pprint(all_combinations_of_params)


            # now add model and unfold the tuples with the key?
            for combo in all_combinations_of_params:
                # transform into a dict with 
                # model_class
                # num_epochs (if not in kwargs then put 50 as a default)
                # model_kwargs
                params_dict = {
                    'model_class': v['model'],
                    'model_kwargs': {
                        t[0]: t[1]
                        for t in combo
                        if t[0]!='num_epochs'
                    },
                }

                # print("combo")
                # pprint(combo)
                # print("params_dict")
                # pprint(params_dict)
                # params names
                params_names = [t[0] for t in combo]
                #pprint(params_names)
                if 'num_epochs' not in params_names:
                    params_dict['num_epochs'] = 50
                else:
                    index_of_num_epochs = params_names.index('num_epochs')
                    params_dict['num_epochs'] = combo[index_of_num_epochs][1]

                #num classes
                params_dict['model_kwargs']['num_classes'] = nclasses

                #size of input = num X columns 
                params_dict['model_kwargs']['d1'] = n_X_cols

                all_model_combos.append(params_dict)
        all_combos.extend(all_model_combos)

    return all_combos






def nn_train_models(X_train, y_train, X_test, y_test, nn_models_params, scores, nclasses, numfolds=3, features='', dataset_version='', results_folder='baseline', models_folder='models/'):
    """
    This function will take all model parameter sets, and create an iterator over all combinations.

    for each combination of parameter values and model class, it will launch a cross validation training. 

    Then later it will retrain the best performing model over the full training set

    Then it will test its prediction and write down the results

    """
    print("\nnn_train_models, nclasses=",nclasses)


    # prepare all model candidates
    nn_models_params_unrolled = unroll_all_possible_model_combos(X_train, nn_models_params, nclasses)

    #pprint(nn_models_params_unrolled)
        
    # CV training for all candidates
    error_scores = []
    for param_set in nn_models_params_unrolled:
        try:
            results_dict = train_nn_model_cv(X_train, y_train, X_test, y_test, param_set, scores, nclasses, numfolds=3 )
            error_scores.append(results_dict['cv_avg_error'])
        except Exception as err:
            print("Error with "+param_set['model']+" and "+scores[0])
            traceback.print_exc()

    # choose best model 
    best_error = max(error_scores)
    best_error_index = error_scores.index(best_error)
    best_model_params = nn_models_params_unrolled[best_error_index]
    
    # print("\nBest model params, with error ",best_error)
    # pprint(best_model_params)
    # print("\nall error scores ")
    # pprint(error_scores)


    # now retrain and test the model
    try:
        results_dict, model = train_nn_model_one_fold(X_train, y_train, X_test, y_test, best_model_params, scores, nclasses )
    except Exception as err:
            print("Error with "+param_set['model']+" and "+scores[0])
            traceback.print_exc()

    # reorganize results
    score = scores[0]
    model_name = list(results_dict.keys())[0]
    results_dict[model_name][score]['score']= scores[0]
    results_dict[model_name][score]['model_instance']=model

    # print("\n After retraining: ")
    # pprint(results_dict)
    
    save_results(results_dict, features, dataset_version, results_folder=results_folder, models_folder=models_folder)

    
    return results_dict




def baseline_training_and_testing(X_train, X_test, y_train, y_test, features, dataset_version='v1',nclasses=3, results_folder='baseline', baseline_models=None, baseline_nn_models=None, models_folder='models/baseline_models/'):
    
    """
        Filter features following the indication
    """
    X_train_numeric, X_train_doc = filter_features_new(X_train, features)
    X_test_numeric, X_test_doc = filter_features_new(X_test, features)


    if baseline_models is None:
        models_and_params = prepare_models()
    else:
        models_and_params = baseline_models

    results_dict = cv_train_models(
        X_train_numeric, y_train, 
        X_test_numeric, y_test, 
        models_and_params, 
        scores=['f1_micro',],
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder)


    if baseline_nn_models is None:
        nn_models_and_params = prepare_nn_models()
    else:
        nn_models_and_params = baseline_nn_models

    results_dict = nn_train_models(
        X_train_numeric, y_train, 
        X_test_numeric, y_test, 
        nn_models_and_params, 
        scores=['f1_micro'], nclasses=nclasses, 
        numfolds=3,
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder
        )
    

    




if __name__=='__main__':


    dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    dataset_version='v1'
    # print(len(dataset))
    # print(dataset.num_classes)
    # print(dataset.num_features)

    # same randomization for all
    X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features='all', min_count=0)

    nlp_nn_training_and_testing(X_train, X_test, y_train, y_test,'document',dataset_version,nclasses)

    print_training_stats('v1',results_folder='nlp')

    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'x_topo_feats',dataset_version,nclasses, models_folder='models/baseline_models')
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'code_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'feats_topo_code',dataset_version)
    

    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    # dataset_version='v2'
    # # # print(len(dataset))
    # # # print(dataset.num_classes)
    # # # print(dataset.num_features)

    # # same randomization for all
    # X_train, X_test, y_train, y_test = dataset_split_shared_splits(dataset)

    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'x_topo_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'code_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'feats_topo_code',dataset_version)
    


    #print_training_stats('v1')

    #print_training_stats('v2')