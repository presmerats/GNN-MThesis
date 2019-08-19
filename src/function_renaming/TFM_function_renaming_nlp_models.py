import pandas as pd 
import numpy as np 
import os 
from os import environ, path 
from pprint import pprint 
import copy

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from datetime import datetime

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset


def purge_minimum_classes(dataset, min_count=40):

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

    max_class_count = min(class_counts.values())
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




def dataset_split_shared_splits(dataset, features, min_count):
    """
    change it , set all featues topo + code
    then if the alg is just for topo then just remove code features
    """

    purged_list, nclasses = purge_minimum_classes(dataset, min_count)

    n = len(purged_list)
    y = [dataset[j].y for j in purged_list]
    y = np.array(y)

    X = []
    for j in purged_list:
        d = dataset[j].__getattribute__('code_feats')

        # document features only 
        if features == 'document':
            cfeats = d['document']

        elif features == 'document_simplified':
            cfeats = d['document_simplified']

        elif features =='document and list funcs':
            cfeats =[d['document'], d['list_funcs']]

        elif features =='document_simplified and list funcs':
            cfeats =[d['document_simplified'], d['list_funcs']]

        # document and topological features
        elif features =='document and topo feats':
            cfeats =[d['document'], d['list_funcs']]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)

        elif features =='document_simplified and topo feats':
            cfeats =[d['document_simplified'], ]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)

        elif features =='document_simplified and list_funcs and topo feats':
            cfeats =[d['document_simplified'], d['list_funcs']]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)

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
            cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]

        elif features =='document_simplified and code feats':
            cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
        elif features =='document_simplified and list_funcs and code feats':
            cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            
        # document, topological and code features
        elif features =='document and topo and code feats':
            cfeats =[d['document'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],  ]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)

        elif features =='document_simplified and topo and code feats':
            cfeats =[d['document_simplified'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)
            
        elif features =='document_simplified and list_funcs and topo and code feats':
            cfeats =[d['document_simplified'], d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)

        X.append(cfeats)
            

    X = np.array(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        #stratify=y,
        random_state = 7)

    return X_train, X_test, y_train, y_test, nclasses



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
                 'C':[10],

                 },
                 
            ]
        },
        # DecisionTreeClassifier.__name__: {
        #     'model': DecisionTreeClassifier,
        #     'params_set': [
        #         {
        #          #'max_depth': [2,4,8,16],
        #          'max_depth': [4,8],
        #          },
        #     ]
        # },
        RandomForestClassifier.__name__: {
            'model': RandomForestClassifier,
            'params_set': [
                {
                 # 'n_estimators': [4,16,50,100,500],
                 # 'max_depth': [2,4,8,16],
                 'n_estimators': [50,100],
                 'max_depth': [4,8],
                 },
            ]
        },
        # XGBClassifier.__name__: {
        #     'model': XGBClassifier,
        #     'params_set': [
        #         {
        #          'booster': ['gbtree','dart'],
        #          'max_depth': [2,4,8,16],
        #          },
        #     ]
        # },
        # XGBClassifier.__name__+'linear': {
        #     'model': XGBClassifier,
        #     'params_set': [
        #         {
        #          'lambda': [0,0.1,1,10],
        #          },
        #     ]
        # },
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



def cv_train_models(X_train, y_train, X_test, y_test, models_params, scores ):

    results_dict={}

    for model_name in models_params.keys():

        print("Training ", model_name)
        model_dict = models_params[model_name]
        model = model_dict['model']()
        parameters = model_dict['params_set']
        results_dict[model_name]={}


        tf_params = {
             'tvec__max_features':[100, 2000],
             'tvec__ngram_range': [(1, 2), (2, 3), (3, 3)],
             #'tvec__stop_words': [None, 'english'],
             'tvec__max_df': [0.8],
             'tvec__min_df': [0.1]
            }

        final_parameters = []

        for param_set in parameters:
            final_parameter = copy.deepcopy(tf_params)

            for k,v in param_set.items():
                final_parameter['clf__'+k]=v

            final_parameters.append(final_parameter)

        t_pipe = Pipeline([
              ('tvec', TfidfVectorizer()),
              ('clf', model_dict['model']())
            ])

        for score in scores:
            try:
                start = time.time()
        
                print("GridseachCV for ", score)
                clf = GridSearchCV(t_pipe, param_grid=final_parameters, cv=3, verbose=1, scoring=score, n_jobs=1)
                clf.fit(X_train, y_train)
                y_true, y_pred = y_test, clf.predict(X_test)

                end = time.time()

                
                results_dict[model_name][score] = classification_report(y_true, y_pred, output_dict=True)
                results_dict[model_name][score]['params'] = clf.best_params_
                results_dict[model_name][score]['cv_score'] = clf.best_score_
                results_dict[model_name][score]['time'] = round(end-start)

                #results_dict[model_name]['best']=clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

    return results_dict


def save_results(results_dict, features, dataset_version, min_count):
    # append results to results_dict on disk
    if dataset_version=='v1':
        r = json.load(open('nlp_training_results.json','r'))
    else:
        r = json.load(open('nlp_training_results_v2.json','r'))

    # how to merge? , just append? or merge for each model?
    for model_name in results_dict.keys():
        if model_name not in r.keys():
            r[model_name]={}

        for score,score_val in results_dict[model_name].items():
            score_val['features'] = features
            score_val['min_count'] = min_count
            r[model_name][score + datetime.now().strftime("%Y-%m-%d_%H_%M_%S")] = score_val 
    

    
    if dataset_version=='v1':
        json.dump(r, open('nlp_training_results.json','w'))
    else:
        json.dump(r, open('nlp_training_results_v2.json','w'))
    #pprint(results_dict)
    #return results_dict


def nlp_models_training_and_testing(X_train, X_test, y_train, y_test, features, dataset_version='v1',min_count=100):
    
    """
        Filter features following the indication
    """
    
    models_and_params = prepare_models()
    results_dict = cv_train_models(
        X_train, y_train, 
        X_test, y_test, 
        models_and_params, 
        #scores=['recall_macro','recall_micro','precision_macro','precision_micro','f1_macro','f1_micro',],
        scores=['recall_micro','f1_micro','precision_micro'],
        )

    save_results(results_dict, features,dataset_version, min_count)


def print_training_stats(dataset_version='v1'):
    """
    read json from disk and print the best model of each model type.
    print also it's scores obviously
    """
    
    if dataset_version=='v1':
        r = json.load(open('nlp_training_results.json','r'))
    else:
        r = json.load(open('nlp_training_results_v2.json','r'))


    info_res = [ (model,score,score_res['cv_score'], score_res['params'], score_res['features'],score_res['min_count']) for model,model_res in r.items() 
                   for score,score_res in model_res.items() ]

    # for each modell print the best result
    models = list(set([ a[0] for a in info_res]))

    best_models = []    
    for model in models:

        scores = [ a[2] for a in info_res if a[0]==model]
        params = [ a[3] for a in info_res if a[0]==model]
        score_names = [ a[1] for a in info_res if a[0]==model]
        features = [ a[4] for a in info_res if a[0]==model]
        min_counts_for_classes = [ a[5] for a in info_res if a[0]==model]
        max_score = max(scores)
        max_model = scores.index(max_score)
        max_model_params = params[max_model] # later this will be params
        score_name = score_names[max_model]
        best_models.append((model, score_name, max_score, max_model_params,features[max_model], min_counts_for_classes[max_model]))

    pprint(best_models)


if __name__=='__main__':

    """
        The BoW TF-IDF models will perform the following:

        - purge minimal classes
        - split dataset
            - prepare code features accordingly
        - prepare models
            -> classifiers used on top of TfidfVectorizer
        - train models
        - save and present results

        -  combine BowTfidf with other features??

    """


    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    # dataset_version='v1'
    # features = 'document'
    # min_count=600
    # X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)


    # features = 'document'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document and list funcs'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and list funcs'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')


    # features = 'document and topo feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and topo feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and list_funcs and topo feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')


    # features = 'document and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and list_funcs and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')


    # features = 'document and topo and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and topo and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and list_funcs and topo and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    



    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    dataset_version='v2'
    features = 'document'
    min_count=30
    X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)


    features = 'document'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v2')

    features = 'document_simplified'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v2')

   