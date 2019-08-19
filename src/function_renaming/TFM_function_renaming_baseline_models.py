import pandas as pd 
import numpy as np 
import os 
from os import environ, path 
from datetime import datetime

from pprint import pprint 


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


def filter_features(X_train, X_test, features):
    
    X_train_filtered = []
    X_test_filtered = []
    if features == 'feats_topo_code':
        # do nothing
        return X_train, X_test

            # X = []
            # for j in purged_list:
            #     d = dataset[j].__getattribute__('code_feats')
            #     d.pop('document_simplified') 
            #     d.pop('document')  
            #     d.pop('list_regs') 
            #     d.pop('list_funcs')
                
            #     cfeats = list(d.values())
            #     topos = dataset[j].__getattribute__('x_topo_feats')
            #     cfeats.extend(topos)
            #     X.append(cfeats)
            
    elif features != 'code_feats':
        # x_topo_features selected
        # code_feats = {
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
        # remove the first 7 columns, 
        col_idx = np.array(list(range(7,X_train.shape[1])))
        X_train_filtered = np.array(X_train[:, col_idx], copy=True)
        X_test_filtered = np.array(X_test[:, col_idx], copy=True)
        
    else:
        # code_feats selected
        col_idx = np.array(list(range(0,7)))
        X_train_filtered = np.array(X_train[:, col_idx], copy=True)
        X_test_filtered = np.array(X_test[:, col_idx], copy=True)

    return X_train_filtered, X_test_filtered


def dataset_split(dataset, features):
    """
    change it , set all featues topo + code
    then if the alg is just for topo then just remove code features
    """

    purged_list = purge_minimum_classes(dataset,0)

    n = len(purged_list)
    y = [dataset[j].y for j in purged_list]
    y = np.array(y)

    if features == 'feats_topo_code':
        X = []
        for j in purged_list:
            d = dataset[j].__getattribute__('code_feats')
            d.pop('document_simplified') 
            d.pop('document')  
            d.pop('list_regs') 
            d.pop('list_funcs')
            
            cfeats = list(d.values())
            topos = dataset[j].__getattribute__('x_topo_feats')
            cfeats.extend(topos)
            X.append(cfeats)
            
    elif features != 'code_feats':
        X = [dataset[j].__getattribute__(features) 
             for j in purged_list]
    else:
        X = []
        for j in purged_list:
            d = dataset[j].__getattribute__(features)
            d.pop('document_simplified') 
            d.pop('document')  
            d.pop('list_regs') 
            d.pop('list_funcs')
            X.append(list(d.values()))
        
    X = np.array(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        #stratify=y,
        random_state = 7)

    return X_train, X_test, y_train, y_test


def dataset_split_shared_splits(dataset):
    """
    change it , set all featues topo + code
    then if the alg is just for topo then just remove code features
    """

    purged_list, nclasses = purge_minimum_classes(dataset,0)

    n = len(purged_list)
    y = [dataset[j].y for j in purged_list]
    y = np.array(y)

    X = []
    for j in purged_list:
        d = dataset[j].__getattribute__('code_feats')
        d.pop('document_simplified') 
        d.pop('document')  
        d.pop('list_regs') 
        d.pop('list_funcs')
        
        cfeats = list(d.values())
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


def prepare_nn_models():
    models_params = {

        mlp1.__name__: {
            'model': mlp1,
            'params_set': [
                {
                 #'d1': [1],
                 'd2': [5,10],
                 'num_epochs': [2],
                 },
                 {
                 'd1': [1],
                 'd2': [3],
                 'num_epochs': [2],
                 }
                 
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
        # DecisionTreeClassifier.__name__: {
        #     'model': DecisionTreeClassifier,
        #     'params_set': [
        #         {
        #          #'max_depth': [2,4,8,16],
        #          'max_depth': [2,4],
        #          },
        #     ]
        # },
        # RandomForestClassifier.__name__: {
        #     'model': RandomForestClassifier,
        #     'params_set': [
        #         {
        #          'n_estimators': [4,16,50,100,500],
        #          'max_depth': [2,4,8,16],
        #          },
        #     ]
        # },
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

                #results_dict[model_name]['best']=clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

    return results_dict








class CustomDataset(Dataset):
    """
        If you just need this code, you can use TensorDataset
    """
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def test_nn_model(X_test, y_test, model_instance,scores, nclasses, results_dict ):
    global device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    results_dict['test_nn']={}
    start = time.time()

    model = model_instance
    model = model.to(device)
    
    test_data = CustomDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long())

    
    # now test it
    model.eval()
    X_test_gpu = test_data.x.to(device)
    y_test_gpu = test_data.y.to(device)
    out = model(X_test_gpu)
    # now select the classes by the max softmax value
    _, y_test_pred = out.max(dim=1)

    # pprint(y_test_gpu[:3])
    # pprint(y_test_pred[:3])
    
    error_measures = F1Score(y_test_pred.flatten().tolist(), 
                       y_test_gpu.flatten().tolist(),
                       nclasses)

    #test_loss = F.nll_loss(out, y_test_gpu)
    #pprint(test_loss.to('cpu').detach().numpy().item())
    #test_error = test_loss.to('cpu').detach().numpy().item()

    end = time.time()
    
    score = 'l2'
    results_dict['test_nn'][score] = error_measures['macroF1']
    results_dict['test_nn']['time'] = round(end-start)

    return results_dict



def train_nn_model_one_fold (X_train, y_train, X_test, y_test, nn_model_params, scores, nclasses ):
    """
        This function is called for each k-fold cv run.
        So here X_test and y_test correspond to the cv testing fold and X_train and y_train to the num folds for training 
    """

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
        
        total_num_graphs==0
        for X,y in loader:
            
            
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            out = model(X)
            target = y 

            # out.to('cpu')
            # target.to('cpu')
            loss = F.nll_loss(out, target)
            #print(loss)
            # out.to('device')
            # target.to('device')
            
            loss_train +=loss
    
            loss.backward()
            optimizer.step()
            total_num_graphs+=X.shape[0]

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
    
    error_measures = F1Score(y_test_pred.flatten().tolist(), 
                       y_test_gpu.flatten().tolist(),
                       nclasses)

    #test_loss = F.nll_loss(out, y_test_gpu)
    #pprint(test_loss.to('cpu').detach().numpy().item())
    #test_error = test_loss.to('cpu').detach().numpy().item()

    end = time.time()
    
    score = 'f1-micro'
    results_dict[model_name][score] = {}
    results_dict[model_name][score]['params'] = nn_model_params['model_kwargs']
    # since the classes are balanced, we can use macro average
    results_dict[model_name][score]['cv_fold_score'] = error_measures['macroF1']
    results_dict[model_name][score]['time'] = round(end-start)


    return results_dict, model


def train_nn_model_cv(X_train, y_train, X_test, y_test, nn_model_params, scores, nclasses, numfolds=3 ):

    """
        - for each param combination instanciate a model
            - for each cvfold do training
                - for each epoch do training
                    - for each batch do sgd
                - save cv_errors
            - average cv_error 
            - select best performing model
        - retrain best model on all X_train 
        - test model  


    """

   
    kfld = KFold(n_splits=numfolds)
    cv_error_score_history = []
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
        cv_fold_results_dict, _ =  train_nn_model_one_fold(X_train, y_train, X_test, y_test, nn_model_params, scores, nclasses )

        model_name = ''
        for k,v in cv_fold_results_dict.items():
            model_name=k

        score = ''
        for k,v in cv_fold_results_dict[model_name].items():
            score=k

        cv_error_score_history.append(cv_fold_results_dict[model_name][score]['cv_fold_score'])



    # average it's error measure (or take it already averaged)
    cv_avg_error_score = 0
    for e in cv_error_score_history:
        cv_avg_error_score+= e 
    cv_avg_error_score = cv_avg_error_score / numfolds 

    results_dict = cv_fold_results_dict 
    results_dict['cv_avg_error'] = cv_avg_error_score 

    return results_dict



def nn_train_models(X_train, y_train, X_test, y_test, nn_models_params, scores, nclasses, numfolds=3 ):
    """
    This function will take all model parameter sets, and create an iterator over all combinations.

    for each combination of parameter values and model class, it will launch a cross validation training. 

    Then later it will retrain the best performing model over the full training set

    Then it will test its prediction and write down the results

    """
    print("nn_train_models, nclasses=",nclasses)

    # get X number of columns
    n_X_cols = X_train.shape[1]

    nn_models_params_unrolled = []
    for model_class_key,v in nn_models_params.items():
        # exclude model_class
        # unfold loop for each parameter
        # for each k get all values
        # print("model", model_class_key)
        # pprint(v)

        all_combos = []
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

                all_combos.append(params_dict)

    nn_models_params_unrolled = all_combos

    pprint(nn_models_params_unrolled)
        
    
    error_scores = []
    for param_set in nn_models_params_unrolled:
    
        results_dict = train_nn_model_cv(X_train, y_train, X_test, y_test, param_set, scores, nclasses, numfolds=3 )
        error_scores.append(results_dict['cv_avg_error'])

    # choose best model 
    best_error = min(error_scores)
    best_error_index = error_scores.index(best_error)
    best_model_params = nn_models_params_unrolled[best_error_index]

    # now retrain and get the error measure
    results_dict, model = train_nn_model_one_fold(X_train, y_train, X_test, y_test, best_model_params, scores, nclasses )

    results_dict['model_instance'] = model

    # now test it
    results_dict = test_nn_model(X_test, y_test, results_dict['model_instance'],scores, nclasses, results_dict )

    return results_dict



def save_results(results_dict, features, dataset_version):
    # append results to results_dict on disk
    if dataset_version=='v1':
        r = json.load(open('training_results.json','r'))
    else:
        r = json.load(open('training_results_v2.json','r'))

    # how to merge? , just append? or merge for each model?
    for model_name in results_dict.keys():
        if model_name not in r.keys():
            r[model_name]={}

        for score,score_val in results_dict[model_name].items():
            score_val['features'] = features
            r[model_name][score + datetime.now().strftime("%Y-%m-%d_%H_%M_%S")] = score_val 
    

    
    if dataset_version=='v1':
        json.dump(r, open('training_results.json','w'))
    else:
        json.dump(r, open('training_results_v2.json','w'))
    #pprint(results_dict)
    #return results_dict


def print_training_stats(dataset_version='v1'):
    """
    read json from disk and print the best model of each model type.
    print also it's scores obviously
    """
    
    if dataset_version=='v1':
        r = json.load(open('training_results.json','r'))
    else:
        r = json.load(open('training_results_v2.json','r'))


    info_res = [ (model,score,score_res['cv_score'], score_res['params'], score_res['features']) for model,model_res in r.items() 
                   for score,score_res in model_res.items() ]

    # for each modell print the best result
    models = list(set([ a[0] for a in info_res]))

    best_models = []    
    for model in models:

        scores = [ a[2] for a in info_res if a[0]==model]
        params = [ a[3] for a in info_res if a[0]==model]
        score_names = [ a[1] for a in info_res if a[0]==model]
        features = [ a[4] for a in info_res if a[0]==model]
        max_score = max(scores)
        max_model = scores.index(max_score)
        max_model_params = params[max_model] # later this will be params
        score_name = score_names[max_model]
        best_models.append((model, score_name, max_score, max_model_params,features[max_model]))

    pprint(best_models)


def baseline_training_and_testing(X_train, X_test, y_train, y_test, features, dataset_version='v1',nclasses=3):
    
    """
        Filter features following the indication
    """
    X_train_filtered, X_test_filtered = filter_features(X_train, X_test, features)

    # models_and_params = prepare_models()
    # results_dict = cv_train_models(
    #     X_train_filtered, y_train, 
    #     X_test_filtered, y_test, 
    #     models_and_params, 
    #     scores=['recall_macro','recall_micro','precision_macro','precision_micro','f1_macro','f1_micro',])

    # save_results(results_dict, features,dataset_version)


    nn_models_and_params = prepare_nn_models()

    results_dict = nn_train_models(
        X_train, y_train, 
        X_test, y_test, 
        nn_models_and_params, 
        None, nclasses=nclasses, 
        numfolds=3
        )
    
    pprint(results_dict)
    #save_results(results_dict, features,dataset_version)




if __name__=='__main__':


    dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    dataset_version='v1'
    # print(len(dataset))
    # print(dataset.num_classes)
    # print(dataset.num_features)

    # same randomization for all
    X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset)

    baseline_training_and_testing(X_train, X_test, y_train, y_test,'x_topo_feats',dataset_version,nclasses)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'code_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'feats_topo_code',dataset_version)
    

    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    # dataset_version='v2'
    # # print(len(dataset))
    # # print(dataset.num_classes)
    # # print(dataset.num_features)

    # # same randomization for all
    # X_train, X_test, y_train, y_test = dataset_split_shared_splits(dataset)

    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'x_topo_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'code_feats',dataset_version)
    
    # baseline_training_and_testing(X_train, X_test, y_train, y_test,'feats_topo_code',dataset_version)
    


    #print_training_stats('v1')

    #print_training_stats('v2')