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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer



from datetime import datetime

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from TFM_function_renaming_dataset_creation import *
from TFM_function_renaming_dataset_creation import FunctionsDataset
from TFM_function_renaming_baseline_models import *
import scipy


from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

class FeatureCombinator(TransformerMixin):
    """
    used in a sklearn Pipeline for combining TFIDF and other features into one feature matrix
    """


    def __init__(self, X2, y):
        self.X2 = X2
        self.y = y

    def transform(self, X1, *_):
        # print("\n Calling transform!!!!")
        # from scipy csr sparse matrix to np.ndarray()
        if isinstance(X1,scipy.sparse.csr.csr_matrix):
            X1  = X1.toarray()

        elif isinstance(X1, list):
            X1 = np.array(X1)

        # usually X1 is list 
        #  X2 is np.array()
        # print("X1 is ", type(X1), X1.shape)
        # print("X2 is ", type(self.X2), self.X2.shape)
        

        # make sure they have the same num rows 

        # if X2 is None or empty return X1
        if self.X2 is None:
            #return np.array(X1), self.y
            return X1

        # X2 is empty
        if self.X2.shape[1]==0:
            #return np.array(X1), self.y
            # print("\n returning only X1 !")
            return X1

        
        #print(X1.shape[0] == self.X2.shape[0])
        if X1.shape[0] != self.X2.shape[0]:
            #return np.array(X1), self.y
            return X1

        # join then into one big matrix
        print("\n combining features X1 and X2 !")
        X_result = []
        for j in range(len(X1)):
            therow = []
            therow.extend(X1[j])
            therow.extend(self.X2[j])
            X_result.append(therow)
            #print(therow)

        # transform to np.ndarray()
        X_result = np.array(X_result)

        print("X1 is ", type(X1), X1.shape)
        print("X2 is ", type(self.X2), self.X2.shape)
        print("final shape", X_result.shape)
        return X_result


    def fit(self, X1, *_):
        return self
        

class FeatureVerify(TransformerMixin):
    """
    debuging class for sklearn Pipeline
    """


    def transform(self, X, y, *_):
        pprint(X)
        pprint(y)
        pprint(_)
        return X, y

    def fit(self, X, y, *_):
        pprint(X)
        pprint(y)
        pprint(_)
        return X, y
        

def prepare_nlp_models():
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
                 'max_iter': [100,200,], #'max_iter': [200],
                 #'C':[1,10,100,1000],
                 'C':[0.1,1,10],

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
                 'n_estimators': [4,16],
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



def cv_train_nlp_models(X_train_numeric, X_train_doc, 
            y_train, 
            X_test_numeric, X_test_doc, 
            y_test, 
            models_params, scores,
            features,
            dataset_version,
            results_folder,
            models_folder='models/nlp_models/'
             ):

    results_dict={}

    for model_name in models_params.keys():
        """
        for all model and theri possibble params, 
                    build a Pipeline(tfidf, featureComb, classifier)
                    run GridSearchCV to optimize tfidf params and classifier params

        Returns best cross-val model retrained and tested
        """

        print("Training ", model_name)
        model_dict = models_params[model_name]
        model = model_dict['model']()
        parameters = model_dict['params_set']
        results_dict[model_name]={}


        tf_params = {
             #'tvec__max_features':[100, 2000],
             'tvec__max_features':[100],
             #'tvec__ngram_range': [(1, 2), (2, 3), (3, 3)],
             'tvec__ngram_range': [(2, 3)],
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
              ('features',FeatureCombinator(X_train_numeric,y_train)),
              ('clf', model_dict['model']())
            ])

        """
        Approach: 
            1- train a CV gridsearch of Tfidfvectorizer and ml model 
            2- in between the feature combinator merges features from tfidf and numeric features like topo or code feats
            3- once fit , get the best params of model and tfidfvectorizer and build 2 new pipelines for transforming test and train(tfidf and featurecombiner), then fit an isolated(no pipeline) ml model -> IMPORTANT: the tfidf must be trained with X_train_doc for both training and testing piepeline, but the featurecombiner in each case must use X_train_numeric or X_test_numeric respectively
            4- finally with test transformed you can predict with the isolated model 
        """

        for score in scores:
            try:
                start = time.time()

                print("GridseachCV for ", score)
                clf = GridSearchCV(t_pipe, param_grid=final_parameters, cv=3, verbose=1, scoring=score, n_jobs=2, refit=False)
                clf.fit(X_train_doc, y_train)

                # print("Results of CV:")
                # pprint(clf.best_params_)

                # print("Training best model with separated pipeline")
                #print(dir(clf))
                tfidfvctr_params = {}
                for k,v in clf.best_params_.items():
                    jj =k.find('tvec__')
                    if jj==-1:
                        continue 
                    tfidfvctr_params[k[6:]]=v                
                pprint(tfidfvctr_params)

                model_params = {}
                for k,v in clf.best_params_.items():
                    jj =k.find('clf__')
                    if jj==-1:
                        continue 
                    model_params[k[5:]]=v                
                pprint(model_params)

                t_pipe_train_final = Pipeline([
                      ('tvec', TfidfVectorizer(**tfidfvctr_params)),
                      ('features',FeatureCombinator(X_train_numeric,y_train))
                    ])
                X_train = t_pipe_train_final.fit_transform(X_train_doc)
                clf2 = model_dict['model'](**model_params)
                
                clf2.fit(X_train, y_train)

                # print("Transfroming test")
                t_pipe_test = Pipeline([
                      ('tvec', TfidfVectorizer(**tfidfvctr_params)),
                      ('features',FeatureCombinator(X_test_numeric,y_test))
                    ])
                test_transformer = t_pipe_test.fit(X_train_doc)
                # ftrcomb = FeatureCombinator(X_test_numeric, y_test)
                # X_test = ftrcomb.fit_transform(X_test_doc)
                X_test = test_transformer.transform(X_test_doc)
                

                # print("predicting test")    
                y_true, y_pred = y_test, clf2.predict(X_test)

                end = time.time()

                
                results_dict[model_name][score] = classification_report(y_true, y_pred, output_dict=True)
                results_dict[model_name][score]['params'] = clf.best_params_
                results_dict[model_name][score]['cv_score'] = clf.best_score_
                results_dict[model_name][score]['time'] = round(end-start)
                results_dict[model_name][score]['model_instance'] = clf

                #results_dict[model_name]['best']=clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

        print("\n\n\nBefore save results, results_folder=",results_folder,"\n\n\n")
        save_results(results_dict, features,dataset_version, results_folder='nlp', models_folder=models_folder)

    return results_dict



def prepare_pipeline_params(parameters, tf_params):


    

    final_parameters = []
    for param_set in parameters:
        final_parameter = copy.deepcopy(tf_params)

        for k,v in param_set.items():
            final_parameter['clf__'+k]=v

        #pprint(final_parameter)
        #print()
        final_parameters.append(final_parameter)

    return final_parameters


class ToNumpyTransformer(TransformerMixin):
    def transform(self, X1, *_):
        # print("ToNumpyTransformer")
        # print("colums ")
        # print(X1.columns)
        # print(X1.shape)
        # print()
        return X1.to_numpy()

    def fit(self, X1, *_):
        return self


class DummyDebug(TransformerMixin):
    def transform(self, X1, *_):
        # print("DummyDebug")
        # print("shape ")
        # print(X1.shape)
        #print(X1[0:1])
        # print()
        return X1

    def fit(self, X1, *_):
        return self

class PreTfidf(TransformerMixin):
    def transform(self, X1, *_):
        # print("pre TFIDF")
        # print(X1.columns)
        # print()
        return X1['document'].array
        
    def fit(self, X1, *_):
        return self

def cv_train_nlp_models_v2(X_train_all , train_numeric_cols, train_nlp_cols, 
            y_train, 
            X_test_all, test_numeric_cols, test_nlp_cols, 
            y_test, 
            models_params, scores,
            features,
            dataset_version,
            results_folder='nlp',
            models_folder='models/nlp_models/'
             ): 
    """
        Improved generation of the pipeline,
        flexible enough to use with the training
    """

    results_dict={}

    # pprint(train_numeric_cols)
    # pprint(train_nlp_cols)
    # print()
    
    if features.find('tfidf')>-1:
        tfidf_transformer = Pipeline([
            ('debug', DummyDebug()),
        ])
        tf_params = {
        }
    else:
        tfidf_transformer = Pipeline([
            ('preTfIdf', PreTfidf()),
            ('tvec', TfidfVectorizer()),
            #('debug', DummyDebug()),
        ])
        tf_params = {
         'preprocessor__tfidf__tvec__max_features':[100, 200,500],
         #'preprocessor__tfidf__tvec__max_features':[100],
         'preprocessor__tfidf__tvec__ngram_range': [(1, 2), (2, 3), (3, 3)],
         #'preprocessor__tfidf__tvec__ngram_range': [(2, 3)],
         #'tvec__stop_words': [None, 'english'],
         'preprocessor__tfidf__tvec__max_df': [0.7,0.8,0.9],
         'preprocessor__tfidf__tvec__min_df': [0.1,0.2]
        }
        

    baseline_transformer = Pipeline(steps=[
         ('tonumpy', ToNumpyTransformer()), # probably it's no needed
         ('scaler', StandardScaler()),
         #('debug', DummyDebug()),
         ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('convfeats', baseline_transformer, train_numeric_cols),
            ('tfidf', tfidf_transformer, train_nlp_cols)]
    )

    for model_name in models_params.keys():
        """
        for all model and theri possibble params, 
                    build a Pipeline(tfidf, featureComb, classifier)
                    run GridSearchCV to optimize tfidf params and classifier params

        Returns best cross-val model retrained and tested
        """

        print("Training ", model_name)
        model_dict = models_params[model_name]
        model = model_dict['model']()
        parameters = model_dict['params_set']
        results_dict[model_name]={}

        final_parameters = prepare_pipeline_params(parameters, tf_params)
        
        train_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('clf', model_dict['model']())
                        ])


        for score in scores:
            try:
                start = time.time()

                print("GridseachCV for ", score)
                clf = GridSearchCV(train_pipeline,final_parameters,cv=5, scoring=score, refit=True)
                clf.fit(X_train_all,y_train )

                end = time.time()


                y_true, y_pred = y_test, clf.predict(X_test_all)
                

                results_dict[model_name][score] = classification_report(y_true, y_pred, output_dict=True)
                
                results_dict[model_name][score]['params'] = clf.best_params_
                results_dict[model_name][score]['cv_score'] = clf.best_score_
                results_dict[model_name][score]['time'] = round(end-start)
                results_dict[model_name][score]['model_instance'] = clf

                #results_dict[model_name]['best']=clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

        print("\n\n\nBefore save results, results_folder=",results_folder,"\n\n\n")
        
        pprint(results_dict[model_name])
        save_results(results_dict, features,dataset_version, results_folder=results_folder,models_folder=models_folder)

    return results_dict


def cv_train_nlp_models_v3_precomp_tfidf(X_train_all , train_numeric_cols, train_nlp_cols, 
            y_train, 
            X_test_all, test_numeric_cols, test_nlp_cols, 
            y_test, 
            models_params, scores,
            features,
            dataset_version,
            results_folder='nlp',
            models_folder='models/nlp_models/'
             ): 
    """
        Improved generation of the pipeline,
        flexible enough to use with the training
    """

    results_dict={}

    for model_name in models_params.keys():
        """
        for all model and theri possibble params, 
                    build a Pipeline(tfidf, featureComb, classifier)
                    run GridSearchCV to optimize tfidf params and classifier params

        Returns best cross-val model retrained and tested
        """

        print("Training ", model_name)
        model_dict = models_params[model_name]
        model = model_dict['model']()
        parameters = model_dict['params_set']
        results_dict[model_name]={}


        final_parameters = parameters

        pprint(train_numeric_cols)
        pprint(train_nlp_cols)
        print()
        
        baseline_transformer = Pipeline(steps=[
             ('tonumpy', ToNumpyTransformer()), # probably it's no needed
             ('scaler', StandardScaler()),
             #('debug', DummyDebug()),
             ])

        tfidf_transformer = Pipeline([
            ('debug', DummyDebug()),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('convfeats', baseline_transformer, train_numeric_cols),
                ('tfidf', tfidf_transformer, train_nlp_cols)]
        )

        train_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('clf', model_dict['model']())
                        ])

        for score in scores:
            try:
                start = time.time()

                print("GridseachCV for ", score)
                clf = GridSearchCV(train_pipeline,final_parameters,cv=5, scoring=score, refit=True)
                clf.fit(X_train_all,y_train )

                end = time.time()


                y_true, y_pred = y_test, clf.predict(X_test_all)
                

                results_dict[model_name][score] = classification_report(y_true, y_pred, output_dict=True)
                
                results_dict[model_name][score]['params'] = clf.best_params_
                results_dict[model_name][score]['cv_score'] = clf.best_score_
                results_dict[model_name][score]['time'] = round(end-start)
                results_dict[model_name][score]['model_instance'] = clf

                #results_dict[model_name]['best']=clf 
            except Exception as err:
                print("Error with "+model_name+" and "+score)
                traceback.print_exc()

        print("\n\n\nBefore save results, results_folder=",results_folder,"\n\n\n")
        
        pprint(results_dict[model_name])
        save_results(results_dict, features,dataset_version, results_folder=results_folder,models_folder=models_folder)

    return results_dict




def final_setup_params(param_set_original):
    """
        transform paraM-set original
        separating them into tvec__ prefix params, and clf__ prefix params

    """
    param_set = copy.deepcopy(param_set_original)
    tf_current_params = {}
    list_to_remove_keys = []
    for k,v in param_set.items():
        jj =k.find('tvec__')
        if jj==-1:
            continue 
        list_to_remove_keys.append(k)
        tf_current_params[k[6:]]=v  

    for k in list_to_remove_keys:
        param_set.pop(k)

    #correct format for later
    param_set2 = {
        'model_class': param_set.pop('model_class'),
        'num_epochs': param_set.pop('num_epochs'),
        'model_kwargs': param_set
    }
    param_set = param_set2

    return tf_current_params, param_set


def final_setup_params_v2(param_set_original):
    """
        transform paraM-set original
        separating them into tvec__ prefix params, and clf__ prefix params

        v2 uses preprocessor__tvec__ instead of removing it
    """
    param_set = copy.deepcopy(param_set_original)
    tf_current_params = {}
    list_to_remove_keys = []
    for k,v in param_set.items():
        jj =k.find('tvec__')
        if jj==-1:
            continue 
        list_to_remove_keys.append(k)
        tf_current_params[k]=v  

    for k in list_to_remove_keys:
        param_set.pop(k)

    #correct format for later
    param_set2 = {
        'model_class': param_set.pop('model_class'),
        'num_epochs': param_set.pop('num_epochs'),
        'model_kwargs': param_set
    }
    param_set = param_set2

    return tf_current_params, param_set




def cv_train_nn_nlp_models(X_train_numeric, X_train_doc,
                 y_train, 
                 X_test_numeric,  X_test_doc,
                 y_test, 
                 nn_models_params, scores, 
                 nclasses, numfolds=3, 
                 features='', dataset_version='', 
                 results_folder='baseline',
                 models_folder='models/baseline_models/' ):
    """
    This function will take all model parameter sets, and create an iterator over all combinations.

    for each combination of parameter values and model class, it will launch a cross validation training. 

    Then later it will retrain the best performing model over the full training set

    Then it will test its prediction and write down the results

    """
    print("\ncv_train_nn_nlp_models, nclasses=",nclasses)

    """
        Rules for combinint tfidf and other features :
        the document will always live in the first position of the X vector

        difficult to cope with X being an nd.array or not

        SOL:
            -> build different pieces
                - list of list for document and list of funcs 
                - nd.array for numeric values for nn
                - 
            -> choose them assemble them together in this function and similar functions
    """

    tfvec_params = {
             #'tvec__max_features':[100, 2000],
             'tvec__max_features':[100,110],
             #'tvec__ngram_range': [(1, 2), (2, 3), (3, 3)],
             'tvec__ngram_range': [(2, 3),(3, 3)],
             #'tvec__stop_words': [None, 'english'],
             'tvec__max_df': [0.8],
             'tvec__min_df': [0.1]
            }

    print("Before unrolling:")
    pprint(nn_models_params)

    # prepare all model candidates
    # nn_models_params_unrolled = unroll_all_possible_model_combos(X_train_embedding, nn_models_params, nclasses)
    nn_models_params_unrolled = unroll_all_possible_model_combos_with_tfidf( nn_models_params, nclasses,tfvec_params)

    pprint(nn_models_params_unrolled)

    # CV training for all candidates
    error_scores = []
    for param_set in nn_models_params_unrolled:
        try:
            # read tfvec params from param_set
            # select the ones with tvec
            tf_current_params, param_set = final_setup_params(param_set)
            
            print("After selecting tfidf params:")
            pprint(tf_current_params)  
            pprint(param_set)

            tfidfv = TfidfVectorizer(**tf_current_params)

            X_train_embedding = tfidfv.fit_transform(X_train_doc)
            X_test_embedding = tfidfv.transform(X_test_doc)

            X_train_embedding = X_train_embedding.toarray()
            X_test_embedding =  X_test_embedding.toarray()

            ftrcomb = FeatureCombinator(X_train_numeric, y_train)
            X_train_embedding = ftrcomb.transform(X_train_embedding, y_train)
            ftrcomb = FeatureCombinator(X_test_numeric, y_train)
            X_test_embedding = ftrcomb.transform(X_test_embedding, y_test)

            # add the last tweaks to the params_set (neural net input size cols)

            
            #size of input = num X columns 
            param_set['model_kwargs']['d1'] = X_train_embedding.shape[1]
            #print("set param_set['model_kwargs']['d1']=",X_train_embedding.shape[1] )


            # print("in cv_train_nn_nlp_models")
            # pprint(param_set)
    

            results_dict = train_nn_model_cv(X_train_embedding, y_train, X_test_embedding, y_test, param_set, scores, nclasses, numfolds=3 )
            error_scores.append(results_dict['cv_avg_error'])
        except Exception as err:
            print("Error with "+param_set+" and "+scores[0])
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
        tf_current_params, param_set = final_setup_params(best_model_params)
            

        tfidfv = TfidfVectorizer(**tf_current_params)

        X_train_embedding = tfidfv.fit_transform(X_train_doc)
        X_test_embedding = tfidfv.transform(X_test_doc)

        X_train_embedding = X_train_embedding.toarray()
        X_test_embedding =  X_test_embedding.toarray()

        ftrcomb = FeatureCombinator(X_train_numeric, y_train)
        X_train_embedding = ftrcomb.transform(X_train_embedding, y_train)
        ftrcomb = FeatureCombinator(X_test_numeric, y_train)
        X_test_embedding = ftrcomb.transform(X_test_embedding, y_test)

        # add the last tweaks to the params_set (neural net input size cols)

        
        #size of input = num X columns 
        param_set['model_kwargs']['d1'] = X_train_embedding.shape[1]
        
    

        results_dict, model = train_nn_model_one_fold(X_train_embedding, y_train, X_test_embedding, y_test, param_set, scores, nclasses )
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


def cv_train_nn_nlp_models_v2(X_train_all , train_numeric_cols,
                train_nlp_cols, 
                y_train, 
                X_test_all, test_numeric_cols, test_nlp_cols, 
                y_test, 
                nn_models_params, scores, 
                nclasses, numfolds=3, 
                features='', dataset_version='', 
                results_folder='baseline',
                models_folder='models/nlp_models/' ):
    """
    This function will take all model parameter sets, and create an iterator over all combinations.

    for each combination of parameter values and model class, it will launch a cross validation training. 

    Then later it will retrain the best performing model over the full training set

    Then it will test its prediction and write down the results

    """
    print("\ncv_train_nn_nlp_models, nclasses=",nclasses)

    """
        Rules for combinint tfidf and other features :
        the document will always live in the first position of the X vector

        difficult to cope with X being an nd.array or not

        SOL:
            -> build different pieces
                - list of list for document and list of funcs 
                - nd.array for numeric values for nn
                - 
            -> choose them assemble them together in this function and similar functions
    """



    print("Before unrolling:")
    pprint(nn_models_params)






    baseline_transformer = Pipeline(steps=[
             ('tonumpy', ToNumpyTransformer()), # probably it's no needed
             ('scaler', StandardScaler()),
             #('debug', DummyDebug()),
             ])

    if features.find('tfidf')>-1:
        tfidf_transformer = Pipeline([
            ('debug', DummyDebug()),
        ])
        # tfidf params must be empty
        tfvec_params = {}
    else:
        tfidf_transformer = Pipeline([
            ('preTfIdf', PreTfidf()),
            ('tvec', TfidfVectorizer()),
            #('debug', DummyDebug()),
        ])
        # tfidf params
        tfvec_params = {
         #'preprocessor__tfidf__tvec__max_features':[100, 2000],
         'preprocessor__tfidf__tvec__max_features':[100],
         #'preprocessor__tfidf__tvec__ngram_range': [(1, 2), (2, 3), (3, 3)],
         'preprocessor__tfidf__tvec__ngram_range': [(2, 3)],
         #'preprocessor__tfidf__tvec__stop_words': [None, 'english'],
         'preprocessor__tfidf__tvec__max_df': [0.8],
         'preprocessor__tfidf__tvec__min_df': [0.1]
        }

    preprocessor = ColumnTransformer(
        transformers=[
            ('convfeats', baseline_transformer, train_numeric_cols),
            ('tfidf', tfidf_transformer, train_nlp_cols)]
    )

    tfidf_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ])

    # prepare all model candidates
    # nn_models_params_unrolled = unroll_all_possible_model_combos(X_train_embedding, nn_models_params, nclasses)
    nn_models_params_unrolled = unroll_all_possible_model_combos_with_tfidf( nn_models_params, nclasses,tfvec_params)
    pprint(nn_models_params_unrolled)

    # CV training for all candidates
    error_scores = []
    for param_set in nn_models_params_unrolled:
        try:
            # read tfvec params from param_set
            # select the ones with tvec
            tf_current_params, param_set = final_setup_params_v2(param_set)
            
            print("After selecting tfidf params:")
            pprint(tf_current_params)  
            pprint(param_set)

            tfidf_pipeline.set_params(**tf_current_params)
            X_train_embedding = tfidf_pipeline.fit_transform(X_train_all)
            X_test_embedding = tfidf_pipeline.transform(X_test_all)

            # X_train_embedding = X_train_embedding.toarray()
            # X_test_embedding =  X_test_embedding.toarray()

            # ftrcomb = FeatureCombinator(X_train_numeric, y_train)
            # X_train_embedding = ftrcomb.transform(X_train_embedding, y_train)
            # ftrcomb = FeatureCombinator(X_test_numeric, y_train)
            # X_test_embedding = ftrcomb.transform(X_test_embedding, y_test)

            # add the last tweaks to the params_set (neural net input size cols)

            
            #size of input = num X columns 
            param_set['model_kwargs']['d1'] = X_train_embedding.shape[1]
            #print("set param_set['model_kwargs']['d1']=",X_train_embedding.shape[1] )


            # print("in cv_train_nn_nlp_models")
            # pprint(param_set)
    

            results_dict = train_nn_model_cv(X_train_embedding, y_train, X_test_embedding, y_test, param_set, scores, nclasses, numfolds=3 )
            error_scores.append(results_dict['cv_avg_error'])
        except Exception as err:
            print("Error with "+param_set+" and "+scores[0])
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
        tf_current_params, param_set = final_setup_params_v2(best_model_params)
            

        tfidf_pipeline.set_params(**tf_current_params)
        X_train_embedding = tfidf_pipeline.fit_transform(X_train_all)
        X_test_embedding = tfidf_pipeline.transform(X_test_all)

        # add the last tweaks to the params_set (neural net input size cols)

        
        #size of input = num X columns 
        param_set['model_kwargs']['d1'] = X_train_embedding.shape[1]

        if not isinstance(X_train_embedding,np.ndarray):
            X_train_embedding = X_train_embedding.toarray()
            X_test_embedding = X_test_embedding.toarray()

        results_dict, model = train_nn_model_one_fold(X_train_embedding, y_train, X_test_embedding, y_test, param_set, scores, nclasses, extra_params=tf_current_params )
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
    
    save_results(results_dict, features, dataset_version, results_folder=results_folder,models_folder=models_folder)

    
    return results_dict




# def save_results(results_dict, features, dataset_version, min_count):
#     # append results to results_dict on disk
#     if dataset_version=='v1':
#         r = json.load(open('nlp_training_results.json','r'))
#     else:
#         r = json.load(open('nlp_training_results_v2.json','r'))

#     # how to merge? , just append? or merge for each model?
#     for model_name in results_dict.keys():
#         if model_name not in r.keys():
#             r[model_name]={}

#         for score,score_val in results_dict[model_name].items():
#             score_val['features'] = features
#             score_val['min_count'] = min_count
#             r[model_name][score + datetime.now().strftime("%Y-%m-%d_%H_%M_%S")] = score_val 
    

    
#     if dataset_version=='v1':
#         json.dump(r, open('nlp_training_results.json','w'))
#     else:
#         json.dump(r, open('nlp_training_results_v2.json','w'))
#     #pprint(results_dict)
#     #return results_dict





def nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test, features, dataset_version='v1',nclasses=3, results_folder='nlp', models_folder='models/nlp_models/'):
    
    """
        Filter features following the indication

        Not used?
    """ 

    # X_train_numeric, X_train_doc = filter_features_new(X_train, features)
    # X_test_filtered, X_test_doc = filter_features_new(X_test, features)
    
    # nn_models_and_params = prepare_nn_models()


    # results_dict = cv_train_nn_nlp_models(
    #     X_train_numeric, X_train_doc, y_train, 
    #     X_test_filtered, X_test_doc, y_test,  
    #     nn_models_and_params, 
    #     scores=['f1_micro'], nclasses=nclasses, 
    #     numfolds=3,
    #     features=features,
    #     dataset_version=dataset_version,
    #     results_folder='nlp',
    #     models_folder='models/nlp_models'
    #     )


    # Numeric is a np.array, doc is a list of strings
    X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v2(X_train, features)
    X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v2(X_test, features)

    nn_models_and_params = prepare_nn_models()

    results_dict = cv_train_nn_nlp_models_v2(
        X_train_all, train_numeric_cols, train_nlp_cols, y_train, 
        X_test_all, test_numeric_cols, test_nlp_cols, y_test,  
        nn_models_and_params, 
        scores=['f1_micro'], nclasses=nclasses, 
        numfolds=3,
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder
        )








def nlp_models_training_and_testing(
            X_train, X_test, 
            y_train, y_test, 
            features, dataset_version='v1',
            nclasses=3, results_folder='nlp', 
            nlp_models=None, nn_models = None,
            models_folder='models/nlp_models/'):
    
    """
        Filter features following the indication
    """
    
    # Numeric is a np.array, doc is a list of strings
    X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v2(X_train, features)
    X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v2(X_test, features)


    if nn_models is None:
        nn_models_and_params = prepare_nn_models()
    else:
        nn_models_and_params = nn_models

    results_dict = cv_train_nn_nlp_models_v2(
        X_train_all, train_numeric_cols, train_nlp_cols, y_train, 
        X_test_all, test_numeric_cols, test_nlp_cols, y_test,  
        nn_models_and_params, 
        scores=['f1_micro'], nclasses=nclasses, 
        numfolds=3,
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder
        )




    if nlp_models is None:
        models_and_params = prepare_nlp_models()
    else:
        models_and_params = nlp_models 

    results_dict = cv_train_nlp_models_v2(
        X_train_all, train_numeric_cols, train_nlp_cols, y_train, 
        X_test_all, test_numeric_cols, test_nlp_cols, y_test, 
        models_and_params, 
        #scores=['recall_macro','recall_micro','precision_macro','precision_micro','f1_macro','f1_micro',],
        scores=['f1_micro'],
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder
        )



def nlp_models_training_and_testing_precomp_tfidf(
            X_train, X_test,
            X_train_tfidf, X_test_tfidf, 
            y_train, y_test, 
            features, dataset_version='v1',
            nclasses=3, results_folder='nlp', 
            nlp_models=None, nn_models = None,
            models_folder='models/nlp_models/'):
    
    """
        Filter features following the indication
    """
    
    # Numeric is a np.array, doc is a list of strings
    X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v3(X_train, X_train_tfidf, features)
    X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v3(X_test, X_test_tfidf, features)


    if nn_models is None:
        nn_models_and_params = prepare_nn_models()
    else:
        nn_models_and_params = nn_models

    results_dict = cv_train_nn_nlp_models_v2(
        X_train_all, train_numeric_cols, train_nlp_cols, y_train, 
        X_test_all, test_numeric_cols, test_nlp_cols, y_test,  
        nn_models_and_params, 
        scores=['f1_micro'], nclasses=nclasses, 
        numfolds=3,
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder
        )




    if nlp_models is None:
        models_and_params = prepare_nlp_models()
    else:
        models_and_params = nlp_models 

    results_dict = cv_train_nlp_models_v2(
        X_train_all, train_numeric_cols, train_nlp_cols, y_train, 
        X_test_all, test_numeric_cols, test_nlp_cols, y_test, 
        models_and_params, 
        #scores=['recall_macro','recall_micro','precision_macro','precision_micro','f1_macro','f1_micro',],
        scores=['f1_micro'],
        features=features,
        dataset_version=dataset_version,
        results_folder=results_folder,
        models_folder=models_folder
        )


def add_tfidf_to_dataset(dataset_folder):
    """
    load tfidf form folder
    then load dataset
    loop dataset and add the corresponding tfidf row to data.tfidf_vec
    then dataset.save_changes(idx,data)

    """
    X_train_tfidf = pickle.load(open(os.path.join(dataset_folder,'X_train_tfidf_document.pickle'),'rb'))
    X_test_tfidf = pickle.load(open(os.path.join(dataset_folder,'X_test_tfidf_document.pickle'),'rb'))
    # y_train = pickle.load(open(os.path.join(dataset_folder,'y_train.pickle'),'rb'))
    # y_test = pickle.load(open(os.path.join(dataset_folder,'y_test.pickle'),'rb'))
    # nclasses = pickle.load(open(os.path.join(dataset_folder,'nclasses.pickle'),'rb'))


    train_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'training_set'))

    for j in range(len(train_dataset)):
        tf_vec = X_train_tfidf[j,:]
        data = train_dataset[j]
        data.tfidf_vec = torch.FloatTensor(tf_vec)
        train_dataset.save_changes(j,data)

    test_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'test_set'))

    for j in range(len(test_dataset)):
        tf_vec = X_test_tfidf[j,:]
        data = test_dataset[j]
        data.tfidf_vec = torch.FloatTensor(tf_vec)
        test_dataset.save_changes(j,data)


def add_tfidf_to_dataset_with_a_trick(dataset_folder):
    """
    load tfidf form folder
    then load dataset
    loop dataset and add the corresponding tfidf row to data.tfidf_vec
    then dataset.save_changes(idx,data)

    instead of using data.tfidf_vec
    we will use data.edge_attr, so the problem with gnn_mode_on() is not present,
    but we can only use this in algorithms that do not make use of edge_attr

    """
    X_train_tfidf = pickle.load(open(os.path.join(dataset_folder,'X_train_tfidf_document.pickle'),'rb'))
    X_test_tfidf = pickle.load(open(os.path.join(dataset_folder,'X_test_tfidf_document.pickle'),'rb'))
    # y_train = pickle.load(open(os.path.join(dataset_folder,'y_train.pickle'),'rb'))
    # y_test = pickle.load(open(os.path.join(dataset_folder,'y_test.pickle'),'rb'))
    # nclasses = pickle.load(open(os.path.join(dataset_folder,'nclasses.pickle'),'rb'))


    train_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'training_set'))

    for j in range(len(train_dataset)):
        tf_vec = X_train_tfidf[j,:]
        data = train_dataset[j]
        data.edge_attr = torch.FloatTensor(tf_vec)
        train_dataset.save_changes(j,data)

    test_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'test_set'))

    for j in range(len(test_dataset)):
        tf_vec = X_test_tfidf[j,:]
        data = test_dataset[j]
        data.edge_attr = torch.FloatTensor(tf_vec)
        test_dataset.save_changes(j,data)



def verify_tfidf_on_dataset(dataset_folder):
    train_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'training_set'))

    for j in range(len(train_dataset)):
        data = train_dataset[j]
        pprint(data.tfidf_vec)



def run_ggnn_with_tfidf():
    """
    run modelSelection for a mxed ggn and tfidf model
    """
    pass

 

def precompute_tfidf(dataset_folder,features = 'document'):
    """
        get folder form where to precompute tfidf bow feature matrix

        Load pickles from disk.

        build, fit & transform tfidf pipeline
            call with:
            - features = 'document'
            - features = 'document simplified'
            - features = 'list_funcs'- document
            
        pickle result to disk 

    """
    print("precomputing tfidf for:\n    "+dataset_folder+"\nfeatures:\n    "+features)
    X_train = pickle.load(open(os.path.join(dataset_folder,'X_train.pickle'),'rb'))
    X_test = pickle.load(open(os.path.join(dataset_folder,'X_test.pickle'),'rb'))
    y_train = pickle.load(open(os.path.join(dataset_folder,'y_train.pickle'),'rb'))
    y_test = pickle.load(open(os.path.join(dataset_folder,'y_test.pickle'),'rb'))
    nclasses = pickle.load(open(os.path.join(dataset_folder,'nclasses.pickle'),'rb'))

    print(" ..read pickle files..")
    
    X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v2(X_train, features)
    X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v2(X_test, features)

    tfidf_parameters =  {
         'tfidf__tvec__max_features':500,
         'tfidf__tvec__ngram_range': (1, 2),
         'tfidf__tvec__max_df': 0.7,
         'tfidf__tvec__min_df': 0.2
        }

    tfidf_transformer = Pipeline([
            ('preTfIdf', PreTfidf()),
            ('tvec', TfidfVectorizer()),
        ])
    preprocessor = ColumnTransformer(
            transformers=[
                ('tfidf', tfidf_transformer, train_nlp_cols)]
        )
    preprocessor = preprocessor.set_params(**tfidf_parameters)
    print(" ..instantiated pipeline with parameters..")

    X_train_tfidf_embedding = preprocessor.fit_transform(X_train_all)
    print(" ..fit_transform on X_train performed..")
    X_test_tfidf_embedding = preprocessor.transform(X_test_all)
    print(" ..transform on X_test performed..")

    pickle.dump(X_train_tfidf_embedding,open(os.path.join(dataset_folder,'X_train_tfidf_'+features.replace(' ','_')+'.pickle'),'wb+'))
    pickle.dump(X_test_tfidf_embedding,open(os.path.join(dataset_folder,'X_test_tfidf_'+features.replace(' ','_')+'.pickle'),'wb+'))
    print(" ..written pickle files..")

    #verification
    embedding = pickle.load(open(os.path.join(dataset_folder,'X_train_tfidf_'+features.replace(' ','_')+'.pickle'),'rb'))
    print(embedding.shape)
    #pprint(embedding)
    
    embedding = pickle.load(open(os.path.join(dataset_folder,'X_test_tfidf_'+features.replace(' ','_')+'.pickle'),'rb'))
    print(embedding.shape)
    #pprint(embedding)
    print(" ..verification done..\n\n")



def precompute_tfidfs():
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_unchanged',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_unchanged',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_unchanged',
        features = 'document_simplified and list funcs')

    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_remove_min',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_remove_min',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_remove_min',
        features = 'document_simplified and list funcs')


    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_undersample_max',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_undersample_max',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_3_precomp_split_undersample_max',
        features = 'document_simplified and list funcs')


def precompute_tfidfs_v1():
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_unchanged',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_unchanged',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_unchanged',
        features = 'document_simplified and list funcs')

    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_remove_min',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_remove_min',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_remove_min',
        features = 'document_simplified and list funcs')


    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_undersample_max',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_undersample_max',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_1_precomp_split_undersample_max',
        features = 'document_simplified and list funcs')




def precompute_tfidfs_v2():
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_unchanged',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_unchanged',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_unchanged',
        features = 'document_simplified and list funcs')

    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_remove_min',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_remove_min',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_remove_min',
        features = 'document_simplified and list funcs')


    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_undersample_max',
        features = 'document')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_undersample_max',
        features = 'document_simplified')
    precompute_tfidf(
        dataset_folder='tmp/symbols_dataset_2_precomp_split_undersample_max',
        features = 'document_simplified and list funcs')



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

    #-----------------tfidf-precomputation----------------

    precompute_tfidfs()
    exit()

    #-----------------tfidf-params-optimization------------------

    

    X_train = pickle.load(open('tmp/symbols_dataset_3_precomp_split_unchangedX_train.pickle','rb'))
    X_test = pickle.load(open('tmp/symbols_dataset_3_precomp_split_unchanged/X_test.pickle','rb'))
    y_train = pickle.load(open('tmp/symbols_dataset_3_precomp_split_unchanged/y_train.pickle','rb'))
    y_test = pickle.load(open('tmp/symbols_dataset_3_precomp_split_unchangedy_test.pickle','rb'))
    nclasses = pickle.load(open('tmp/symbols_dataset_3_precomp_split_unchangednclasses.pickle','rb'))

    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_3')
    # dataset_version='v3'
    # features = 'document'
    # min_count=0
    # X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)

    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='results/tfidf_params_hp_search.json')
    print_training_stats('v3',results_folder='results/tfidf_params_hp_search_v3.json')

    features = 'document and topo feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='results/tfidf_params_hp_search_v3.json')

    exit()




    min_count=0
    dataset_version='v1'
    features = 'document'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='tfidf_params_hp_search.json')
    print_training_stats('v1',results_folder='tfidf_params_hp_search.json')

    features = 'document and topo feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='tfidf_params_hp_search.json')    
    print_training_stats('v1',results_folder='tfidf_params_hp_search.json')




    dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    dataset_version='v2'
    features = 'document'
    min_count=0
    X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)

    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='tfidf_params_hp_search.json')
    print_training_stats('v2',results_folder='tfidf_params_hp_search.json')

    features = 'document and topo feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count,results_folder='tfidf_params_hp_search.json')
    




    print_training_stats('v2',results_folder='tfidf_params_hp_search.json')

    
    #-------------------------------------------------------------


    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    # dataset_version='v1'
    # features = 'document'
    min_count=0
    # X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)


    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_1')
    dataset_version='v1'
    # # print(len(dataset))
    # # print(dataset.num_classes)
    # # print(dataset.num_features)

    # # same randomization for all
    # X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features='all', min_count=0)

    # pickle.dump(X_train,open('X_train.pickle','wb'))
    # pickle.dump(X_test,open('X_test.pickle','wb'))
    # pickle.dump(y_train,open('y_train.pickle','wb'))
    # pickle.dump(y_test,open('y_test.pickle','wb'))
    # pickle.dump(nclasses,open('nclasses.pickle','wb'))

    X_train = pickle.load(open('X_train.pickle','rb'))
    X_test = pickle.load(open('X_test.pickle','rb'))
    y_train = pickle.load(open('y_train.pickle','rb'))
    y_test = pickle.load(open('y_test.pickle','rb'))
    nclasses = pickle.load(open('nclasses.pickle','rb'))

    features = 'document'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v1',results_folder='nlp')

    # nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,features, dataset_version,nclasses)
    # print_training_stats('v1',results_folder='nlp')

    nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,features, dataset_version,nclasses,results_folder='nlp')
    print_training_stats('v1',results_folder='nlp')
    




    # features = 'document_simplified'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')

    # features = 'document and list funcs'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)
    # nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')

    # features = 'document_simplified and list funcs'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')


    features = 'document and topo feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v1',results_folder='nlp')

    #nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,'document and topo feats',dataset_version,nclasses)
    # print_training_stats('v1',results_folder='nlp')


    # features = 'document_simplified and topo feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')

    # features = 'document_simplified and list_funcs and topo feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')


    features = 'document and code feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v1',results_folder='nlp')

    #nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,features, dataset_version,nclasses)
    # print_training_stats('v1',results_folder='nlp')



    # features = 'document_simplified and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')

    # features = 'document_simplified and list_funcs and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1',results_folder='nlp')


    features = 'document and topo and code feats'
    nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    print_training_stats('v1',results_folder='nlp')

    #nlp_nn_models_training_and_testing(X_train, X_test, y_train, y_test,features, dataset_version,nclasses)
    # print_training_stats('v1',results_folder='nlp')


    # features = 'document_simplified and topo and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    # features = 'document_simplified and list_funcs and topo and code feats'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v1')

    



    # dataset = FunctionsDataset(root='./tmp/symbols_dataset_2')
    # dataset_version='v2'
    # features = 'document'
    # min_count=30
    # X_train, X_test, y_train, y_test, nclasses = dataset_split_shared_splits(dataset, features= features, min_count=min_count)


    # features = 'document'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v2')

    # features = 'document_simplified'
    # nlp_models_training_and_testing(X_train, X_test, y_train, y_test,features,dataset_version,min_count)    
    # print_training_stats('v2')

   

   