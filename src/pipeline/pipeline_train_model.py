import argparse
import yaml
from pprint import pprint
import logging
import os

from pathlib import Path
import sys
import inspect
import pkgutil
from importlib import import_module



parentdir = os.path.join(os.path.abspath('..'))
sys.path.insert(0,os.path.join(parentdir,'function_renaming'))
sys.path.insert(0,os.path.join(parentdir,'graph_classification'))

#sys.path.insert(0,'..')
#sys.path.insert(0,'../..')
#sys.path.insert(0,os.path.join(parentdir,'graph_classification'))
from TFM_function_renaming_baseline_models import *
from TFM_function_renaming_nlp_models import *
import TFM_graph_classification_models 
from TFM_graph_classification_models import *
import TFM_graph_classification 
from TFM_graph_classification import *



# logger setup
logger = logging.getLogger('training_jobs')
logger.setLevel(logging.DEBUG)
log_file= 'trainings.log'
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def add_dataset_file(results_file, dataset_folder,thetime):
    with open(results_file,'r+') as resf:
        resdict = json.load(resf)
        for k,v in resdict.items():
            for k2,v2 in v.items():
                v2['dataset']=dataset_folder
                v2['time']=thetime
        resf.seek(0)
        json.dump(resdict, resf)
        resf.truncate()



def training_dispatcher(jobdict):
    """
        Reads the dataset folder and model class name,
        and decides:
            1- which dataset files to read/import 
            2- which training method to call
        Finally calls training method:
            -> sets the results file to write to
            -> sets the folder to save the models to
    """


    # load configuration file
    conf =  yaml.load(open('training_conf.yml','r'), Loader=yaml.FullLoader)

    # change to function_renaming folder
    os.chdir('../function_renaming')

    

    # logger.debug('test dynamic import')
    # #imported_module = import_module('.' + '', package=__name__)
    # GGNN1_class = getattr(TFM_graph_classification_models, 'GGNN1')
    # logger.debug('imported class '+GGNN1_class.__name__)

    # GGNN_training_func = getattr(TFM_graph_classification, 'modelSelection')
    # logger.debug('imported func '+GGNN_training_func.__name__)


    # read dataset 
    dataset_folder = jobdict['dataset']

    # read feature type
    features = ''
    if 'features' in jobdict.keys():
        features = jobdict['features']

    # read model class name 
    model_class = jobdict['model']


    logger.debug("training with: ")
    logger.debug(dataset_folder)
    logger.debug(model_class)
    logger.debug(features)

    # decide training type and load dataset, model and training func
    if model_class in conf['training_types']['baseline']['models'] \
       and features in conf['training_types']['baseline']['features']:

        logger.debug("baseline training")
        # preapre load model 
        model_module = conf['training_types']['baseline']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['baseline']['training_module']
        training_module = import_module(training_module)
        training_func = conf['training_types']['baseline']['training_func']
        
        # load dataset
        X_train = pickle.load(open(os.path.join(dataset_folder
        ,'X_train.pickle'),'rb'))
        X_test = pickle.load(open(os.path.join(dataset_folder
        ,'X_test.pickle'),'rb'))
        y_train = pickle.load(open(os.path.join(dataset_folder
        ,'y_train.pickle'),'rb'))
        y_test = pickle.load(open(os.path.join(dataset_folder
        ,'y_test.pickle'),'rb'))
        nclasses = pickle.load(open(os.path.join(dataset_folder
        ,'nclasses.pickle'),'rb'))

        X_train_numeric, X_train_doc = filter_features_new(X_train, features)
        X_test_numeric, X_test_doc = filter_features_new(X_test, features)


        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
        for k,v in params_dict.items():
            params_dict[k]=[v]
        training_params = {
            themodel_class.__name__:{
            'model': themodel_class,
            'params_set':[
                params_dict
            ]
            }
        }


        # prepare training func kwargs
        results_file = 'baseline_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        kwargs = {
            'X_train': X_train_numeric, 
            'y_train': y_train, 
            'X_test': X_test_numeric, 
            'y_test': y_test, 
            'models_params': training_params, 
            'scores': ['f1_macro',],
            'features': features,
            'dataset_version': '',
            'results_folder': results_file, #folder and file
            'models_folder': 'models/baseline_models/'
        }


        # performing training(and testing)
        start = time.time()

        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open('../pipeline/'+log_file,'a'))
        results_dict = training_func(**kwargs)

        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)

        add_dataset_file(results_file, dataset_folder, str(round(end-start)))


    elif model_class in conf['training_types']['baseline_nn']['models'] \
       and features in conf['training_types']['baseline_nn']['features']:

        logger.debug("baseline_nn training")
        # preapre load model 
        model_module = conf['training_types']['baseline_nn']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['baseline_nn']['training_module']
        training_module = import_module(training_module)
        training_func = conf['training_types']['baseline_nn']['training_func']
        
        # load dataset
        X_train = pickle.load(open(os.path.join(dataset_folder
        ,'X_train.pickle'),'rb'))
        X_test = pickle.load(open(os.path.join(dataset_folder
        ,'X_test.pickle'),'rb'))
        y_train = pickle.load(open(os.path.join(dataset_folder
        ,'y_train.pickle'),'rb'))
        y_test = pickle.load(open(os.path.join(dataset_folder
        ,'y_test.pickle'),'rb'))
        nclasses = pickle.load(open(os.path.join(dataset_folder
        ,'nclasses.pickle'),'rb'))

        X_train_numeric, X_train_doc = filter_features_new(X_train, features)
        X_test_numeric, X_test_doc = filter_features_new(X_test, features)


        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
        for k,v in params_dict.items():
            params_dict[k]=[v]
        training_params = {
            themodel_class.__name__:{
            'model': themodel_class,
            'params_set':[
                params_dict
            ]
            }
        }


        # prepare training func kwargs
        results_file = 'baseline_nn_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        kwargs = {
            'X_train': X_train_numeric, 
            'y_train': y_train, 
            'X_test': X_test_numeric, 
            'y_test': y_test, 
            'nn_models_params': training_params, 
            'scores': ['f1_macro',],
            'nclasses': nclasses,
            'numfolds': 3,
            'features': features,
            'dataset_version': '',
            'results_folder': results_file, #folder and file
            'models_folder': 'models/baseline_models/'
        }


        # performing training(and testing)
        start = time.time()

        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open(log_file,'a'))
        results_dict = training_func(**kwargs)

        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)

        add_dataset_file(results_file, dataset_folder, str(round(end-start)))


    elif model_class in conf['training_types']['nlp']['models'] \
       and features in conf['training_types']['nlp']['features']:

        logger.debug("nlp training")
        # preapre load model 
        model_module = conf['training_types']['nlp']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['nlp']['training_module']
        training_module = import_module(training_module)
        training_func = conf['training_types']['nlp']['training_func']
        
        # load dataset
        X_train = pickle.load(open(os.path.join(dataset_folder
        ,'X_train.pickle'),'rb'))
        X_test = pickle.load(open(os.path.join(dataset_folder
        ,'X_test.pickle'),'rb'))
        y_train = pickle.load(open(os.path.join(dataset_folder
        ,'y_train.pickle'),'rb'))
        y_test = pickle.load(open(os.path.join(dataset_folder
        ,'y_test.pickle'),'rb'))
        nclasses = pickle.load(open(os.path.join(dataset_folder
        ,'nclasses.pickle'),'rb'))
        X_train_tfidf = pickle.load(open(os.path.join(dataset_folder
        ,'X_train_tfidf_document.pickle'),'rb'))
        X_test_tfidf = pickle.load(open(os.path.join(dataset_folder
        ,'X_test_tfidf_document.pickle'),'rb'))


        X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v3(X_train, X_train_tfidf, features)
        X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v3(X_test, X_test_tfidf, features)



        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
        for k,v in params_dict.items():
            params_dict[k]=[v]
        training_params = {
            themodel_class.__name__:{
            'model': themodel_class,
            'params_set':[
                params_dict
            ]
            }
        }


        # prepare training func kwargs
        results_file = 'nlp_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        kwargs = {
            'X_train_all': X_train_all, 
            'train_numeric_cols': train_numeric_cols,
            'train_nlp_cols': train_nlp_cols,
            'y_train': y_train, 
            'X_test_all': X_test_all, 
            'test_numeric_cols': test_numeric_cols,
            'test_nlp_cols': test_nlp_cols,
            'y_test': y_test, 
            'models_params': training_params, 
            'scores': ['f1_macro',],
            'features': features,
            'dataset_version': '',
            'results_folder': results_file, #folder and file
            'models_folder': 'models/nlp_models/'
        }


        # performing training(and testing)
        start = time.time()

        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open(log_file,'a'))
        results_dict = training_func(**kwargs)

        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)

        add_dataset_file(results_file, dataset_folder, str(round(end-start)))


    elif model_class in conf['training_types']['nlp_nn']['models'] \
       and features in conf['training_types']['nlp_nn']['features']:

        logger.debug("nlp_nn training")
        # preapre load model 
        model_module = conf['training_types']['nlp_nn']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['nlp_nn']['training_module']
        training_module = import_module(training_module)
        training_func = conf['training_types']['nlp_nn']['training_func']
        
        # load dataset
        X_train = pickle.load(open(os.path.join(dataset_folder
        ,'X_train.pickle'),'rb'))
        X_test = pickle.load(open(os.path.join(dataset_folder
        ,'X_test.pickle'),'rb'))
        y_train = pickle.load(open(os.path.join(dataset_folder
        ,'y_train.pickle'),'rb'))
        y_test = pickle.load(open(os.path.join(dataset_folder
        ,'y_test.pickle'),'rb'))
        nclasses = pickle.load(open(os.path.join(dataset_folder
        ,'nclasses.pickle'),'rb'))
        X_train_tfidf = pickle.load(open(os.path.join(dataset_folder
        ,'X_train_tfidf_document.pickle'),'rb'))
        X_test_tfidf = pickle.load(open(os.path.join(dataset_folder
        ,'X_test_tfidf_document.pickle'),'rb'))


        X_train_all, train_numeric_cols, train_nlp_cols = filter_features_new_v3(X_train, X_train_tfidf, features)
        X_test_all, test_numeric_cols, test_nlp_cols = filter_features_new_v3(X_test, X_test_tfidf, features)



        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
        params_dict['num_classes']=nclasses

        for k,v in params_dict.items():
            params_dict[k]=[v]

        training_params = {
            themodel_class.__name__:{
            'model': themodel_class,
            'params_set':[
                params_dict
            ]
            }
        }


        # prepare training func kwargs
        results_file = 'nlp_nn_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        kwargs = {
            'X_train_all': X_train_all, 
            'train_numeric_cols': train_numeric_cols,
            'train_nlp_cols': train_nlp_cols,
            'y_train': y_train, 
            'X_test_all': X_test_all, 
            'test_numeric_cols': test_numeric_cols,
            'test_nlp_cols': test_nlp_cols,
            'y_test': y_test, 
            'nn_models_params': training_params, 
            'scores': ['f1_macro',],
            'nclasses': nclasses,
            'numfolds': 3,
            'features': features,
            'dataset_version': '',
            'results_folder': results_file, #folder and file
            'models_folder': 'models/nlp_models/'
        }


        # performing training(and testing)
        start=time.time()

        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open(log_file,'a'))
        results_dict = training_func(**kwargs)

        
        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)
        add_dataset_file(results_file, dataset_folder, str(round(end-start)))



    elif model_class in conf['training_types']['ggnn_nlp']['models'] \
       and features in conf['training_types']['ggnn_nlp']['features']:

        logger.debug("ggnn_nlp training")
        # preapre load model 
        model_module = conf['training_types']['ggnn_nlp']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['ggnn_nlp']['training_module']
        training_module = import_module(training_module)
        training_func = conf['training_types']['ggnn_nlp']['training_func']
        
        # load tfidf into dataset
        add_tfidf_to_dataset(dataset_folder)

        # load dataset
        train_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'training_set'))
        train_dataset.gnn_mode_on()
        num_classes = train_dataset.num_classes
        test_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'test_set'))
        test_dataset.gnn_mode_on()

        X_train_tfidf = pickle.load(open(os.path.join(os.path.join(dataset_folder,'X_train_tfidf_document.pickle') ),'rb'))


        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
               
        epochs = params_dict.pop('epochs')
        learning_rate = params_dict.pop('learning_rate')
        weight_decay = params_dict.pop('weight_decay')
        batch_size = params_dict.pop('batch_size')
        params_dict['num_classes']=num_classes
        model_kwargs = params_dict
        params_dict2 = {
            'kwargs': model_kwargs,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'model': themodel_class,
            'weight_decay': float(weight_decay)
        }

        training_params = [params_dict2]

        # update any param with value -1 with X-train_tfidf.shape[1]
        for k,v in params_dict2['kwargs'].items():
            if v == -1:
                params_dict2['kwargs'][k]=X_train_tfidf.shape[1]


        # prepare training func kwargs
        results_file = 'ggnn_nlp_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        
        kwargs = {
            'model_list': training_params, 
            'k': 3,
            'train_dataset': train_dataset , 
            'balanced': False, 
            'force_numclasses': None, 
            'unbalanced_split': False, 
            'debug_training': True,
            'tfidf_indices': None
            #'results_folder': results_file, #folder and file
            #'models_folder': 'models/gnn/'
        }


        # performing training(and testing)
        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open(log_file,'a'))
        start = time.time()
        results_dict2 = training_func(**kwargs)


        results_dict = {'best_models_list':[], 'models':{} , 'best_models':{}, 'autoincrement': 0 }
        reportModelSelectionResult(results_dict2,results_dict)
        logger.debug("test_multiple_models")
        test_multiple_models(
            results_dict,
            test_dataset,
            results_file=results_file,
            models_folder='models/gnn_nlp/')
        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)

        add_dataset_file(results_file, dataset_folder, str(round(end-start)))

    #elif model_class in conf['training_types']['ggnn']['models'] \
    #   and features in conf['training_types']['ggnn']['features']:
    else:
        logger.debug("ggnn training")
        # preapre load model 
        model_module = conf['training_types']['ggnn']['model_module']
        model_module = import_module(model_module)
        themodel_class = getattr(model_module, model_class)
        

        # prepare load training func 
        training_module = conf['training_types']['ggnn']['training_module']
        training_module = import_module(training_module)

        training_func = conf['training_types']['ggnn']['training_func']
        

        # load dataset
        train_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'training_set'))
        #train_dataset = add_node_degree_v2(os.path.join(dataset_folder,'training_set'))
        train_dataset.gnn_mode_on()
        num_classes = train_dataset.num_classes
        test_dataset = FunctionsDataset(root=os.path.join(dataset_folder,'test_set'))
        #test_dataset = add_node_degree_v2(os.path.join(dataset_folder,'test_set'))
        test_dataset.gnn_mode_on()



        # prepare training params dict (jobdict + model )
        params_dict = copy.deepcopy(jobdict)
        params_dict.pop('features')
        params_dict.pop('dataset')
        params_dict.pop('model')
        
        params_dict['num_classes']=num_classes
        epochs = params_dict.pop('epochs')
        learning_rate = params_dict.pop('learning_rate')
        weight_decay = params_dict.pop('weight_decay')
        batch_size = params_dict.pop('batch_size')
        model_kwargs = params_dict
        params_dict2 = {
            'kwargs': model_kwargs,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'model': themodel_class,
            'weight_decay': float(weight_decay)
        }

        training_params = [params_dict2]


        # prepare training func kwargs
        results_file = 'ggnn_'+features.replace(' ','_')+'.json'
        datetime_str=datetime.now().strftime("%Y%m%d_%H%M%S_")
        results_file = 'results/'+datetime_str+results_file
        logger.debug(" saving results to "+results_file)


        kwargs = {
            'model_list': training_params, 
            'k': 3,
            'train_dataset': train_dataset , 
            'balanced': True, 
            'force_numclasses': None, 
            'unbalanced_split': False, 
            'debug_training': True,
            'tfidf_indices': None
            #'results_folder': results_file, #folder and file
            #'models_folder': 'models/gnn/'
        }


        # performing training(and testing)
        start = time.time()
        training_func = getattr(training_module, training_func)
        logger.debug(" calling "+training_func.__name__)
        #logger.debug(kwargs)
        #pprint(kwargs,open(log_file,'a'))
        results_dict2 = training_func(**kwargs)

        results_dict = {'best_models_list':[], 'models':{} , 'best_models':{}, 'autoincrement': 0 }
        #pprint(results_dict2,open('../pipeline/'+log_file,'a'))
        reportModelSelectionResult(results_dict2,results_dict)
        #pprint(results_dict,open('../pipeline/'+log_file,'a'))
        logger.debug("test_multiple_models")
        logger.debug(" saving model to models/gnn")
        test_multiple_models(
            results_dict,
            test_dataset,
            results_file=results_file,
            models_folder='models/gnn/')
        end= time.time()
        logger.debug("training time: "+str(round(end-start))+"s")
        logger.debug("saving to "+results_file)

        add_dataset_file(results_file, dataset_folder, str(round(end-start)))


    # return 
    os.chdir('../pipeline')

# read file path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file',type=str)
args= parser.parse_args()
logger.debug("training "+args.file)

# load file info
jobdict = yaml.load(open(args.file,'r'), Loader=yaml.FullLoader)
pprint(jobdict,open(log_file,'a'))


# train!
try:
    training_dispatcher(jobdict)

    # move training results to tests folder 
    yaml.dump(jobdict,open(args.file.replace('trains','done_trainings'),'w'))
    if os.path.exists(args.file):
      os.remove(args.file)

    logger.debug("moved jobdict to "+args.file.replace('trains','done_trainings'))
    logger.debug("Finished!\n")
except Exception as err:
    logger.exception("Error with "+args.file+"\n")
    