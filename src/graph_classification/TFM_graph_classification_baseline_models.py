from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch
import torch.nn.functional as F
from TFM_graph_classification import *



def get_static_topological_features_of_graphs(dataset):
    """
        get x_topo_feats for each graph of the dataset and merge it into a torch.LongTensor()

        get y for each graph of the dataset and merge it into a torch.LognTensor

        Returns the matrix of static topological features x for of all the graphs in the dataset.
        Returns also the vector of classes y
    """

    n = len(dataset)
    ncols = len(dataset[0].x_topo_feats)
    #X = torch.FloatTensor(dataset[0].x_topo_feats).view(-1,ncols)
    #Y = torch.LongTensor(dataset[0].y)
    # for i in range(1,n):
    #     new_X_row = torch.FloatTensor(dataset[i].x_topo_feats).view(-1,ncols)
    #     X = torch.cat((new_X_row,X), dim=0)
    #     Y = torch.cat((torch.LongTensor(dataset[i].y),Y), dim=0)

    X = []
    Y = []
    for i in range(n):
        X.append(dataset[i].x_topo_feats)
        Y.append(dataset[i].y)

    X = np.array(X)
    Y = np.array(Y)
    print(type(X), type(Y))
    return X,Y



def report_baseline_model_selection(model_dict, results_dict):

    """
        Add new trained model to models but also to testing.

        PENDING IMPROVEMENT: the testing should contain the top n best models... However this is still useful 

    """

    results_dict['models'][model_dict['model']+model_dict['params']]=model_dict
    results_dict['testing'][model_dict['model']+model_dict['params']]=model_dict

    pprint(model_dict)


def fit_model(train_dataset, test_dataset, datasetname, clf, model_name, params_list):


    X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
    X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
    clf.fit(X_train, y_train)  
    
    # predicting
    y_test_pred = clf.predict(X_test)

    # Error measures: 
    cv_measures = F1Score(clf.predict(X_train), y_train)
    measures = F1Score(y_test_pred, y_test)
    #measures = { 
    #    i:{'TP':0, 'TN':0, 'FP':0, 
    #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
    #    for i in range(num_classes)}


    params_str = ''
    if 'best_params_' in dir(clf) and len(params_list)>0:
        for p in params_list:
            params_str += str(p)+"="+str(clf.best_params_[p])+"_"

    # Add here any other special case of best_params

    #print(clf.best_params_)
    result = {
        'model': model_name,
        'model_long': model_name +"-"+ params_str ,
        'params': params_str,
        'dataset': datasetname,
        'cv_accuracy': cv_measures['accuracy'],
        'cv_microF1': cv_measures['microF1'],
        'cv_macroF1': cv_measures['macroF1'],
        'accuracy': measures['accuracy'],
        'microF1': measures['microF1'],
        'macroF1': measures['macroF1']
    }
    return result



def fit_logistic_regression(train_dataset, test_dataset, datasetname):

    clf = linear_model.LogisticRegression(
        solver='lbfgs', 
        C=1e5,
        multi_class='multinomial'
        )

    parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
    clf = GridSearchCV(clf, parameters, cv=3, refit=True)

    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'Logistic', 
              parameters.keys())
    


def fit_svm(train_dataset, test_dataset, datasetname):
    

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')

    # model selection
    parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
    clf = GridSearchCV(clf, parameters, cv=3, refit=True)

    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'SVM', 
              parameters.keys())
    
    



def fit_linear_svm(train_dataset, test_dataset, datasetname):
    
  
    # fitting
    clf = LinearSVC(random_state=0, tol=1e-5)

    # model selection
    parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
    clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'linearSVM', 
              parameters.keys())



def fit_knn(train_dataset, test_dataset, datasetname):
    
    
    # fitting
    n_neighbors = 3
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # model selection
    parameters = {
        'n_neighbors': [1,3,5,10,100],
        'weights' : ['uniform','distance']
        }
    clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
        
    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'knn', 
              parameters.keys())


def fit_decision_tree(train_dataset, test_dataset, datasetname):
    
    
    # fitting
    n_neighbors = 3
    weights = 'uniform'
    clf = tree.DecisionTreeClassifier()

    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'DecisionTree', 
              [])




def fit_random_forrest(train_dataset, test_dataset, datasetname):
    
    
    # fitting
    n_neighbors = 3
    weights = 'uniform'
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    # model selection
    parameters = {
        'n_estimators': [50,100,250,500],
        'max_depth' : [2,5,10]
        }
    clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
    return fit_model(train_dataset, 
              test_dataset, 
              datasetname, 
              clf, 
              'RandomForrest', 
              parameters.keys())



class mlp1(torch.nn.Module):
    def __init__(self, d1=10,d2=20,num_classes=3):
        super(mlp1, self).__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_classes)
        
    def forward(self, data):
        x = data.x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

def get_topo_features_of_batch(batch):

    X,Y = get_static_topological_features_of_graphs(batch)

    return(Data(x=X,y=Y))

def train_mlp(model, loader, optimizer, train_loss_history):
    global device 
    
    model.train()
    loss_train = 0.0
    total_num_graphs = 0
    for batch in loader:
        
        # first transform the batch 
        # to a torch.Tensor X matrix
        data = get_topo_features_of_batch(batch)

        data = batch.to(device)

        optimizer.zero_grad()
        out = model(data)
        target = data.y
        loss = F.nll_loss(out, target)
        loss_train +=loss
        loss.backward()
        optimizer.step()
        total_num_graphs += data.num_graphs
        
    loss_train = loss_train /total_num_graphs
    train_loss_history.append(loss_train.item()) 
    

# def mlp_dataloader(train, batch_size=10, shuffle=True):
#     """
#         Goal is an iterable that contians bs number of graphs in each item
#     """
#     bs = batch_size
#     n = len(train)
#     num_batches = n//bs
#     if n % bs > 0:
#         num_batches +=1
#     loader = []
#     for i in range(num_batches):

#         graphi = i*bs
#         graphj = (i+1)*bs # last elem is not picked inside list slice
#         if i == num_batches -1 :
#             # last batch
#             graphj = len(train)
#         # get bs graphs

#         #  list of graphs from training
#         train_list = []
#         for iindex in range(graphi, graphj+1):
#             try:
#                 train_list.append(train[iindex])
#             except:
#                 pass
        

#         # extract their X vector and Y value
#         X,Y = get_static_topological_features_of_graphs(train_list)

#         # construct a torch.Tensor X and Y
#         loader.append(Data(x=X, y=Y))

#     return loader

def dummy_mlp_training(train_dataset,test_dataset, datasetname, epochs=5):
    """
        NO CV just loader 
    """

    #  initial setup
    # initial setup
    global device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss_history = []

    model = mlp1(d1=10,d2=5) 
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


    #loader = mlp_dataloader(train_dataset, batch_size=10, shuffle=True)

    loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    pprint(loader)
    print(dir(loader))
    

    for epoch in range(epochs):
        train_mlp(model, loader, optimizer, train_loss_history)
      
    return model, train_loss_history  


def fit_mlp(train_dataset, test_dataset, datasetname):
    model, train_loss = dummy_mlp_training(train_dataset, test_dataset, datasetname)
    return {}


def model_selection_mlp(train_dataset, test_dataset, datasetname,model_list, k=3, balanced=True, force_numclasses=None, unbalanced_split=False):
    """
        PyTorch MLP implementation

        tasks
        -----
        1. construct model selection function
        2. implement the loader

    """
    # initial setup
    global device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if force_numclasses is not None:
        train_dataset.num_classes = force_numclasses

    kfolds = kFolding2(train_dataset,k, balanced, unbalanced_split)


    for modeldict in model_list:

        train_loss_history = []
        val_history = {'loss':[], 'accuracy':[], 'microF1':[],'macroF1':[]}
        modeldict['cv_val_loss']=0.0
        modeldict['cv_val_accuracy']=0.0
        modeldict['cv_val_microF1'] =0.0
        modeldict['cv_val_macroF1'] =0.0

        epochs = modeldict['epochs']
        modelclass = modeldict['model']
        kwargs = modeldict['kwargs']

        try:
            model = modelclass(**kwargs) # model parameters are inside kwargs dict
            model = model.to(device)
            modeldict['model_instance'] = model
            start2 = time.time()

            lr = modeldict['learning_rate']
            wd = modeldict['weight_decay']
            bs = modeldict['batch_size']

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            for kfold in kfolds:

                train = train_dataset[kfold[0]]
                val = train_dataset[kfold[1]]

                # MODIFY
                loader = DataLoader(train, batch_size=bs, shuffle=True)
                loader_val = DataLoader(val, batch_size=bs, shuffle=True)
                for epoch in range(epochs):
                    train_model(model, loader, optimizer, train_loss_history)
                    val_loss_model(model, loader_val, optimizer, val_history)

                # save results
                modeldict['train_loss_history']=train_loss_history
                modeldict['val_loss_history']=val_history['loss']
                modeldict['val_accuracy_history']=val_history['accuracy']
                modeldict['val_loss']=val_history['loss'][-1]
                modeldict['accuracy']=val_history['accuracy'][-1]
                modeldict['microF1']=val_history['microF1'][-1]
                modeldict['macroF1']=val_history['macroF1'][-1]

                modeldict['cv_val_loss']+=modeldict['val_loss']
                modeldict['cv_val_accuracy']+=modeldict['accuracy']
                modeldict['cv_val_microF1']+=modeldict['microF1']
                modeldict['cv_val_macroF1']+=modeldict['macroF1']
        
            modeldict['cv_val_loss']=modeldict['cv_val_loss']/len(kfolds)
            modeldict['cv_val_accuracy']=modeldict['cv_val_accuracy']/len(kfolds)
            modeldict['cv_val_microF1']=modeldict['cv_val_microF1']/len(kfolds)
            modeldict['cv_val_macroF1']=modeldict['cv_val_macroF1']/len(kfolds)

        except:
            print("Problem training model "+modeldict['model'].__name__)
            traceback.print_exc()

        end2 = time.time()
        modeldict['time'] = end2 - start2
        # report model results
        if debug_training: 

            reportTrainedModel(modeldict)
        

        
    # select best model
    # and train again the best model
    modelsdict = select_best_model(model_list, train_dataset)

    
    return modelsdict


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# def fit_logistic_regression(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     log = linear_model.LogisticRegression(solver='lbfgs', C=1e5,
#                                           multi_class='multinomial')


#     # model selection
#     parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
#     log = GridSearchCV(log, parameters, cv=3, refit=True)
#     #print(y_train)
#     log.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = log.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(log.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

#     #print(log.best_params_)
#     result = {
#         'model': 'Logistic',
#         'model_long': log.best_estimator_,
#         'params': 'C='+str(log.best_params_['C']),
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result


# def fit_svm(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     clf = svm.SVC(gamma='scale', decision_function_shape='ovo')

    

#     # model selection
#     parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
#     clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
#     clf.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = clf.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(clf.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

    
#     print(clf.best_params_)
#     result = {
#         'model': 'SVM',
#         'model_long': 'SVM' 
#          +"_C=" + str(clf.best_params_['C'])
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         # +"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'params': "_C=" + str(clf.best_params_['C'])
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         # +"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result


# def fit_linear_svm(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     clf = LinearSVC(random_state=0, tol=1e-5)

#     # model selection
#     parameters = {'C': [1e6,  1e4,  1e2,1,  0.01]}
#     clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
#     clf.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = clf.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(clf.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

#     #print(clf.best_params_)
#     result = {
#         'model': 'linearSVM',
#         'model_long': 'linearSVM' 
#          +"_C=" + str(clf.best_params_['C'])
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         #+"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'params': "_C=" + str(clf.best_params_['C'])
#          #+"_krnl=" + str(clf.best_params_['kernel'])
#          #+"_gamma=" + str(clf.best_params_['gamma'])
#          ,
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result



# def fit_knn(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     n_neighbors = 3
#     weights = 'uniform'
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

#     # model selection
#     parameters = {
#         'n_neighbors': [1,3,5,10,100],
#         'weights' : ['uniform','distance']
#         }
#     clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
#     clf.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = clf.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(clf.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

#     #print(clf.best_params_)
#     result = {
#         'model': 'KNN',
#         'model_long': 'KNN' 
#           + str(clf.best_params_)
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         #+"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'params':  str(clf.best_params_)
#          #+"_krnl=" + str(clf.best_params_['kernel'])
#          #+"_gamma=" + str(clf.best_params_['gamma'])
#          ,
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result


# def fit_decision_tree(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     n_neighbors = 3
#     weights = 'uniform'
#     clf = tree.DecisionTreeClassifier()

#     # no parametere search
#     clf.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = clf.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(clf.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

#     #print(clf.best_params_)
#     result = {
#         'model': 'DecisionTree',
#         'model_long': 'DecisionTree' 
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         #+"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'params': ''
#          #+"_krnl=" + str(clf.best_params_['kernel'])
#          #+"_gamma=" + str(clf.best_params_['gamma'])
#          ,
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result





# def fit_random_forrest(train_dataset, test_dataset, datasetname):
    
#     X_train,y_train = get_static_topological_features_of_graphs(train_dataset)
#     X_test,y_test = get_static_topological_features_of_graphs(test_dataset)
    
#     # fitting
#     n_neighbors = 3
#     weights = 'uniform'
#     clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

#     # model selection
#     parameters = {
#         'n_estimators': [50,100,250,500],
#         'max_depth' : [2,5,10]
#         }
#     clf = GridSearchCV(clf, parameters, cv=3, refit=True)
    
#     clf.fit(X_train, y_train)  
    
#     # predicting
#     y_test_pred = clf.predict(X_test)

#     # Error measures: 
#     cv_measures = F1Score(clf.predict(X_train), y_train)
#     measures = F1Score(y_test_pred, y_test)
#     #measures = { 
#     #    i:{'TP':0, 'TN':0, 'FP':0, 
#     #       'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
#     #    for i in range(num_classes)}

#     #print(clf.best_params_)
#     result = {
#         'model': 'RandomForrest',
#         'model_long': 'RandomForrest' 
#           + str(clf.best_params_)
#         # +"_krnl=" + str(clf.best_params_['kernel'])
#         #+"_gamma=" + str(clf.best_params_['gamma'])
#         ,
#         'params':  str(clf.best_params_)
#          #+"_krnl=" + str(clf.best_params_['kernel'])
#          #+"_gamma=" + str(clf.best_params_['gamma'])
#          ,
#         'dataset': datasetname,
#         'cv_accuracy': cv_measures['accuracy'],
#         'cv_microF1': cv_measures['microF1'],
#         'cv_macroF1': cv_measures['macroF1'],
#         'accuracy': measures['accuracy'],
#         'microF1': measures['microF1'],
#         'macroF1': measures['macroF1']
#     }
#     return result





