"""
This is a hack to use this kind of gnn layer with the stable version of
PyTorch Geometric as of 2019-03-25

"""
import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
#from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool
import torch.nn as nn


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import os


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, out_channels, num_layers, aggr='add', bias=True):
        super(GatedGraphConv, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()


    def forward(self, x, edge_index):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])
            # original master 1.0.3 (new version with problems when using rnn)
            #m = self.propagate(edge_index, x=m)
            # hacky version to use with the pip installation of pytorch-geometric 20190325
            m = self.propagate('add',edge_index, x=m)
            h = self.rnn(m, h)

        return h


    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)


# compute PRE, REC and F1
def PRE(measuresdict):
    m = measuresdict
    measuresdict['PRE'] = float(m['TP'])/float(m['TP']+m['FP'])
    
def REC(m):
    m['REC'] = float(m['TP'])/float(m['FN']+m['TP'])

def F1(m):
    m['F1']=2.0*(m['PRE']*m['REC'])/(m['PRE']+m['REC'])
    
def macroAndMicroScores(m):
    # average all precisions
    # average all recalls
    # compute macroF1
    macroPRE = 0.0
    macroREC = 0.0
    num_classes = 0
    microPREnumerator = 0.0
    microPREdenominator = 0.0
    microRECnumerator = 0.0
    microRECdenominator = 0.0
    for k,v in m.items():
        try:
            a = int(k)
            macroPRE+=m[k]['PRE']
            macroREC+=m[k]['REC']
            
            microPREnumerator+=m[k]['TP']
            microPREdenominator+=m[k]['TP']
            microPREdenominator+=m[k]['FP']
            
            microRECnumerator+=m[k]['TP']
            microRECdenominator+=m[k]['TP']
            microRECdenominator+=m[k]['FN']
            
            num_classes+=1
        except:
            # only keys related to classes
            # avoid macro and micro keys
            pass
        
    macroPRE = macroPRE/float(num_classes)
    macroREC = macroREC/float(num_classes)
    macroF1 = 2.0*(macroPRE*macroREC)/(macroPRE+macroREC)
    m['macroPRE'] = macroPRE
    m['macroREC'] = macroREC
    m['macroF1'] = macroF1
    
    microPRE = microPREnumerator/microPREdenominator
    microREC = microRECnumerator/microRECdenominator
    microF1 = 2.0*(microPRE*microREC)/(microPRE+microREC)
    m['microPRE'] = microPRE
    m['microREC'] = microREC
    m['microF1'] = microF1
    

def F1Score(pred, target):
    predset = set(pred)
    targetset = set(target)
    #print(predset)
    #print(targetset)
    num_classes = max(len(predset),len(targetset))

    # for each class save pred_indices, and target_indices
    preddict = { i:[] for i in range(num_classes) }
    targetdict = { i:[] for i in range(num_classes) }
    #print(preddict)

    for i in range(len(pred)):
        preddict[pred[i]].append(i)
        targetdict[target[i]].append(i)

    #print("preddict", preddict)
    #print("targetdict", targetdict)

    measures = { 
        i:{'TP':0, 'TN':0, 'FP':0, 
           'FN':0, 'PRE':0.0, 'REC': 0.0, 'F1':0.0} 
        for i in range(num_classes)}
    for i in range(num_classes):
        for j in range(len(preddict[i])):
            if preddict[i][j] in targetdict[i]:
                measures[i]['TP']+=1
            else:
                measures[i]['FP']+=1

        for j in range(len(targetdict[i])):
            if targetdict[i][j] not in preddict[i]:
                measures[i]['FN']+=1

        for j in range(len(pred)):
            if pred[j] not in preddict[i] and pred[j] not in targetdict[i]:
                measures[i]['TN']+=1

    #print(" single measures",measures)
    for k,mdict in measures.items():
        try:
            PRE(mdict)
        except:
            #print("could not compute PRE on class ",k)
            pass
        try:
            REC(mdict)
        except:
            #print("could not compute REC on class ",k)
            pass
        try:
            F1(mdict)
        except:
            #print("could not compute F1 on class ",k)
            pass

    macroAndMicroScores(measures)

    return measures


# k-fold cross-validation

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import os

# count how many graphs of each class in the dataset
def printDatasetBalance(dataset):
    num_classes = dataset.num_classes
    class_counts = { i:0 for i in range(num_classes)}
    #print(class_counts)
    for graph in dataset:
        class_counts[int(graph.y.item())]+=1
    print(class_counts)
    
    
def balancedDatasetSplit_list(dataset, prop):
    
    dataset = dataset.shuffle()
    n = len(dataset)
    test_lim= int(prop*n)
    num_classes = dataset.num_classes
    
    train_dataset = []
    test_dataset = []
    datasets_byclass = {i:[] for i in range(num_classes)}
    
    
    # for each class repeat balanced split
    for graph in dataset:
        datasets_byclass[int(graph.y.item())].append(graph)
    
    for c in range(num_classes):
        nc = len(datasets_byclass[c])
        limit = int(prop*nc)
        train_dataset.extend(datasets_byclass[c][:limit])
        test_dataset.extend(datasets_byclass[c][limit:])
        
    return train_dataset, test_dataset

    
def balancedDatasetSplit_slice(dataset, prop):
    
    #dataset = dataset.shuffle()
    n = len(dataset)
    test_lim= int(prop*n)
    num_classes = dataset.num_classes
    
    x=torch.Tensor([True,False,True])==True

    train_list = []
    test_list = []
    train_dataset_slice = [False]*n
    test_dataset_slice = [False]*n
    datasets_byclass = {i:[] for i in range(num_classes)}
    
    #print("train_dataset_slice", train_dataset_slice)
    #print("test_dataset_slice", test_dataset_slice)
    #print("datasets_byclass", datasets_byclass)
    
    # for each class repeat balanced split
    for i in range(n):
        graph = dataset[i]
        datasets_byclass[int(graph.y.item())].append(i)

    #print("datasets_byclass",datasets_byclass)
        
    for c in range(num_classes):
        nc = len(datasets_byclass[c])
        limit = int(prop*nc)
        train_list.extend(datasets_byclass[c][:limit])
        test_list.extend(datasets_byclass[c][limit:])
        
    #print("train_list", train_list)
    #print("test_list", test_list)

        
    # now from list of integers(indices) to boolean mask tensor
    #for i in range(len(train_list)):
    #    real_index = train_list[i]
    #    train_dataset_slice[real_index] = True
        
    #for i in range(len(test_list)):
    #    real_index = test_list[i]
    #    test_dataset_slice[real_index] = True
        
    #print("train_dataset_slice", train_dataset_slice)
    #print("test_dataset_slice", test_dataset_slice)
        
    train_dataset = dataset[torch.LongTensor(train_list)]
    test_dataset = dataset[torch.LongTensor(test_list)]
        
    #print("train_dataset", train_dataset)
    #print("test_dataset", test_dataset)
    
    return train_dataset, test_dataset


def balancedDatasetKfoldSplit_slice(dataset,k):
    
    #dataset = dataset.shuffle()
    n = len(dataset)
    
    foldsize = int(n/k)
    num_classes = dataset.num_classes
    num_items_x_class = int(foldsize/num_classes)
    
    # list of items for each class
    train_list = []
    test_list = []
    datasets_byclass = {i:[] for i in range(num_classes)}
    for i in range(n):
        graph = dataset[i]
        datasets_byclass[int(graph.y.item())].append(i)

    #print(datasets_byclass)
    
    folds = []
    for i in range(k):
        folds.append([])
        for c in range(num_classes):
            for j in range(num_items_x_class):
                index = datasets_byclass[c].pop()
                folds[i].append(index)
        
    # returns a list of list of indices
    return folds

def kFolding(train_dataset, k):
    n = len(train_dataset)
    fold_size = int(n/k)
    
    # build folds
    #folds = []
    #for i in range(k):
    #    i1 = i*fold_size
    #    i2 = i1+fold_size
    #    folds.append((i1,i2))
    #print(folds)
    
    # build train-val sets
    train_sets =[]
    for i in range(k):
        preval_index = (0,i*fold_size)
        val_index = (i*fold_size,i*fold_size+fold_size)
        postval_index = (i*fold_size+fold_size,n)
        train_sets.append((preval_index, val_index, postval_index))
        
    #print(train_sets)
    return train_sets

def kFolding2(train_dataset, k):

    #print(" train_dataset len:", len(train_dataset))
    folds = balancedDatasetKfoldSplit_slice(train_dataset, k)
    train_sets =[]
    for i in range(k):
        # each train_set must have a torch.LongTensor for train indices
        # and a torch.LongTensor for val indices
        val_merge = folds[i]
        train_merge = [] 
        for j in range(k):
            if j != i:
                train_merge.extend(folds[j])
        train_sets.append((torch.LongTensor(train_merge), torch.LongTensor(val_merge)))
    
    return train_sets
    


def accuracy(pred, batch):
    correct = pred.eq(batch.y).sum().item()
    #acc = correct / test_dataset.sum().item()
    acc = correct / batch.num_graphs
    return acc



def train_model(model, loader, optimizer, train_loss_history):
    global device 
    
    model.train()
    loss_train = 0.0
    total_num_graphs = 0
    for batch in loader:
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
    
def val_loss_model(model, loader, optimizer, val_history):
    global device 
    
    model.eval()
    loss_val = 0.0
    total_num_graphs = 0
    total_pred = []
    total_acc = []
    total_gt = []
    
    for batch in loader:
        data = batch.to(device)
        pred = model(batch)
        total_pred.extend(pred.flatten().tolist())
        total_gt.extend(batch.y.flatten().tolist())
        
        _, predacc = pred.max(dim=1)
        total_acc.extend(predacc.flatten().tolist())
        
        target = data.y
        loss = F.nll_loss(pred, target)
        loss_val += loss
        total_num_graphs += data.num_graphs
        
    loss_val = loss_val / total_num_graphs
    val_history['loss'].append(loss_val.item())
    
    # accuracy needs correction
    val_history['accuracy'].append(accuracy(predacc, batch))
    
    # compute F1 scores
    #pred2 = pred.to('cpu')
    #pred2 = pred2.flatten().tolist()
    #target = batch.y.to('cpu')
    #target = target.flatten().tolist()
    
    #print("total_acc",total_acc)
    #print("total_gt",total_gt)
    measures = F1Score(total_acc, total_gt)
    val_history['microF1'].append(measures['microF1'])
    val_history['macroF1'].append(measures['macroF1'])

# Retrain the best model
def final_model_train(modeldict, train_dataset):
    global device 
    
    epochs = modeldict['epochs']
    modelclass = modeldict['model']
    kwargs = modeldict['kwargs']
    model = modelclass(**kwargs)
    model = model.to(device)
    train_loss_history=[]
    
    lr = modeldict['learning_rate']
    wd = modeldict['weight_decay']
    bs = modeldict['batch_size']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    for epoch in range(epochs):
        train_model(model, loader, optimizer, train_loss_history)
        
    return model



# model saving


def modelSaveName(modeldict):
    classname = modeldict['model_instance'].__class__.__name__
    architecture = ""
    for k,v in modeldict['kwargs'].items():
        architecture = architecture+'_'+str(k)+'-'+str(v)
    
    epochs = modeldict['epochs']
    lr = modeldict['learning_rate']
    wd = modeldict['weight_decay']
    bs = modeldict['batch_size']
    
    finalname = classname + "_" + architecture + "_" + \
                str(epochs) + "_" + str(lr) + \
                str(wd) + "_" + str(bs)
    return finalname

def saveModel(modeldict):
    import traceback
    
    """
        based on :
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    
    try:
        
        if not os.path.exists('./models'):
            os.mkdir('./models')
        
        # model naming convention
        model = modeldict['model_instance']
        path = './models/'+modelSaveName(modeldict)
            
        # save operation
        torch.save(model.state_dict(),path)
            
        return path
    except Exception as err:
        print("ERROR SAVING MODEL "+model.__class__.__name__)
        print(err)
        
        traceback.print_exc()
        return None
        
def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    
    
def testSavingLoadingModel(train_dataset, test_dataset):

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    m1 = {'epochs': 20,
    'model': Net1,
    'kwargs':{'d1': 50,'d2': 20,'num_layers':2, 'aggr_type':'mean'}, 
    'learning_rate': 0.01, 'weight_decay':5e-4, 'batch_size': 10}

    # train model
    model = final_model_train(m1, train_dataset)
    

    # test model and print accuracy
    testresult = testModel(model, test_dataset)

    # save model
    m1['model_instance']=model
    path = saveModel(m1)

    # create new similar model
    m2 = {'epochs': 200,
    'model': Net1,
    'kwargs':{'d1': 50,'d2': 20,'num_layers':2, 'aggr_type':'mean'}, 
    'learning_rate': 0.01, 'weight_decay':5e-4, 'batch_size': 32}
        
    epochs = m2['epochs']
    modelclass = m2['model']
    kwargs = m2['kwargs']
    model2 = modelclass(**kwargs)
    model2 = model2.to(device)
    m2['model_instance']=model2
    

    # load state_dict
    print("path", path)
    if path is not None:
        loadModel(model2, path)

        # test new model and print accuracy
        testresult = testModel(model2, test_dataset)


    
def saveModels(modelsdict):
    for k,model in modelsdict['best_models'].items():
        saveModel(model)

def reportTrainedModel(modeldict):
    print(" trained model: ",modeldict['model'].__class__.__name__,
              modeldict['kwargs'], " epochs:",modeldict['epochs'],
             ' val loss=',modeldict['cv_val_loss'],
          ' val accuracy=',modeldict['cv_val_accuracy'],
         ' val microF1=',modeldict['cv_val_microF1'],
          ' val macroF1=',modeldict['cv_val_macroF1'])
    
def selectBestModel(model_list):
    # select the best model (lower validation loss)
    losses = np.array([ modeldict['cv_val_loss'] for modeldict in model_list])
    accuracies = np.array([ modeldict['cv_val_accuracy'] for modeldict in model_list])
    microF1 = np.array([ modeldict['cv_val_microF1'] for modeldict in model_list])
    macroF1 = np.array([ modeldict['cv_val_macroF1'] for modeldict in model_list])
    best_model_loss = model_list[np.argmin(losses)]
    best_model_acc = model_list[np.argmax(accuracies)]
    best_model_microF1 = model_list[np.argmax(microF1)]
    best_model_macroF1 = model_list[np.argmax(macroF1)]
    
    # save selections to model_list
    modelsdict = {}
    modelsdict['models'] = model_list
    modelsdict['best_models']={}
    modelsdict['best_models']['loss'] = best_model_loss
    modelsdict['best_models']['accuracy'] = best_model_acc
    modelsdict['best_models']['microF1'] = best_model_microF1
    modelsdict['best_models']['macroF1'] = best_model_macroF1

    modelsdict['testing']={}
    
    return modelsdict
    

    


def modelSelection(model_list,k, train_dataset ):    

    global device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfolds = kFolding2(train_dataset,k)

    for modeldict in model_list:

        epochs = modeldict['epochs']
        modelclass = modeldict['model']
        kwargs = modeldict['kwargs']
        model = modelclass(**kwargs)
        model = model.to(device)
        modeldict['model_instance'] = model
        
        lr = modeldict['learning_rate']
        wd = modeldict['weight_decay']
        bs = modeldict['batch_size']

        train_loss_history = []
        val_history = {'loss':[], 'accuracy':[], 'microF1':[],'macroF1':[]}
        modeldict['cv_val_loss']=0.0
        modeldict['cv_val_accuracy']=0.0
        modeldict['cv_val_microF1'] =0.0
        modeldict['cv_val_macroF1'] =0.0

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for kfold in kfolds:

            train = train_dataset[kfold[0]]
            val = train_dataset[kfold[1]]
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
        
        # report model results
        reportTrainedModel(modeldict)
        

        
    # select best model
    modelsdict = selectBestModel(model_list)
    
    # save model to disk + save file path    
    # or save model in the dict.. (could take too much memory)
    saveModels(modelsdict)
    
    return modelsdict
        

    
def reportModelSelectionResult(modeldict):
    best_model_loss = modeldict['best_models']['loss']
    best_model_acc = modeldict['best_models']['accuracy']
    best_model_microF1 = modeldict['best_models']['microF1']
    best_model_macroF1 = modeldict['best_models']['macroF1']
    
    print("\n selected model from loss: ",best_model_loss['model'].__name__,
      best_model_loss['kwargs']," epochs:", best_model_loss['epochs'], 
      best_model_loss['cv_val_loss'], best_model_loss['cv_val_accuracy'], 
      best_model_loss['cv_val_microF1'], best_model_loss['cv_val_macroF1'])
    print(" selected model from accuracy: ",best_model_acc['model'].__name__,
          best_model_acc['kwargs']," epochs:",best_model_acc['epochs'],  
          best_model_acc['cv_val_loss'], best_model_acc['cv_val_accuracy'], 
          best_model_loss['cv_val_microF1'], best_model_loss['cv_val_macroF1'])
    print(" selected model from microF1: ",best_model_microF1['model'].__name__,best_model_microF1['kwargs'],
          " epochs:", best_model_microF1['epochs'],  
          best_model_microF1['cv_val_loss'], best_model_microF1['cv_val_accuracy'], 
          best_model_microF1['cv_val_microF1'], best_model_microF1['cv_val_macroF1'])

    print(" selected model from macroF1: ",best_model_macroF1['model'].__name__,best_model_macroF1['kwargs'],
          " epochs:", best_model_macroF1['epochs'],  
          best_model_macroF1['cv_val_loss'], best_model_macroF1['cv_val_accuracy'], 
          best_model_macroF1['cv_val_microF1'], best_model_macroF1['cv_val_macroF1'])
    
    # report with Pandas table
    res = pd.DataFrame({
        'best_model_loss': best_model_loss, 
        'best_model_acc' : best_model_acc, 
        'best_model_microF1' : best_model_microF1, 
        'best_model_macroF1': best_model_macroF1})
    

def reportTest(batch, pred, measures, test_dataset):
    print("len(test_dataset): ", len(test_dataset))
    print("num graphs: ", batch.num_graphs)
    print(pred)
    print(batch.y)
    print('Accuracy: {:.4f}'.format(measures['accuracy'])," macroF1:",measures['macroF1'], " microF1:", measures['microF1'])

def testModel(model, test_dataset):
    global device
    
    model.eval()
    loader = DataLoader(test_dataset, batch_size= len(test_dataset), shuffle=True)
    for batch in loader:
        batch = batch.to(device)
        #_, pred = model(test_dataset).max(dim=1)
        _, pred = model(batch).max(dim=1)
        acc = accuracy(pred, batch)    
        pred2 = pred.to('cpu')
        pred2 = pred2.flatten().tolist()
        target = batch.y.to('cpu')
        target = target.flatten().tolist()
        measures = F1Score(pred2, target)
        measures['accuracy']=acc
        
        reportTest(batch, pred, measures, test_dataset)
        
        
        
    return measures

def reportAllTest(modelsdict):
    reportDict = [{'name':k, 
                   'accuracy': v['accuracy'],
                  'macroF1': v['macroF1'],
                  'microF1': v['microF1']} for k,v in modelsdict['testing'].items()]
    #print(reportDict)
    res = pd.DataFrame(reportDict)
    display(res)
        
def saveResults(modelsdict):
    import json
    import datetime
    import os
    import copy
    
    savedict = {}
    for model in modelsdict['models']:
        savedict['models']=[]
        #v2['train_loss_history']=[]
        #v2['val_loss_history']=[]
        #v2['val_accuracy_history']=[]
        mod = copy.deepcopy(model)
        mod['model']=model['model'].__name__
        mod['model_instance']=model['model_instance'].__class__.__name__
        savedict['models'].append(mod)
        
    savedict['best_models']={}
    for k,v2 in modelsdict['best_models'].items():
        #v2['train_loss_history']=[]
        #v2['val_loss_history']=[]
        #v2['val_accuracy_history']=[]
        savedict['best_models'][k]=copy.deepcopy(v2)
        savedict['best_models'][k]['model'] =v2['model'].__name__
        savedict['best_models'][k]['model_instance'] =v2['model_instance'].__class__.__name__
            
    savedict['tests'] = modelsdict['testing']
            
    
    
    d = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') 
    if not os.path.exists('./results'):
        os.mkdir('./results')
    results_file = './results/experiment_'+d
    
    with open(results_file, 'w') as outfile:
        json.dump(savedict, outfile)



def test_saving_model():
    # testing saving Model
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    dataset = dataset.shuffle()
    k = 3
    n = len(dataset)
    train_dataset, test_dataset = balancedDatasetSplit_slice(dataset, prop=0.8)
    testSavingLoadingModel(train_dataset, test_dataset)

if __name__=='__main__':
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    dataset = dataset.shuffle()
    k = 3
    n = len(dataset)
    print(" n:",n," k folds=",k)
    train_dataset, test_dataset = balancedDatasetSplit_slice(dataset, prop=0.8)
    print("Datasets balancing: ")
    printDatasetBalance(dataset )
    printDatasetBalance(train_dataset )
    printDatasetBalance(test_dataset )
    print()

    modelsdict = modelSelection(model_list,k, train_dataset)
    reportModelSelectionResult(modelsdict)

    bmodel = final_model_train(modelsdict['best_models']['loss'], train_dataset)
    testresult = testModel(bmodel, test_dataset)
    modelsdict['testing'][bmodel.__class__.__name__+'loss']=testresult

    bmodel = final_model_train(modelsdict['best_models']['accuracy'], train_dataset)
    testresult = testModel(bmodel, test_dataset)
    modelsdict['testing'][bmodel.__class__.__name__+'accuracy']=testresult

    bmodel = final_model_train(modelsdict['best_models']['microF1'], train_dataset)
    testresult = testModel(bmodel, test_dataset)
    modelsdict['testing'][bmodel.__class__.__name__+'microF1']=testresult

    bmodel = final_model_train(modelsdict['best_models']['macroF1'], train_dataset)
    testresult = testModel(bmodel, test_dataset)
    modelsdict['testing'][bmodel.__class__.__name__+'macroF1']=testresult

    reportAllTest(modelsdict)
    saveResults(modelsdict)     