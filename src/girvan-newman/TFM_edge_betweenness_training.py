import networkx as nx
import time
import torch
from torch_geometric.data import DataLoader
import importlib
import torch
from torch_geometric.data import DataLoader
import networkx as nx
from torch_geometric.data import Data
import pickle
import math
import pandas as pd
from IPython.display import display, HTML
import os
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec

from TFM_edge_betweenness_models import *

#%matplotlib inline
plt.style.use('seaborn-whitegrid')

#--------------------dataset-preparation-------------

def loadDataset(collection, name=None):
    try:
        # import datasets
        themodule = importlib.import_module("torch_geometric.datasets")
        # get the function corresponding to collection
        method_to_call = getattr(themodule, collection)
        if name:
            dataset = method_to_call(root='./data/'+str(collection), name=name)
            dataset.filename = name
            return dataset
        else:
            return method_to_call(root='./data/'+str(collection)) 
    except:
        # custom module
        method_to_call = globals()[collection]
       
        if name:
            
            dataset = method_to_call(root='./data/'+str(collection), name=name)
            dataset.filename = name
            return dataset
        else:
            return method_to_call(root='./data/'+str(collection)) 
        

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    try:
        return math.floor(n*multiplier + 0.5) / multiplier
    except:
        return 0

def transformMask(mask):
    train_mask = []
    i = 0
    for pick in mask:
        if pick[0]==1:
            train_mask.append(i)
        i+=1
    return torch.unsqueeze(torch.LongTensor(train_mask),dim=1)


def shuffleTrainTestMasks(data, trainpct = 0.7):
    ysize = list(data.y.size())[0]
    data.train_mask = torch.zeros(ysize,1, dtype=torch.long)
    data.train_mask[int(ysize*trainpct):] = 1
    data.train_mask = data.train_mask[torch.randperm(ysize)]
    data.test_mask = torch.ones(ysize,1, dtype=torch.long) - data.train_mask
    
    data.train_mask = transformMask(data.train_mask)
    data.test_mask = transformMask(data.test_mask)
  

def shuffleTrainTestValMasks(data, trainpct = 0.7, valpct = 0.2):

    ysize = list(data.y.size())[0]
    #print("total ", ysize)
    #print(" train ",int(ysize*trainpct)-int(ysize*trainpct*valpct))
    #print(" val ",int(ysize*trainpct*valpct))
    #print(" test ",int(ysize*(1- trainpct) ))
    data.train_mask = torch.zeros(ysize,1, dtype=torch.long)
    data.train_mask[:int(ysize*trainpct)] = 1
    data.train_mask = data.train_mask[torch.randperm(ysize)]
    #print(" train sum ",data.train_mask.sum())
    data.test_mask = torch.ones(ysize,1, dtype=torch.long) - data.train_mask
    #print(" test sum ",data.test_mask.sum())
    
    # transform to list of indexes
    data.train_mask = transformMask(data.train_mask)
    data.test_mask = transformMask(data.test_mask)
    
    
    data.val_mask = data.train_mask[:int(ysize*trainpct*valpct)]
    data.train_mask = data.train_mask[int(ysize*trainpct*valpct):]

    # print("shuffleTrainTestValMasks:")
    # print(data.train_mask)
    # print(data.val_mask)
    # print(data.test_mask)
    # print()
    print(type(data.train_mask))


def kFolding(dataset, k):
    """
       Pre:
            - The dataset must contain only one graph
            - the dataset must contain data.train_mask, test_mask and val_mask

        Post:
            - each kfold must contain nodes of the graph that are from the training and validation type
            or only from the training type

            - we will use indexes instead of actual nodes
 
            - no dataset balancing is performed since it's a regression task, and we would need to use some kind of histograming or distribution
    """
    train_sets = []
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for data in loader:
        shuffleTrainTestValMasks(data)
        print(type(dataset.train_mask))
        print(type(data.train_mask))
        print("outside function train mask: ", data.train_mask[:5])
        print("outside function  val mask: ", data.val_mask[:5])
        print("outside function test mask: ", data.test_mask[:5])
        print(" dataset.train_mask.size() ", dataset.train_mask.size())
        print(" data.train_mask.size() ", data.train_mask.size())
        #dataset.train_mask = torch.stack((dataset.train_mask,data.train_mask), dim=1)
        dataset.train_mask = torch.cat([dataset.train_mask, data.train_mask], dim=0)
        dataset.val_mask = torch.cat([dataset.val_mask,data.val_mask],dim=0)
        dataset.test_mask = torch.cat([dataset.test_mask,data.test_mask],dim=0)
        

        length_train_samples = dataset.train_mask.size()[0]+dataset.val_mask.size()[0]
        # try:
        #     ysize = list(data.y.size())[0]

        #     length_train_samples = int(ysize * 0.8)
        #     #length_val_samples = int(length_train_samples*0.2) # not used anymore!
        #     #data.train_mask = data.train_mask.extend(data.val_mask)
        # except:
        #     length_train_samples =length_train_samples = int(ysize * 0.8)
        #     #length_val_samples = 0 not used anymore!
        
        print("Kfolding: ")
            
        samples_per_fold = int(length_train_samples/k)
        for i in range(k):
            validation_indices = data.train_mask[i*samples_per_fold:(i+1)*samples_per_fold]
            training_indices = []
            for j in range(k):
                if j!=i:
                    training_indices.extend(data.train_mask[j*samples_per_fold:(j+1)*samples_per_fold])
            
            print(training_indices[:5])
            print(validation_indices[:5])
            print()

            train_sets.append((torch.LongTensor(training_indices), torch.LongTensor(validation_indices)))

        


    
    return train_sets




def datasetFromKFold(Kfold, dataset):
    """
        Given the training, validation and testin indices in kfold,
        modify the type of corresponding node in the graph to be in accordance to that kfold
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for data in loader:
        ysize = list(data.y.size())[0]
        data.train_mask = torch.zeros(ysize,1, dtype=torch.long)
        data.train_mask[ Kfold[0]] = 1
        data.val_mask = torch.zeros(ysize,1, dtype=torch.long)
        data.val_mask[ Kfold[1]] = 1
        data.test_mask = torch.ones(ysize,1, dtype=torch.long) - data.train_mask - data.val_mask
        
        data.train_mask = transformMask(data.train_mask)
        data.val_mask = transformMask(data.val_mask)
        data.test_mask = transformMask(data.test_mask)





 
 #--------------------error---------------------------------------   
    
def errors(t,y):
    """
        Compute RMSE and NRMSE
        
        ||t -y ||^2/n is the mean of the size of the error squared
        (mean from all samples)
        
        sqrt(||t-y||^2/n) is the root of this mean of squared errors
        so it has the units of t and y
        
        but the normalization means dividing by maxmin, 
        or sd, or IQR, 
        
        if this term is bigger than the rmse, then nrmse should be smaller than one.
        otherwise it should be bigger than one,
        meaning the error is so big that is is bigger than maxmin distance
        
    """

    print("\n\n errors()::------------------------")

    t = t.to('cpu')
    y = y.to('cpu')
    num_samples = list(t.size())[0]

    print("t",t)
    print("t[2:4] ",t[2:4])
    print(" t.size() ",t.size())
    print(" errors(), num_samples: ", num_samples)

    length_cuts = int(num_samples/10)
    num_cuts = int(num_samples/length_cuts)

    nrmse = 0
    rmse = 0
    normalizer = 0

    t_sum = 0
    t_tavg_sum = 0
    maxmin = 0
    smean = 0

    for i in range(num_cuts):

        t2 = t[i*length_cuts:(i+1)*length_cuts]
        t2 = t2.to(device)
        
        print(t2)
        t_sum += sum(t2).item()
        print(t_sum)
        
    print(" t_sum: ", t_sum)


    t_avg = float(t_sum/float(num_samples))

    for i in range(num_cuts):
        t2 = t[i*length_cuts:(i+1)*length_cuts]
        t2 = t2.to(device)
        #t2 = torch.squeeze(t2,1)
        print(" t2.sizer() ", t2.size())
        
        t_tavg_sum += sum((t2 - t_avg)**2).item()
        

        #maxmin = max(t) - min(t)
        #smean = torch.mean(t)

    #print("t shape",list(t.size()))
    #print("num_samples", num_samples)
    
    #print("t",t)
    #print("t_avg", t_avg)
    #print("t - t_avg",t - t_avg)
    #print((t - t_avg))
    #print((t - t_avg)**2)
    #print(sum((t - t_avg)**2))
    
    print(" t_tavg_sum", t_tavg_sum)
    print(num_samples)
    print()
    sample_variance = t_tavg_sum/(num_samples - 1)
    print("sample_variance ", sample_variance)
    import math
    sample_std = math.sqrt(sample_variance)

    # in wikipedia they use tmax-tmin or IQR
    
    
    
    normalizer = sample_std
    print("normalizer ", normalizer)
    
    #print("normalizer", normalizer)

    rmse = 0
    for i in range(num_cuts):
        t2 = t[i*length_cuts:(i+1)*length_cuts]
        t2 = t2.to(device)
        y2 = y[i*length_cuts:(i+1)*length_cuts]
        y2 = y2.to(device)
        y2 = torch.squeeze(y2,1)

        print(" t2.sizer() ", t2.size(), " y2.size() ", y2.size())
        thesum = torch.sum(t2 - y2, 0)
        print("thesum.size() ", thesum.size())
        thesq = thesum**2
        rmse += thesq.item()
        print(rmse)



    rmse = rmse/num_samples
    #nrmse = torch.sum((t - y) ** 2)/((num_samples-1)*sample_variance)
    #nrmse = rmse*num_samples/((num_samples-1)*sample_variance)
    nrmse = rmse
    rmse = math.sqrt(rmse)
    print(" rmse.sqrt() ", rmse)
    print(" normalizer ", normalizer)
    nrmse = math.sqrt(nrmse)/normalizer
    print(nrmse)
    #print("rmse",rmse)
    #print("nrmse", nrmse)
    return rmse, nrmse, normalizer

#----------------training-------------------------------------------




def train(model, loader, optimizer, train_loss_history):
    global device
    model.train()
    loss_train = 0.0
    total_num_nodes = 0
    for batch in loader:
        data = batch.to(device)

        optimizer.zero_grad()
        out = model(data)
        target = data.y
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss_train +=loss
        loss.backward()
        optimizer.step()
        total_num_nodes += data.train_mask.sum()
        
    loss_train = loss_train /total_num_nodes
    train_loss_history.append(loss_train.item()) 


def validation(model, loader, optimizer, val_loss_history):
    global device
    model.eval()
    loss_val = 0.0
    total_num_nodes = 0
    for batch in loader:
        data = batch.to(device)

        optimizer.zero_grad()
        out = model(data)
        target = data.y
        loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask])
        loss_val +=loss
        loss.backward()
        optimizer.step()
        total_num_nodes += data.val_mask.sum()
        
    loss_val = loss_val /total_num_nodes
    val_loss_history.append(loss_val.item()) 




def trainValidate(dataset, modeldict, res_dict={}):
    #global Net
    global device
    start = time.time()

    # 1.  prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #model = Net.to(device)  
    epochs = modeldict['epochs']
    modelclass = modeldict['model']
    kwargs = modeldict['kwargs']
    model = modelclass(**kwargs)
    model = model.to(device)

    train_loss_history = []
    val_history = {'loss':[]}
    modeldict['cv_val_loss']=0.0
    
    lr = modeldict['learning_rate']
    wd = modeldict['weight_decay']
    bs = modeldict['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # 2.  create a train_mask, and a test_mask (val_mask for further experiments)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    
    # 3. train & validate some epochs
    for epoch in range(epochs):
        train(model , loader, optimizer, train_loss_history )
        validation(model, loader, optimizer, val_history['loss'])

    modeldict['train_loss_history']=train_loss_history
    modeldict['val_loss_history']=val_history['loss']
    modeldict['val_loss']=val_history['loss'][-1]
    modeldict['cv_val_loss']+=modeldict['val_loss']

    model = model.to(torch.device('cpu'))
    modeldict['model_instance']=model



def selectBestModel(model_list):
    
    # select the best model (lower validation loss)
    losses = np.array([ modeldict['cv_val_loss'] for modeldict in model_list['models']])
    best_model_loss = model_list['models'][np.argmin(losses)]
    
    # save selections to model_list
    model_list['best_models']={}
    model_list['best_models']['loss'] = best_model_loss
    model_list['testing']={}




def finalTraining(model_list,  dataset ):
    
    global device
    start = time.time()

    # 1.  prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modeldict = model_list['best_models']['loss'] 
    epochs = modeldict['epochs']
    modelclass = modeldict['model']
    kwargs = modeldict['kwargs']
    model = modelclass(**kwargs)
    model = model.to(device)

    train_loss_history = []
    val_history = {'loss':[]}
    modeldict['cv_val_loss']=0.0
    
    lr = modeldict['learning_rate']
    wd = modeldict['weight_decay']
    bs = modeldict['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # 2. dataset preparation
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    
    # 3. train & validate some epochs
    for epoch in range(epochs):
        train(model , loader, optimizer, train_loss_history )

    model = model.to(torch.device('cpu'))
    modeldict['model_instance']=model
    

def testModel(model_list, dataset,  res_dict, start):
    
    print("testModel(), dataset.test_mask: ", dataset.test_mask[:5])
    #print("testModel(), testing_indices: ", testing_indices[:5])

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modeldict = model_list['best_models']['loss']
    modelclass = modeldict['model']
    kwargs = modeldict['kwargs']
    model = modelclass(**kwargs)
    model = model.to(device)
    epochs = modeldict['epochs']
    
    model.eval()
    loader = DataLoader(dataset, batch_size= len(dataset), shuffle=True)
    t = torch.FloatTensor(dataset[0].y.size())
    t = t.to(device)
    print(dataset[0].y.size())
    y = torch.FloatTensor(dataset[0].y.size())
    y = y.to(device)
    y = torch.unsqueeze(y, dim=1)
    for batch in loader:
        data = batch.to(device)

        #---------old code------------

        # regression 
        pred = model(data)
        #print("target: ",data.y[data.test_mask])
        #print("prediction: ",pred[data.test_mask])
        #print(pred[data.test_mask].type())
        #print(data.y[data.test_mask].type())
        
        # prepare the normalized mean root squared error
        #t.append(data.y[data.test_mask])
        #y.append(pred[data.test_mask])


        ground_truth = torch.index_select(data.y, 0, data.test_mask)
        print("testModels()")
        print("data.y.size() ", data.y.size())
        print("t.size() ", t.size())

        print("data.y[data.test_mask].size() " ,ground_truth.size())
        print(" data.test_mask[:3] ",data.test_mask[:3])
        t = torch.cat((t ,ground_truth) )
        print("pred.size() ", pred.size())
        print(" y.size() ", y.size())
        prediction = torch.index_select(pred, 0, data.test_mask)
        y = torch.cat((y , prediction) )
        

    negatives = False
    
    #b = torch.Tensor(len(y),1)
    #yout = torch.cat(y, out=b)

    #t = torch.LongTensor(t)
    #tout= torch.cat(t, out=b)
    
    if 0 > (sum(y<0)):
        negatives = True
    rmse, nrmse, normalizer = errors(t,y)

    endtime = time.time()
    
    
    # getting model hyper params
    hyperparams = ""
    model_ds = [elem+"="+str(getattr(model.__class__, elem)) for elem in dir(model.__class__) if elem.startswith("d") and len(elem)==2]
    hyperparams = "_".join(model_ds)
    theepochs = "epochs="+str(epochs)
    #dataset name
    thedataset = dataset.filename
    # basename
    thedataset = os.path.basename(thedataset)
    # extension
    thedataset = os.path.splitext(thedataset)[0]
    
    thetime = str(round_half_up(endtime-start,3) )
    print(" endtime", endtime, " start", start, " endtime - start ", endtime-start, " total time ", thetime)


    result = str(model)+" " \
            +hyperparams+" " \
            +theepochs+" " \
            +str(thedataset)+" " \
            +"rmse="+str(round_half_up(rmse,3))+" " \
            +"nrmse="+str(round_half_up(nrmse,3))+" " \
            +"time="+thetime  \
            +" negatives?"+str(negatives) \
            +"\n"
    
    res_dict['tables'][str(model)]={
                            #"model": str(model),
                             "hyperparams": hyperparams,
                             "dataset": str(thedataset),
                             "epochs": epochs,
                             "rmse":round_half_up(rmse,3),
                             "nrmse":round_half_up(nrmse,3),
                             "std":round_half_up(normalizer,3),
                             "time":thetime,
                            "neg vals": negatives }

    t = t.to('cpu')
    tfinal = np.array(t.detach().numpy())
    y = y.to('cpu')
    yfinal = np.array(y.detach().numpy())
    res_dict['scatterplots'][str(model)]={'targets': tfinal, 
                                          'predictions': yfinal }
    
    print(result)
    #res = pd.DataFrame(res_dict['tables'])
    #display(res)

    #with open("results.txt","a") as f:
    #    #print(dir(dataset))
    #    #f.write("\n")
    #    #print(result)
    #    f.write(result)


def modelSelection(dataset, model_list, res_dict={'tables':{}, 'scatterplots':{}},batch_size=32, k=3):
    start = time.time()

    # 1. training and validation
    kfolds = kFolding(dataset,k)
    #print("test_mask after kfodling(): ", dataset[0].train_mask[:5])
    print("kfolds train after kfodling(): ", kfolds[0][0][:5])
    print("kfolds val after kfodling(): ", kfolds[0][1][:5])
    print("test_mask after kfodling(): ", dataset.test_mask[:5])
    for modeldict in model_list['models']:
        for kfold in kfolds:
            datasetFromKFold(kfold,dataset)
            trainValidate(dataset, modeldict,  res_dict)
        modeldict['cv_val_loss']=modeldict['cv_val_loss']/len(kfolds)


    # 2. Model selection 
    selectBestModel(model_list)

    # 3. final training
    finalTraining(model_list, dataset)


    # 4. Final Testing
    testModel(model_list, dataset, res_dict, start)

    


    

    
def trainTestEvalAllDataset(dataset, epochs=1, batch_size=32, res_dict={}):
    
    """
        version 1:
        Every graph of the dataset is fed into the algorithm at each epoch
        train/test/val nodes are selected among the same graph

        # transductive setting
        train_list, test_list and val_list are Nx disjoint_unions then back to PyTorchGeom
        for each graph: # this can be done on Dataset_preparation_notebook
            get train nodes -> append to train_list (saves graph and node index)
            get test nodes -> append to test_list
            get val nodes -> append to val list
            
        they can be marked as test/train/val beforehand, but then less randomization of the training will exist
        
        for hyperparameter search
            for each epoch:        
                train(train_list)
            validate(val_list)
            save_best_hyperparams()
            
        test(tes_list, best_hyperparams)

        modelPerformance()
            
            
                
        version 2:(inductive setting)
        Every graph of the dataset is fed into the algorithm at each epoch
        all test/val nodes come from graphs different than those of the train nodes
        
        # inductive setting
        for each graph in train: # this can be done on Dataset_preparation_notebook
            get nodes -> append to train_list (saves graph and node index)
            
        for each graphp in test:
            get nodes -> append to test_list
            
        for each graph in val
            get nodes -> append to val list
        
        for hyperparameter search
            for each epoch:        
                train(train_list)
            validate(val_list)
            save_best_hyperparams()
            
        test(tes_list, best_hyperparams)
        
        modelPerformance()
        
    """
    
    global Net
    start = time.time()


    # 1.  prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("using ",device)
    model = Net.to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    
    # 3. train some epochs
    for epoch in range(epochs):
        
        
        # 2.  create a train_mask, and a test_mask (val_mask for further experiments)
        # loader = DataLoader(dataset,  shuffle=False)   # ??
        # for every graph of the dataset
        #   load it, select train/val/ and test nodes
        G = dataset.data
        data = G.to(device)
        #shuffleTrainTestMasks(data)
        #shuffleTrainTestValMasks(data)
        # no longer need for that 20190323
        #shuffleTrainTestMasks(data)


        
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        #if epoch % 25 == 0 :
        #    print("epoch-loss: ",epoch, loss)

    # 4. Model evaluation
    model.eval()
    #  classification in a multiclass setting
    #_, pred = model(data).max(dim=1)
    #correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    #acc = correct / data.test_mask.sum().item()
    #print('Accuracy: {:.4f}'.format(acc))


    # regression 
    pred = model(data)
    #print("target: ",data.y[data.test_mask])
    #print("prediction: ",pred[data.test_mask])
    #print(pred[data.test_mask].type())
    #print(data.y[data.test_mask].type())
    
    # prepare the normalized mean root squared error
    t = data.y[data.test_mask]
    y = pred[data.test_mask]

    negatives = False
    if 0 > (sum(y<0)):
        negatives = True
    rmse, nrmse, normalizer = errors(t,y)
    endtime = time.time()
    
    # getting model hyper params
    hyperparams = ""
    model_ds = [elem+"="+str(getattr(Net, elem)) for elem in dir(Net) if elem.startswith("d") and len(elem)==2]
    hyperparams = "_".join(model_ds)
    theepochs = "epochs="+str(epochs)
    #dataset name
    thedataset = dataset.filename
    # basename
    thedataset = os.path.basename(thedataset)
    # extension
    thedataset = os.path.splitext(thedataset)[0]
    
    result = str(model)+" " \
            +hyperparams+" " \
            +theepochs+" " \
            +str(thedataset)+" " \
            +"rmse="+str(round_half_up(rmse.item(),3))+" " \
            +"nrmse="+str(round_half_up(nrmse.item(),3))+" " \
            +"time="+str(round_half_up(endtime-start,3) )  \
            +" negatives?"+str(negatives) \
            +"\n"
    
    res_dict['tables'][str(model)]={
                            #"model": str(model),
                             "hyperparams": hyperparams,
                             "dataset": str(thedataset),
                             "epochs": epochs,
                             "rmse":round_half_up(rmse.item(),3),
                             "nrmse":round_half_up(nrmse.item(),3),
                             "std":round_half_up(normalizer.item(),3),
                             "time":round_half_up(endtime-start,3),
                            "neg vals": negatives }
    res_dict['scatterplots'][str(model)]={'targets':np.array(t.detach().numpy()), 
                                          'predictions': np.array(y.detach().numpy()) }
    
    with open("results.txt","a") as f:
        #print(dir(dataset))
        #f.write("\n")
        #print(result)
        f.write(result)
    
    del model
    
    

def reporting(res_dict):
    df_original = pd.DataFrame(res_dict['tables'])
    df = df_original.T
    #df.style.set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
    # Assuming that dataframes df1 and df2 are already defined:
    display(df)
    #display(HTML(df.to_html()))
    #print(df)
    
    #now scatter plot for each model (use a grid)
    sp = res_dict['scatterplots']

    # make a grid
    N = len(sp.keys())
    cols = 3
    rows = int(math.ceil(N / cols))

    gs = gridspec.GridSpec(rows, cols)
    fig=plt.figure(figsize=(14, 12), dpi= 60, facecolor='w', edgecolor='k')
    n = 0
    for name,trainres in sp.items():
        
        # subplots
        ax = fig.add_subplot(gs[n])
        n+=1
        
        newt = np.array(trainres['targets'])
        newy = np.array(trainres['predictions'])
        ax.plot(newt, newy,'o', color='black')
        #ax.xlabel('target betweenness')
        #ax.ylabel('betweenness prediction');
        # 45 degree 
        newt = np.append(newt,[0.9,1])
        ax.plot(newt, newt, color = 'red', linewidth = 2)
        # ranges
        #ax.xlim(0, 1)
        #ax.ylim(0, 1)
        # title
        ax.set_title(name)
        
    #fig.suptitle('Scatter plots') # or plt.suptitle('Main title')
    fig.tight_layout()