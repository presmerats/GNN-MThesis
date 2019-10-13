import os
import importlib
import time
import pickle
import traceback
import math
from datetime import datetime
import json
from IPython.display import display, HTML

import pandas as pd
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, QM9, QM7b, PPI, Planetoid, KarateClub
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv


from TFM_node_betweenness_models import Net1,Net2,Net3,Net4,Net5,Net6



      
    
def writeAdjacencyMatrixToDisk(G, filename='temp_adjacency_matrix.txt'):
    """
        Transform to networkx dataset

        possible formats: GML, Adjacency matrix, ..
        start by Adjcency list 
             --> (ignoring edge/node features)
             --> line format: source target target2 target3 ... 
        later we can improve this...
    """
    f = open(filename,'w')
    _ni=-1
    newline = False
    theline = []
    careturn = ""
    for ei in range(G.edge_index.size()[1]):
        print(ei)
        if int(G.edge_index[0,ei].item()) != _ni:
            newline=True
            _ni=int(G.edge_index[0,ei].item())
            
        else:
            newline=False
            
            
        ni = str(G.edge_index[0,ei].item())
        vi = str(G.edge_index[1,ei].item())
        if newline:
            f.write(''.join(theline))
            #print(''.join(theline))
            #print(" --> "+str(_ni))
            theline =[]
            theline.append(careturn+ni+" ")
            theline.append(vi+" ")
            careturn = "\n"
        else:
            theline.append(vi+" ")
        # print("({},{})".format(ni,vi))


def pyTorchGeometricDatasetToNx(G,prefix = './temp/temp_aj_m',suffix=0):
    """
        Alternatives:
            - to disk, to nx, then dict of betweenness
            - transform in memory
            - directly pickle a G object with the betweenness
    """
    # 1. PyTorch Geometric graph -> nx -> compute betweenness 
    #             -> PyTorch Geom with target the betweenness-------
    # Transform to networkx graph
    # write to adjacency matrix on disk
    writeAdjacencyMatrixToDisk(G, filename=prefix+str(suffix)+'.txt')

    # load into a networkx graph object
    g2 = nx.read_adjlist(prefix+str(suffix)+'.txt')
    #g2 = nx_createNxGraphInMem(G)
    
    return g2


def pytorch_geometric_dataset_to_Nx2(data, prefix='./temp/temp_edge_list'):

    with open('temp/temp_to_nx.txt','w') as f:
        edge_neighbors = data.edge_index.tolist()
        for j in range(len(edge_neighbors[0])):
            f.write(str(edge_neighbors[0][j]) + ' ' + str(edge_neighbors[1][j]) + '\n')
                
    
        f.close()

        g = nx.read_edgelist('temp/temp_to_nx.txt', nodetype=int)
        return g

    return None

def inspectGraphDataset(dataset, name):


    dataset_container = dataset
    dataset = dataset.data
    print("Analysis of "+name+" Dataset-----------------------------------")
    try:
        print("number of graphs in the dataset: ",len(dataset))
    except:
        pass
    try:
        print("num classes: ", dataset_container.num_classes)
    except:
        pass

    try:
        print("num features: ", dataset_container.num_features)
    except:
        pass

    print(type(dataset))
    if isinstance(dataset,Data):
        #print("it's a Data type")
        data = dataset
    else:
        #print("it's a Dataset type")
        print("\ninspecting first graph in dataset: ")
        data = dataset[0]
        print(data)
        print("data.is_undirected(): ",data.is_undirected())
        try:
            print("num features: ", data.num_features)
        except:
            pass

        print("\ninspecting some random graph in dataset: ")
        dataset = dataset.shuffle()
        data = dataset[0]
        print(data)
        # random permutation
        # dataset = dataset.shuffle()
        # perm = torch.randperm(len(dataset))
        # dataset = dataset[perm]
        print("\n",dir(data))
        print("\n Showing some target values")
        print("num edges: ", data.num_edges)
        print("num nodes: ", data.num_nodes)
        print("contains_isolated_nodes: ", data.contains_isolated_nodes())
        print("contains_self_loops: ", data.contains_self_loops())
        print("data.is_undirected(): ",data.is_undirected())
        print("data.is_directed(): ",data.is_directed())
        print("edge_attributes: ",data.edge_attr)
        try:
            print("node features: ",data.x[0])
        except:
            pass
        print("target values: ", data.y[0])
        print("edge_index: ",data.edge_index)
        print()

    
    
    # transform to NX
    g = pyTorchGeometricDatasetToNx(data)
    #print(dir(g))

    # visualize
    print("\nDraw")
    if data.num_nodes < 1000:
        print(data.y[0].item())
        print(data.y[0])
        try:
            if isinstance(data.y[0].item(), int):

                attrs={}
                palette = ['yellow', 'cyan','orange','red','magenta', 'silver','grey','blue','green',   'pink','black',]


                for node in g.nodes():
                    #print(palette[color % len(palette)])
                    print(int(node))
                    print(data.y)
                    print(data.y[int(node)])
                    print(palette)
                    attrs[node]={'color':palette[data.y[int(node)] % len(palette)]}
                nx.set_node_attributes(g, attrs)
                colors = nx.get_node_attributes(g, 'color')
                nx.draw(g, node_color=colors.values())
                plt.draw()
            else:
                nx.draw(g)
                plt.draw()
        except:
            # for graph classification tasks
            nx.draw(g)
            plt.draw()
                
    else:
        print("-->too big to draw! Slicing first 300 nodes..")
        nodelist = list([ n[0] for n in g.edges()])[:300]
        #print(list(range(300)))
        g2 = g.subgraph(nodelist)
        print(g2.edges())
        nx.draw(g2)
        plt.draw()
    #print("Draw circular")
    #nx.draw_circular(g)
    #plt.draw()
    #print("Draw random")
    #nx.draw_random(g)
    #plt.draw()


#------------------__DATASET-------------------------------------

class MyOwnDataset2():
    def __init__(self,  root, name, transform=None, pre_transform=None):
        f = open(name, 'rb')
        self.data = pickle.load(f) 
        #print(self.data.num_features)
        self.num_features = self.data.num_features
        self.num_classes = 1
        f.close()
        
        



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
 




#------------------TRAINING---------------------------------------

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def transformMask(mask):
    train_mask = []
    i = 0
    for pick in mask:
        if pick[0]==1:
            train_mask.append(i)
        i+=1
    return train_mask


def shuffleTrainTestMasks(data, trainpct = 0.7):
    ysize = list(data.y.size())[0]
    data.train_mask = torch.zeros(ysize,1, dtype=torch.long)
    data.train_mask[int(ysize*trainpct):] = 1
    data.train_mask = data.train_mask[torch.randperm(ysize)]
    data.test_mask = torch.ones(ysize,1, dtype=torch.long) - data.train_mask
    
    data.train_mask = transformMask(data.train_mask)
    data.test_mask = transformMask(data.test_mask)
    
    #print(data.train_mask)
    #print(data.test_mask)

    # print min,mean, median, max, sd of each split
    ytrains = data.y[data.train_mask].cpu()
    ytests =  data.y[data.test_mask].cpu()



def shuffleTrainTestMasks_analysis(data, trainpct = 0.5):
    ysize = list(data.y.size())[0]
    data.train_mask = torch.zeros(ysize,1, dtype=torch.long)
    data.train_mask[int(ysize*trainpct):] = 1
    data.train_mask = data.train_mask[torch.randperm(ysize)]
    data.test_mask = torch.ones(ysize,1, dtype=torch.long) - data.train_mask
    
    data.train_mask = transformMask(data.train_mask)
    data.test_mask = transformMask(data.test_mask)
    
    #print(data.train_mask)
    #print(data.test_mask)

    # print min,mean, median, max, sd of each split
    ytrains = data.y[data.train_mask].cpu()
    ytests =  data.y[data.test_mask].cpu()

    # print(ytrains.shape)
    # print(ytests.shape)

    ytr_min = torch.min(ytrains).item()
    ytr_max = torch.max(ytrains).item()
    ytr_mean = torch.mean(ytrains).item()
    ytr_median = torch.median(ytrains).item()

    yt_min = torch.min(ytests).item()
    yt_max = torch.max(ytests).item()
    yt_mean = torch.mean(ytests).item()
    yt_median = torch.median(ytests).item()

    # print("train split: ",ytr_min,ytr_mean,ytr_median,ytr_max)
    # print("test split: ",yt_min,yt_mean,yt_median,yt_max)

    plt.boxplot([ytrains.tolist(),ytests.tolist()])
    plt.show()

    # sort then 
    ytrains,_ = torch.sort(ytrains)
    ytests,_ = torch.sort(ytests)

    nrmse = torch.sum((ytrains - ytests) ** 2)/len(data.test_mask)
    nrmse = nrmse.sqrt()
    #print("rmse: ",nrmse.item())

    return nrmse


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

    
    #print(data.train_mask)
    #print(data.val_mask)
    #print(data.test_mask)
    
    
def seedAnalysis(Net,dataset, epochs=1, batch_size=32, res_dict={}, seed=None):
    
    

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    loader = DataLoader(dataset,  shuffle=False)
    G = dataset.data
    rmse =shuffleTrainTestMasks_analysis(G)
    return seed, rmse

def trainTestEval(dataset, epochs=1, batch_size=32, res_dict={}, seed=None):
    global Net

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    loader = DataLoader(dataset,  shuffle=False)
    i = 0
    #print(loader)
    #print(dir(loader))
    
    G = dataset.data
    try:
    
        #print(G)
        start = time.time()


        # 1.  prepare model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("using ",device)
        model = Net.to(device)  
        data = G.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        # 2.  create a train_mask, and a test_mask (val_mask for further experiments)
        #shuffleTrainTestMasks(data)
        #shuffleTrainTestValMasks(data)
        shuffleTrainTestMasks_analysis(data)

        # 3. train some epochs
        for epoch in range(epochs):
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
        nrmse = torch.sum((t - y) ** 2)/len(data.test_mask)
        nrmse = nrmse.sqrt()
        #print("RMSE: ",nrmse)

        m = torch.mean(t).item()
        m2 = torch.mean(y).item()
        med = torch.median(t).item()
        med2 = torch.median(y).item()
        #print("mean",m)
        tmax = torch.max(t).item()
        tmax2 = torch.max(y).item()
        tmin = torch.min(t).item()
        tmin2 = torch.min(y).item()
        sd = tmax-tmin
        sd = sd
        sd2 = tmax2-tmin2
        sd2 = sd2
        #print("sd",sd)
        #nrmse = (nrmse - m)/sd
        #print("NRMSE:",nrmse)


        endtime = time.time()
        #print("Total train-test time: "+str(endtime-start))

        # getting model hyper params
        hyperparams = ""
        
        model_ds = [elem+"="+str(getattr(Net, elem)) for elem in dir(Net) if elem.startswith("epochs") or elem.startswith("num_layers") or (elem.startswith("d") and len(elem)==2)]
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
                +"nrmse="+str(round_half_up(nrmse.item(),3))+" " \
                +"time="+str(round_half_up(endtime-start,3) )  \
                +" negatives?"+str(negatives) \
                +"\n"
                
        res_dict['tables'][str(model)]={
                                #"model": str(model),
                                 "hyperparams": hyperparams,
                                 "dataset": str(thedataset),
                                 "epochs": epochs,
                                 "nrmse":round_half_up(nrmse.item(),3),
                                 "time":round_half_up(endtime-start,3),
                                #"neg vals": negatives,
                                "GTavg": m,
                                "PREDavg": m2,
                                #"GTmed": med,
                                #"PREDmed": med2,
                                "GTmin": tmin,
                                "PREDmin": tmin2,
                                "GTmax": tmax,
                                "PREDmax": tmax2,
                                #"GTsd": sd,
                                #"PREDsd": sd2,

                                }
        res_dict['scatterplots'][str(model)]={
            'predictions': y.to('cpu').detach().numpy(),
            'targets': t.to('cpu').detach().numpy(),
        }

        with open("results.txt","a") as f:
            #print(dir(dataset))
            #f.write("\n")
            #print(result)
            f.write(result)

        model.to('cpu')  
        data.to('cpu')
        del data
        del model
        torch.cuda.empty_cache()


        #i+=1
        #if i==1:
        #    break
    except Exception as err:
        traceback.print_exc()
        model.to('cpu')  
        G.to('cpu')
        del G
        del model
        torch.cuda.empty_cache()



def trainTestEval2(Net,dataset, epochs=1, batch_size=32, res_dict={}, seed=None):
    

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        

    loader = DataLoader(dataset,  shuffle=False)
    i = 0
    #print(loader)
    #print(dir(loader))
    
    G = dataset.data
    try:
    
        #print(G)
        start = time.time()


        # 1.  prepare model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("using ",device)
        model = Net.to(device)  
        data = G.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        # 2.  create a train_mask, and a test_mask (val_mask for further experiments)
        #shuffleTrainTestMasks(data)
        #shuffleTrainTestValMasks(data)
        shuffleTrainTestMasks_analysis(data)

        # 3. train some epochs
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            out = out.view(out.shape[0])
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
        nrmse = torch.sum((t - y) ** 2)/len(data.test_mask)
        nrmse = nrmse.sqrt()
        #print("RMSE: ",nrmse)

        m = torch.mean(t).item()
        m2 = torch.mean(y).item()
        med = torch.median(t).item()
        med2 = torch.median(y).item()
        #print("mean",m)
        tmax = torch.max(t).item()
        tmax2 = torch.max(y).item()
        tmin = torch.min(t).item()
        tmin2 = torch.min(y).item()
        sd = tmax-tmin
        sd = sd
        sd2 = tmax2-tmin2
        sd2 = sd2
        #print("sd",sd)
        #nrmse = (nrmse - m)/sd
        #print("NRMSE:",nrmse)


        endtime = time.time()
        #print("Total train-test time: "+str(endtime-start))

        # getting model hyper params
        hyperparams = ""
        
        model_ds = [elem+"="+str(getattr(Net, elem)) for elem in dir(Net) if elem.startswith("epochs") or elem.startswith("num_layers") or (elem.startswith("d") and len(elem)==2)]
        hyperparams = "_".join(model_ds)
        theepochs = "epochs="+str(epochs)
        hyperparams = hyperparams+"_"+theepochs
        # previously used str(model)
        #model_name = str(model)
        model_name = model.__class__.__name__+"-"+hyperparams
        #dataset name
        thedataset = dataset.filename
        # basename
        thedataset = os.path.basename(thedataset)
        # extension
        thedataset = os.path.splitext(thedataset)[0]

        # result = str(model)+" " \
        #         +hyperparams+" " \
        #         +theepochs+" " \
        #         +str(thedataset)+" " \
        #         +"nrmse="+str(round_half_up(nrmse.item(),3))+" " \
        #         +"time="+str(round_half_up(endtime-start,3) )  \
        #         +" negatives?"+str(negatives) \
        #         +"\n"
                
        
        res_dict['tables'][model_name]={
                                #"model": str(model),
                                 "hyperparams": hyperparams,
                                 "dataset": str(thedataset),
                                 "epochs": epochs,
                                 "batch_size": batch_size,
                                 "seed": seed,
                                 "nrmse":round_half_up(nrmse.item(),3),
                                 "time":round_half_up(endtime-start,3),
                                #"neg vals": negatives,
                                "GTavg": m,
                                "PREDavg": m2,
                                #"GTmed": med,
                                #"PREDmed": med2,
                                "GTmin": tmin,
                                "PREDmin": tmin2,
                                "GTmax": tmax,
                                "PREDmax": tmax2,
                                #"GTsd": sd,
                                #"PREDsd": sd2,

                                }
        res_dict['scatterplots'][model_name]={
            'predictions': y.to('cpu').detach().numpy(),
            'targets': t.to('cpu').detach().numpy(),
        }


        datetime_str =  datetime.today().strftime('%Y-%m-%d_%H-%M-%S') 

        with open("results/"+Net.__class__.__name__+"_"+datetime_str+".json","w") as f:
            json.dump(res_dict['tables'],f)

        model.to('cpu')  
        data.to('cpu')
        del data
        del model
        torch.cuda.empty_cache()


        #i+=1
        #if i==1:
        #    break
    except Exception as err:
        traceback.print_exc()
        model.to('cpu')  
        G.to('cpu')
        del G
        del model
        torch.cuda.empty_cache()


def trainTestEval3(Net,dataset, epochs=1, batch_size=32, res_dict={}, seed=None):
    

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    loader = DataLoader(dataset,  shuffle=False)
    i = 0
    #print(loader)
    #print(dir(loader))
    
    G = dataset.data
    try:
    
        #print(G)
        start = time.time()


        # 1.  prepare model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("using ",device)
        model = Net.to(device)  
        data = G.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        # 2.  create a train_mask, and a test_mask (val_mask for further experiments)
        #shuffleTrainTestMasks(data)
        #shuffleTrainTestValMasks(data)
        shuffleTrainTestMasks(data)

        # 3. train some epochs
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            out = out.view(out.shape[0])
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
        nrmse = torch.sum((t - y) ** 2)/len(data.test_mask)
        nrmse = nrmse.sqrt()
        #print("RMSE: ",nrmse)

        m = torch.mean(t).item()
        m2 = torch.mean(y).item()
        med = torch.median(t).item()
        med2 = torch.median(y).item()
        #print("mean",m)
        tmax = torch.max(t).item()
        tmax2 = torch.max(y).item()
        tmin = torch.min(t).item()
        tmin2 = torch.min(y).item()
        sd = tmax-tmin
        sd = sd
        sd2 = tmax2-tmin2
        sd2 = sd2
        #print("sd",sd)
        #nrmse = (nrmse - m)/sd
        #print("NRMSE:",nrmse)


        endtime = time.time()
        #print("Total train-test time: "+str(endtime-start))

        # repeat 3 tiems and get the average?
        # or fix the split


        # getting model hyper params
        hyperparams = ""
        
        model_ds = [elem+"="+str(getattr(Net, elem)) for elem in dir(Net) if elem.startswith("epochs") or elem.startswith("num_layers") or (elem.startswith("d") and len(elem)==2)]
        hyperparams = "_".join(model_ds)
        theepochs = "epochs="+str(epochs)
        hyperparams = hyperparams+"_"+theepochs
        # previously used str(model)
        #model_name = str(model)
        model_name = model.__class__.__name__+"-"+hyperparams
        #dataset name
        thedataset = dataset.filename
        # basename
        thedataset = os.path.basename(thedataset)
        # extension
        thedataset = os.path.splitext(thedataset)[0]

        # result = str(model)+" " \
        #         +hyperparams+" " \
        #         +theepochs+" " \
        #         +str(thedataset)+" " \
        #         +"nrmse="+str(round_half_up(nrmse.item(),3))+" " \
        #         +"time="+str(round_half_up(endtime-start,3) )  \
        #         +" negatives?"+str(negatives) \
        #         +"\n"
                
        
        res_dict['tables'][model_name]={
                                #"model": str(model),
                                 "hyperparams": hyperparams,
                                 "dataset": str(thedataset),
                                 "epochs": epochs,
                                 "batch_size": batch_size,
                                 "seed": seed,
                                 "nrmse":round_half_up(nrmse.item(),3),
                                 "time":round_half_up(endtime-start,3),
                                #"neg vals": negatives,
                                "GTavg": m,
                                "PREDavg": m2,
                                #"GTmed": med,
                                #"PREDmed": med2,
                                "GTmin": tmin,
                                "PREDmin": tmin2,
                                "GTmax": tmax,
                                "PREDmax": tmax2,
                                #"GTsd": sd,
                                #"PREDsd": sd2,

                                }
        res_dict['scatterplots'][model_name]={
            'predictions': y.to('cpu').detach().numpy(),
            'targets': t.to('cpu').detach().numpy(),
        }


        datetime_str =  datetime.today().strftime('%Y-%m-%d_%H-%M-%S') 

        with open("results/"+Net.__class__.__name__+"_"+datetime_str+".json","w") as f:
            json.dump(res_dict['tables'],f)

        model.to('cpu')  
        data.to('cpu')
        del data
        del model
        torch.cuda.empty_cache()


        #i+=1
        #if i==1:
        #    break
    except Exception as err:
        traceback.print_exc()
        model.to('cpu')  
        G.to('cpu')
        del G
        del model
        torch.cuda.empty_cache()


def reporting_from_csv(filepath,label='results',title='Girvan Newman experiments results'):

    pd.set_option("display.max_colwidth", 10000)
    results = pd.read_csv(filepath, names=['Model','Parameters','Loss','Accuracy','Time(min)'])
    latex_str = results.to_latex(index=False)
    

    #latex_str = latex_str.replace('\\\\','\\')
    latex_str = latex_str.replace("llrlr",'|llllccc|')
    latex_str = latex_str.replace("\\toprule",'\\hline')
    latex_str = latex_str.replace("\\midrule",'\\hline')
    latex_str = latex_str.replace("\\bottomrule",'\\hline')
    latex_str = "\\begin{table}[H]\n\\centering\n"+latex_str
    caption = "\\label{"+label+"}\\caption{"+title+"}\n\\end{table}"
    latex_str = latex_str + caption

    latex_str = latex_str.replace('d4=','d')
    latex_str = latex_str.replace('\_d5=','d')
    latex_str = latex_str.replace('\_hus=','h')
    latex_str = latex_str.replace('\_eus=','e')
    latex_str = latex_str.replace('\_n1us=','n')
    latex_str = latex_str.replace('\_n2us=','n')
    latex_str = latex_str.replace('\_r=','r')
    latex_str = latex_str.replace('\_epochs=','epochs')
    latex_str = latex_str.replace('\_split-','\_')


    latex_str = latex_str.replace("5r","5 & ")
    latex_str = latex_str.replace("0r","0 & ")
    latex_str = latex_str.replace("epochs","-")
    latex_str = latex_str.replace("\_"," & ")
    latex_str = latex_str.replace("Accu & racy",'Accuracy')
    latex_str = latex_str.replace("Pa & ramete & rs",'Parameters')
    latex_str = latex_str.replace("Parameters",'Paramteres  &  Runs\/Epochs  &  Splits')

    display(results)

    print(latex_str)


    return latex_str

def reporting_from_csv_f1macro(filepath,label='results',title='Girvan Newman experiments results'):

    pd.set_option("display.max_colwidth", 10000)
    results = pd.read_csv(filepath, names=['Model','Parameters','Loss','Accuracy','F1-macro','Time(min)'])
    latex_str = results.to_latex(index=False)
    

    #latex_str = latex_str.replace('\\\\','\\')
    latex_str = latex_str.replace("llrlr",'|llllccc|')
    latex_str = latex_str.replace("\\toprule",'\\hline')
    latex_str = latex_str.replace("\\midrule",'\\hline')
    latex_str = latex_str.replace("\\bottomrule",'\\hline')
    latex_str = "\\begin{table}[H]\n\\centering\n"+latex_str
    caption = "\\label{"+label+"}\\caption{"+title+"}\n\\end{table}"
    latex_str = latex_str + caption

    latex_str = latex_str.replace('d4=','d')
    latex_str = latex_str.replace('\_d5=','d')
    latex_str = latex_str.replace('\_hus=','h')
    latex_str = latex_str.replace('\_eus=','e')
    latex_str = latex_str.replace('\_n1us=','n')
    latex_str = latex_str.replace('\_n2us=','n')
    latex_str = latex_str.replace('\_r=','r')
    latex_str = latex_str.replace('\_epochs=','epochs')
    latex_str = latex_str.replace('\_split-','\_')


    latex_str = latex_str.replace("5r","5 & ")
    latex_str = latex_str.replace("0r","0 & ")
    latex_str = latex_str.replace("epochs","-")
    latex_str = latex_str.replace("\_"," & ")
    latex_str = latex_str.replace("Accu & racy",'Accuracy')
    latex_str = latex_str.replace("Pa & ramete & rs",'Parameters')
    latex_str = latex_str.replace("Parameters",'Paramteres  &  Runs\/Epochs  &  Splits')

    display(results)

    print(latex_str)


    return latex_str

def reporting_simple(d):
    df_original = pd.DataFrame(d['tables'])
    df = df_original.T
    #df.style.set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
    # Assuming that dataframes df1 and df2 are already defined:
    display(df)
    #display(HTML(df.to_html()))
    #print(df)


def reporting(res_dict):
    df_original = pd.DataFrame(res_dict['tables'])
    df = df_original.T

    plt.plot(
        df['epochs'],
        df['nrmse'],
        )
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.title('validation performance by epoch')
    plt.show()

    #df.style.set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
    # Assuming that dataframes df1 and df2 are already defined:
    display(df)
    #display(HTML(df.to_html()))
    #print(df)
    
    #now scatter plot for each model (use a grid)
    sp = res_dict['scatterplots']
    tbs = res_dict['tables']
    
    # find best model 
    min_nrmse=1000
    best_model = ''
    for name,trainres in tbs.items():
        if trainres['nrmse']<min_nrmse:
            min_nrmse=trainres['nrmse']
            best_model=name
            
            
    # make a grid
    #N = len(sp.keys())
    #cols = 3
    #rows = int(math.ceil(N / cols))

    #gs = gridspec.GridSpec(rows, cols)
    gs = gridspec.GridSpec(1, 2) # scatter and 2 boxplots
    fig=plt.figure(figsize=(16, 8), dpi= 60, facecolor='w', edgecolor='k')
    n = 0



    for name,trainres in sp.items():
        
        if name != best_model:
            continue
        
        # subplots - scatterplot
        ax = fig.add_subplot(gs[n])
        n+=1
        
        newt = np.array(trainres['targets'])
        newy = np.array(trainres['predictions'])
        
        ax.plot(newt, newy,'o', color='black')
        plt.xlabel('target')
        plt.ylabel('prediction');
        ax.plot(newt, newt, color = 'red', linewidth = 2)
        # ranges
        #ax.xlim(0, 1)
        #ax.ylim(0, 1)
        # title
        ax.set_title(name)
        
        # subplots - boxplot
        ax = fig.add_subplot(gs[n])
        n+=1
        newy.shape = newt.shape
        ax.boxplot([newt,newy])
        ax.set_title(name)
        

        
    #fig.suptitle('Scatter plots') # or plt.suptitle('Main title')
    fig.tight_layout()


#------------------TRAINING-INSTANCES-------------------------

def basic_training(training_dict, res_dict ,visualize=False):
    """ 
        Put all params inside a dict
        so it's easier to do HP search
        and easier to save characteristics of models on disk



    """

    # epochs = training_dict['epochs']
    # batch_size = training_dict['batch_size']
    # seed = training_dict['seed']
    # dataset_name = training_dict['dataset_name']
    epochs = training_dict.pop('epochs')
    batch_size = training_dict.pop('batch_size')
    seed = training_dict.pop('seed')
    dataset_name = training_dict.pop('dataset_name')


    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dataset_name)
    if visualize: 
        inspectGraphDataset(dataset, 'vis_'+dataset_name)
    model_class=training_dict.pop('class')
    kwargs = training_dict
    kwargs.update({'num_features': dataset.num_features,
                   'num_classes': dataset.num_classes,
                   })
    #Net=Net6(d1=15,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    Net=model_class(**kwargs)
    trainTestEval2(Net, dataset,  epochs=epochs, batch_size=batch_size, res_dict=res_dict, seed=seed)
    
    del Net
    del dataset
    torch.cuda.empty_cache()


def basic_analysis(training_dict, res_dict ,visualize=False):
    """ 
        Put all params inside a dict
        so it's easier to do HP search
        and easier to save characteristics of models on disk



    """

    # epochs = training_dict['epochs']
    # batch_size = training_dict['batch_size']
    # seed = training_dict['seed']
    # dataset_name = training_dict['dataset_name']
    epochs = training_dict.pop('epochs')
    batch_size = training_dict.pop('batch_size')
    seed = training_dict.pop('seed')
    dataset_name = training_dict.pop('dataset_name')


    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dataset_name)
    if visualize: 
        inspectGraphDataset(dataset, 'vis_'+dataset_name)
    model_class=training_dict.pop('class')
    kwargs = training_dict
    kwargs.update({'num_features': dataset.num_features,
                   'num_classes': dataset.num_classes,
                   })
    #Net=Net6(d1=15,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    Net=model_class(**kwargs)
    s,e = seedAnalysis(Net, dataset,  epochs=epochs, batch_size=batch_size, res_dict=res_dict, seed=seed)
    res_dict['tables'][s]=e.item()
    del Net
    del dataset
    torch.cuda.empty_cache()



def trainingExplosion(dname, res_dict):

    training_dicts =[]
    for model in [Net6]:
        for d1 in [15]:
            for d2 in [50]:
                for d3 in [20]:
                    for d4 in [20]:
                        for num_layers in [2,4,8,10]:
                            for epochs in [2,20]:
                                training_dicts.append(

                                    {
                                    # model params
                                    'class': model,
                                    'd1': d1,
                                    'd2': d2,
                                    'd3': d3,
                                    'd4': d4,
                                    'num_layers': num_layers,
                                    # now training params
                                    'dataset_name': dname,
                                    'epochs': epochs,
                                    'batch_size': 8,
                                    'seed': 35,
                                    }
                                    )

    for td in training_dicts:
        basic_training(
        td,
        res_dict,
        False
        )

def experimentBlock_new(dname, res_dict, visualize=False):

    basic_training({
        # model params
        'class': Net1,
        'd1': 20,
        # now training params
        'dataset_name': dname,
        'epochs': 20,
        'batch_size': 32,
        'seed': None,
        },
        res_dict,
        visualize
        )

    
    
def experimentBlock_quick(dname, res_dict, visualize=False):
    basic_training({
        # model params
        'class': Net6,
        'd1': 20,
        'd2': 15,
        'd3': 50,
        'd4': 20,
        'num_layers': 2,
        # now training params
        'dataset_name': dname,
        'epochs': 2,
        'batch_size': 8,
        'seed': 35,
        },
        res_dict,
        visualize
        )

def experimentBlock_long1(dname, res_dict, visualize=False):
    basic_training({
        # model params
        'class': Net2,
        'd1': 50,
        'd2': 15,
        'num_layers': 2,
        # now training params
        'dataset_name': dname,
        'epochs': 100,
        'batch_size': 16,
        'seed': 35,
        },
        res_dict,
        visualize
    )

    basic_training({
        # model params
        'class': Net6,
        'd1': 15,
        'd2': 50,
        'd3': 20,
        'num_layers': 2,
        # now training params
        'dataset_name': dname,
        'epochs': 100,
        'batch_size': 16,
        'seed': 35,
        },
        res_dict,
        visualize
    )

    basic_training({
        # model params
        'class': Net6,
        'd1': 15,
        'd2': 50,
        'd3': 20,
        'num_layers': 8,
        # now training params
        'dataset_name': dname,
        'epochs': 100,
        'batch_size': 16,
        'seed': 35,
        },
        res_dict,
        visualize
    )


def experimentBlock_old(dname):
    global Net
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net1(d1=20, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net1(d1=50, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net1(d1=100, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net2(d1=100,d2=15, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net2(d1=50,d2=15, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net2(d1=20,d2=10, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net3(d1=50,d2=10, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net4(d1=80,d2=40,d3=10, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net5(d1=100,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=100,d2=50,d3=20, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=50,d2=50,d3=20, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=200,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()



def experimentBlock(dname, visualize=False):
    global Net
    
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    if visualize: inspectGraphDataset(dataset, 'vis_'+dname)
    Net=Net2(d1=50,d2=15, num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()

   
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=15,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=25,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=50,d2=50,d3=20,num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=15,d2=50,d3=20, 
             num_layers=8,
             num_features=dataset.num_features, 
             num_classes=dataset.num_classes,
            )
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=25,d2=50,d3=20, 
             num_layers=8,
             num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=50,d2=50,d3=20,
             num_layers=8,
             num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=15,d2=50,d3=20, 
             num_layers=10,
             num_features=dataset.num_features, 
             num_classes=dataset.num_classes,
            )
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=25,d2=50,d3=20,
             num_layers=10,
             num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=50,d2=50,d3=20, 
             num_layers=10,
             num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=100, res_dict=res_dict)
    del Net
    del dataset
    torch.cuda.empty_cache()


def experimentBlock_quick_old(dname, epochs=2, seed=35,nlayers=2,d1=15,d2=50,d3=20,d4=50):
    global Net
    
    
    dataset = loadDataset(
        collection='MyOwnDataset2', 
        name=dname)
    Net=Net6(d1=d1,d2=d2,d3=d3,
             num_layers=nlayers,
             num_features=dataset.num_features, num_classes=dataset.num_classes)
    trainTestEval(dataset,  epochs=epochs, batch_size=8, res_dict=res_dict, seed=seed)
    del Net
    del dataset
    torch.cuda.empty_cache()
    


