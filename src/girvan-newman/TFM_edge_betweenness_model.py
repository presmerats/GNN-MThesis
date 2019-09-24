import inspect
from pprint import pprint
import traceback
import pickle
import networkx as nx
import time
import numpy as np


import torch
from torch_geometric.utils import scatter_
from torch.nn import Parameter
from torch_scatter import scatter_add
#from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.nn.conv import LinkMessagePassing
from torch_geometric.utils import add_self_loops

from torch_geometric.nn.inits import glorot, zeros
#from ..inits import glorot, zeros
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import LinkGCNConv
from torch_geometric.nn import SAGEConv
import torch.nn as nn

from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
import importlib
from torch_geometric.data import Data


from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

from TFM_node_betweenness_training import *

class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        #self.edge_mlp = Seq(Lin(2 * 10 + 5 + 20, 5), ReLU(), Lin(5, 5))
        self.edge_mlp = Seq(Lin(4,10), ReLU(), Lin(10, 19))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        #print(src.shape, dest.shape, edge_attr.shape, u[batch].shape)
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        #self.node_mlp_1 = Seq(Lin(15, 10), ReLU(), Lin(10, 10))
        self.node_mlp_1 = Seq(Lin(20, 10), ReLU(), Lin(10, 19))
        #self.node_mlp_2 = Seq(Lin(2 * 10 + 20, 10), ReLU(), Lin(10, 10))
        self.node_mlp_2 = Seq(Lin(19+2, 10), ReLU(), Lin(10, 19))


    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(20 + 10, 20), ReLU(), Lin(20, 20))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        #return self.global_mlp(out)
        return u


class META1(torch.nn.Module):
    def __init__(self, d1=0, d2=0, d3=0, d4 =19,d5=10,num_edges=6):
        super(META1, self).__init__()

        self.op = MetaLayer(EdgeModel(), NodeModel(),  GlobalModel())
        self.fc1 = nn.Linear(d4, d5)
        self.dense1_bn = nn.BatchNorm1d(d5)
        self.fc2 = nn.Linear(d5, 1) # num_edges is wrong, use 1
        self.dense2_bn = nn.BatchNorm1d(1)
        

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x, edge_index, edge_attr,  batch = data.x, data.edge_index, data.edge_attr,  torch.tensor([0] * len(data.x))        
        #batch must say for each row, which graph it belongs to
        # since only one graph is trained here -> all 0s with length of data.x

        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, torch.ones(1,1).to(device), batch.to(device))
        #x = torch.cat([x2,edge_attr2])
        x = edge_attr2
        #print("after metalayer",x.shape)

        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #print("after fc layers",x.shape)

        return x



class META2(torch.nn.Module):
    def __init__(self, d1=0, d2=0, d3=0, d4 =19,d5=10,num_edges=6):
        super(META2, self).__init__()

        self.op = MetaLayer(EdgeModel(), NodeModel(),  GlobalModel())
        self.fc1 = nn.Linear(d4, d5)
        self.dense1_bn = nn.BatchNorm1d(d5)
        self.fc2 = nn.Linear(d5, 1) # num_edges is wrong, use 1
        self.dense2_bn = nn.BatchNorm1d(1)
        

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x, edge_index, edge_attr,  batch = data.x, data.edge_index, data.edge_attr,  torch.tensor([0] * len(data.x))        
        #batch must say for each row, which graph it belongs to
        # since only one graph is trained here -> all 0s with length of data.x

        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, torch.ones(1,1).to(device), batch.to(device))
        #x = torch.cat([x2,edge_attr2])
        x = x2
        #print("after metalayer",x.shape)

        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #print("after fc layers",x.shape)

        return x
        



def manual_training(dataset,Net,epochs=2,batch_size=1,res_dict={}, seed=16):

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
    try:
    
        #print(G)
        start = time.time()


        # 1.  prepare model--------------------------------------
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        #print("using ",device)
        model = Net.to(device)  
        data = dataset.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        # 2.  create a train_mask, and a test_mask (val_mask for #shuffleTrainTestMasks(data)
        #shuffleTrainTestValMasks(data)
        shuffleTrainTestMasks_analysis(data)

        # 3. train some epochs
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            #print(out.shape)
            #pprint(out)
            #print(data.y.shape)
            #pprint(data.y)
            #out = out.view(out.shape[0])
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            #if epoch % 25 == 0 :
            #    print("epoch-loss: ",epoch, loss)

        # 4. Model evaluation
        model.eval()
        pred = model(data)
        
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
        thedataset = ''
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
                
        #print(nrmse)
        #pprint(res_dict)

        preds = y.to('cpu').detach().numpy().flatten()
        targets = t.to('cpu').detach().numpy().flatten()

        pprint(preds)
        print(preds.shape)
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
            'predictions': preds,
            'targets': targets,
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
        data.to('cpu')
        del model
        torch.cuda.empty_cache()