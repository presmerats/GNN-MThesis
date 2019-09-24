import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import traceback
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class MyNet():

    def __str__(self):
        #print(dir(self))
        #print()
        #print([attr+"-"+str(getattr(self,attr)) for attr in dir(self) if attr.startswith("d")])
        hp = [attr+"="+str(getattr(self,attr)) for attr in dir(self) if (attr.startswith("d") and len(attr)==2) or attr.startswith("num_layers") ]
        hp = "_".join(hp)
        return "%s-%s" % (self.__class__.__name__,hp)
        #return "%s-gcn(%d,%d)-gcn(%d,%d)" % (self.__class__.__name__,dataset.num_features,self.d1,self.d1,
        #                                       dataset.num_classes)
        
        
class Net1(torch.nn.Module, MyNet):
    def __init__(self, d1=16,num_features=1, num_classes=1, **kwargs):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(num_features, d1)
        self.conv2 = GCNConv(d1, num_classes)
        self.d1 = d1
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # output as multiclass target
        #return F.log_softmax(x, dim=1)
        
        # output as regression target
        return x
    


    
class Net2(torch.nn.Module, MyNet):
    def __init__(self, d1=300,d2=100,num_features=1, num_classes=1, **kwargs):
        super(Net2, self).__init__()
        self.conv1 = GCNConv(num_features, d1 )
        #self.conv2 = GCNConv(16, dataset.num_classes)
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_features)
        self.d1 = d1
        self.d2 = d2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 2 fc layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # output as regression target
        return x
        
    
    #def __str__(self):
    #    return "%s-gcn(%d,%d)-fc(%d,%d)-fc(%d,%d)" % (self.__class__.__name__,dataset.num_features,self.d1,self.d1,
    #                                                    self.d2,self.d2,
    #                                                    dataset.num_features)

    
    
class Net3(torch.nn.Module, MyNet):
    def __init__(self, d1=300,d2=100,num_features=1, num_classes=1, **kwargs):
        super(Net3, self).__init__()
        self.conv1 = SAGEConv(num_features, d1 )
        #self.conv2 = SAGEConv(16, dataset.num_classes)
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_features)
        self.d1 = d1
        self.d2 = d2
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 2 fc layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # output as regression target
        return x
        
    def __str__(self):
        return "Net3-SAGEgcn(%d,%d)-fc(%d,%d)-fc(%d,%d)" % (self.num_features,self.d1,self.d1,
                                                        self.d2,self.d2,
                                                        self.num_features)

class Net4(torch.nn.Module, MyNet):
    def __init__(self, d1=300,d2=100,d3=10,num_features=1, num_classes=1, **kwargs):
        super(Net4, self).__init__()
        self.conv1 = GCNConv(num_features, d1 )
        self.conv2 = GCNConv(d1, d2)
        self.fc1 = nn.Linear(d2, d3)
        self.fc2 = nn.Linear(d3, num_features)
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 2 fc layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # output as regression target
        return x
        
    #def __str__(self):
    #    return "Net4-gcn(%d,%d)-fc(%d,%d)-fc(%d,%d)" % (dataset.num_features,self.d1,self.d1,
    #                                                    self.d2,self.d2,
    #                                                    dataset.num_features)

    
class Net5(torch.nn.Module, MyNet):
    def __init__(self, d1=90,d2=80,d3=50,d4=20,num_features=1, num_classes=1, **kwargs):
        super(Net5, self).__init__()
        self.conv1 = GCNConv(num_features, d1 )
        self.conv2 = GCNConv(d1, d2)
        self.fc1 = nn.Linear(d2, d3)
        self.fc2 = nn.Linear(d3, d4)
        self.fc3 = nn.Linear(d4, num_features)
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 3 fc layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # output as regression target
        return x
        
    #def __str__(self):
    #    #print(dir(self))
    #    #print()
    #    #print([attr+"-"+str(getattr(self,attr)) for attr in dir(self)])
    #    return "Net5-gcn(%d,%d)-fc(%d,%d)-fc(%d,%d)" % (dataset.num_features,self.d1,self.d1,
    #                                                   self.d2,self.d2,
    #                                                    dataset.num_features)

    
class Net6(torch.nn.Module, MyNet):
    def __init__(self, d1=90,d2=80,d3=50,num_features=1, num_classes=1, num_layers=4, **kwargs):
        super(Net6, self).__init__()
        self.conv1 = GCNConv(num_features, d1 )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(d1, d1 ))
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, d3)
        self.fc3 = nn.Linear(d3, num_features)
        self.num_layers = num_layers
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        # 3 fc layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # output as regression target
        return x