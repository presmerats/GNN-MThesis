import inspect
from pprint import pprint
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
from pprint import pprint

import networkx as nx
import time
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
import importlib
from torch_geometric.data import Data
import pickle
import numpy as np



class LinkMessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(LinkMessagePassing, self).__init__()
        
        # this calls will get all params of the self.message() func
        # defined in the instance that inherits from this class
        #  ecample def message(self, x_j, norm) -> x_j and norm
        self.message_args = inspect.getargspec(self.message)[0][1:]
        # with GCN this is [x_j, norm]
        self.update_args = inspect.getargspec(self.update)[0][2:]
        # with GCN this is []

    #def propagate(self, aggr, edge_index, edge_neighbors, **kwargs):
    #def propagate(self, aggr, edge_index, **kwargs):
    def propagate(self, aggr, edge_neighbors, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        #kwargs['edge_index'] = edge_index
        kwargs['edge_neighbors'] = edge_neighbors
        #print("self update_args: ", self.update_args)
        
        #print("self message_args: ",self.message_args)
        #print("kwargs: ",kwargs)
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]]) #appends x tensors of the selected nodes in edge list
            elif arg[-2:] == '_l':
                # this branch is for link/edge feature based messages
                """
                   create a edge_neighborhood list
                   this lists says which edge index corresponds to neighbors of the current edge
                   the messages passed will correspond to those of the edge_neighbors
                  
                   some idea would be: use the same, just change messages from links and change
                   x features to correspond to edges, maybe add as xedge
                   
                   so message would be xedges[neighbor_edge_list]
                   
                   norm would be the same           
                   
                   way propagate is called changes from LinkGCNConv
                   message and update parameteres change also
                   
                   maybe: the aggregate could take into account that there's 2 ends of an edge,
                   like aggregating from each side and doing a mean 

                """
                tmp = kwargs[arg[:-2]] # this is the values of the edge betweennesses for each edge
                size = tmp.size(0)
                message_args.append(tmp[edge_neighbors[0]]) #appends x tensors of the selected nodes in edge list
            else:
                message_args.append(kwargs[arg]) # appends full norm tensor
        #print("message_args: ")
        #pprint(message_args)
        update_args = [kwargs[arg] for arg in self.update_args]
        
        #print("update_args: ", update_args)
        out = self.message(*message_args)

        #print("\n normalized messages (edge attribute) ")
        #pprint(out)
        #print("\n edge_neighbor index used ")
        #pprint(edge_neighbors[0])

        
        out = scatter_(aggr, out, edge_neighbors[0], dim_size=size)
        #print("\n scatter output ")
        #pprint(out)

        out = self.update(out, *update_args)

        #print("\n")

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out



class LinkGCNConv(LinkMessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(LinkGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_neighbors, edge_weight=None):
        """
            needs to be changed to have:
            - edge_neighborhoods
            - xedge as features for edges
            - aggregate function 

        """
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_neighbors.size(1), ), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_neighbors.size(1)

        edge_neighbors = add_self_loops(edge_neighbors, num_nodes=x.size(0))
        loop_weight = torch.full(
            (x.size(0), ),
            1 if not self.improved else 2,
            dtype=x.dtype,
            device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_neighbors
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        return self.propagate('add', edge_neighbors, x=x, norm=norm)

    def message(self, x_l, norm):
        return norm.view(-1, 1) * x_l

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class MyOwnDataset2_old():
    def __init__(self,  root, name, transform=None, pre_transform=None):
        f = open(name, 'rb')
        self.data = pickle.load(f) 
        #print(self.data)
        #print(self.data.edge_index)
        #print(self.data.num_features)
        self.num_features = self.data.num_features
        self.num_classes = 1
        self.filename = name
        
        
        # prepare edge_neighbors
        edges_dict = {}
        i=0
        for edge in self.data.edge_index[0]:
            edges_dict[i]= (self.data.edge_index[0][i],
                            self.data.edge_index[1][i])
            i+=1

        #print("\n edges_dict:")
        #pprint(edges_dict)
        #print("\n")
        """
            {
            0 : (1,3),
            1 : (1,4),
            2 : (2,0)
            ...
            }

            edge_neighbors= [[],[]]
            for edge in edges_dict.keys():
                for node in edges_dict[edge]:
                    for edge2 in edges_dict.keys():
                        if edge2 != edge and \
                           ( edges_dict[edge2][0] == node or \
                             edges_dict[edge2][1] == node ):
                             edge_neighbors[0].append(edge)
                             edge_neighbors[1].append(edge2)

        """
        edge_neighbors= [[],[]]
        for edge in edges_dict.keys():
            for node in edges_dict[edge]:
                for edge2 in edges_dict.keys():
                    if edge2 != edge and ( edges_dict[edge2][0] == node or edges_dict[edge2][1] == node ):
                        edge_neighbors[0].append(edge)
                        edge_neighbors[1].append(edge2)

        self.data.edge_neighbors = torch.LongTensor(edge_neighbors)

        #print()
        #print("edge_neighbors")
        #pprint(self.data.edge_neighbors)
        #print()
        #print(type(self.data.edge_neighbors))
        #print()
        
        f.close()
        
class MyOwnDataset2_old2():
    def __init__(self,  root, name, transform=None, pre_transform=None):
        f = open(name, 'rb')
        self.data = pickle.load(f) 

        self.num_features = self.data.num_features
        self.num_classes = 1
        self.filename = name
         
        f.close()


class MyOwnDataset2(InMemoryDataset):
    
    raw_file_name = ''
    train_mask = torch.LongTensor([])
    val_mask = torch.LongTensor([])
    test_mask = torch.LongTensor([])

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.raw_file_name = name
        super(MyOwnDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        print("self.processed_paths",self.processed_paths)
        print("self.root",self.root)
        print("self.raw_dir",self.raw_dir)
        print("self.raw_file_names",self.raw_file_names)
        print("self.raw_paths",self.raw_paths)


    @property
    def raw_file_names(self):
        return [self.raw_file_name]

    @property
    def processed_file_names(self):
        return ['myowndataset2.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # this step is not implemented. You need to manually
        # copy the folders that contain the output of the plugin
        # inside the raw folder
        print("Not implemented. Missing folders with graph files in txt for Nx format, for each program to be included in the dataset")
        

    def process(self):


        print(" MyOwnDataset2: process()")
        print("self.processed_paths",self.processed_paths)
        print("self.root",self.root)
        print("self.raw_dir",self.raw_dir)
        print("self.raw_file_names",self.raw_file_names)
        print("self.raw_paths",self.raw_paths)


        # Read data into huge `Data` list.
        
        f = open(self.raw_file_name, 'rb')
        data = pickle.load(f)
        data.train_mask = torch.LongTensor([])
        data.val_mask = torch.LongTensor([])
        data.test_mask = torch.LongTensor([])
        
        
        data_list = [data]

        print(dir(data))
        print("Finished reading dataset")
        print()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        print("After pre filter-transform, ", len(data_list))
    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
          
    def addEdgeNeighbors(self, dataset):
        """
            transform nodes to edges and edges to nodes?
            In fact, it's just that a new adjacency list is added,
            the edge adjacency list, called edge_neighbors
        """
        edges_dict = {}
        i=0
        for edge in dataset.edge_index[0]:
            edges_dict[i]= (dataset.edge_index[0][i],
                            dataset.edge_index[1][i])
            i+=1

        edge_neighbors= [[],[]]
        for edge in edges_dict.keys():
            for node in edges_dict[edge]:
                for edge2 in edges_dict.keys():
                    if edge2 != edge and ( edges_dict[edge2][0] == node or edges_dict[edge2][1] == node ):
                        edge_neighbors[0].append(edge)
                        edge_neighbors[1].append(edge2)
            
        dataset.edge_neighbors = torch.LongTensor(edge_neighbors)
        return dataset


class MyNet():

    def __str__(self):
        #print(dir(self))
        #print()
        #print([attr+"-"+str(getattr(self,attr)) for attr in dir(self) if attr.startswith("d")])
        hp = [attr+"="+str(getattr(self,attr)) for attr in dir(self) if attr.startswith("d") and len(attr)==2]
        hp = "_".join(hp)
        return "%s-%s" % (self.__class__.__name__,hp)
        #return "%s-gcn(%d,%d)-gcn(%d,%d)" % (self.__class__.__name__,dataset.num_features,self.d1,self.d1,
        #         

class NetLGCN1(torch.nn.Module,MyNet):

    def __init__(self, d1=16,num_features=1, num_classes=1):
        super(NetLGCN1, self).__init__()
        self.conv1 = LinkGCNConv(num_features, d1)
        self.conv2 = LinkGCNConv(d1, num_classes)
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
    

class NetLGCN2(torch.nn.Module, MyNet):
    def __init__(self, d1=16,d2=16,num_features=1, num_classes=1):
        super(NetLGCN2, self).__init__()
        self.conv1 = LinkGCNConv(num_features, d1)
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_features)
        self.d1 = d1
        self.d2 = d2
        

    def forward(self, data):
        x, edge_neighbors = data.x, data.edge_neighbors

        x = self.conv1(x, edge_neighbors)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # output as multiclass target
        #return F.log_softmax(x, dim=1)
        
        # output as regression target
        return x
    
    
class NetLGCN3(torch.nn.Module, MyNet):
    def __init__(self, d1=16,d2=16,d3=10,num_features=1, num_classes=1):
        super(NetLGCN3, self).__init__()
        self.conv1 = LinkGCNConv(num_features, d1)
        self.conv1 = LinkGCNConv(d1, d2)
        self.fc1 = nn.Linear(d2, d3)
        self.fc2 = nn.Linear(d3, num_classes)
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def forward(self, data):
        x, edge_neighbors = data.x, data.edge_neighbors

        x = self.conv1(x, edge_neighbors)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_neighbors)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # output as multiclass target
        #return F.log_softmax(x, dim=1)
        
        # output as regression target
        return x
    

    

class Net1(torch.nn.Module, MyNet):
    def __init__(self, d1=16,num_features=1, num_classes=1):
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
    def __init__(self, d1=300,d2=100,num_features=1, num_classes=1):
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
    def __init__(self, d1=300,d2=100,num_features=1, num_classes=1):
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
    def __init__(self, d1=300,d2=100,d3=10,num_features=1, num_classes=1):
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
    def __init__(self, d1=90,d2=80,d3=50,d4=20,num_features=1, num_classes=1):
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





class META1(torch.nn.Module):
    def __init__(self, d1=3, d2=50, d3=15, d4 =15,d5=10,num_classes=6):
        super(META1, self).__init__()

        self.edge_mlp = Seq(Lin(d1*3, d2), ReLU(), Lin(d2, d3))
        self.node_mlp = Seq(Lin(d1*6, d2), ReLU(), Lin(d2, d3))
        self.global_mlp = Seq(Lin(d3+1, d2), ReLU(), Lin(d2, d3))
        
        self.fc1 = nn.Linear(d4, d5)
        self.dense1_bn = nn.BatchNorm1d(d5)
        self.fc2 = nn.Linear(d5, num_classes)
        self.dense2_bn = nn.BatchNorm1d(num_classes)
        self.global_pool = global_mean_pool

        def edge_model(source, target, edge_attr, u):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            #print("edge_model")
            #print(source.size())
            #print(target.size())
            #print(edge_attr.size())
            out = torch.cat([source, target, edge_attr], dim=1)
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            row, col = edge_index
            
            #print("node_model")
            #print(row.size())
            #print(col.size())
            #print(x[col].size())
            #print(edge_attr.size())
            
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp(out)
            return scatter_mean(out, row, dim=0, dim_size=x.size(0))

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            
            #print("global_Model")
            #print("u.size():")
            #print(u.size())
            #print("scatter_mean(x,batch,..):")
            #smean = scatter_mean(x, batch, dim=0)
            #print(smean.size())
            
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            
            #print("out.size():")
            #print(out.size())
            return self.global_mlp(out)

        #self.op = MetaLayer(edge_model, node_model, global_model)
        self.op = MetaLayer(edge_model, node_model, None)

    def forward(self, data):
        
        x, edge_index, edge_attr,  batch = data.x, data.edge_index, data.edge_attr,  data.batch        
        
        # output of meta is x,edge_attr, u
        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, None, batch)
        
        # idea1 is to cat x2, edge_attr2 and u2?
        # idea2 is to update edge_attr and u...
        data.x = x2
        data.edge_attr = edge_attr2
        #data.u = u2

        # version using only u
        x = F.relu(self.dense1_bn(self.fc1(u2)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
        