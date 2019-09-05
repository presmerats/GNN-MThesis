import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.inits import uniform
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
#from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
#from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv



#from TFM_function_renaming import *



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
            
            #print("ggnn version, m=",m," edge_index", edge_index)
            #print(" m.size()=",m.size(), " edge_index.size()=", edge_index.size())
            #print(" m.max()=",m.max(), " edge_index.max()=", edge_index.max())
            m = self.propagate('add',edge_index, x=m)
            h = self.rnn(m, h)

        return h


    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)




class GGNN1(torch.nn.Module):
    def __init__(self, d1=50,d2=20,num_classes=6, num_layers=2, aggr_type='mean'):
        super(GGNN1, self).__init__()
        self.ggnn = GatedGraphConv(out_channels=d1, num_layers=num_layers,aggr=aggr_type, bias=True)
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, num_classes)
        self.global_pool = global_mean_pool
        
        

    def forward(self, data):
        x, edge_index, batch_vector = data.x, data.edge_index, data.batch

        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # until here the output is for each node
        
        x = self.global_pool(x, batch_vector) # this makes the output to be graph level?
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        #x = self.pool1(x, batch )
        x = F.log_softmax(x, dim=1)
        #x = torch.argmax(x, dim=1)  # we output softmax to use the nll_loss
        
        return x
    
class GGNN2(torch.nn.Module):
    def __init__(self, d1=50,d2=20,num_classes=6, num_layers=2, aggr_type='mean'):
        super(GGNN2, self).__init__()
        self.ggnn = GatedGraphConv(out_channels=d1, num_layers=num_layers,aggr=aggr_type, bias=True)
        self.fc1 = nn.Linear(d1, num_classes)
        self.global_pool = global_mean_pool
        
    def forward(self, data):
        x, edge_index, batch_vector = data.x, data.edge_index, data.batch

        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # until here the output is for each node
        
        x = self.global_pool(x, batch_vector) # this makes the output to be graph level?
        
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
class GGNN3(torch.nn.Module):
    def __init__(self, d1=50,d2=20, d3=10,num_classes=6, num_layers=2, aggr_type='mean'):
        super(GGNN3, self).__init__()
        self.ggnn = GatedGraphConv(out_channels=d1, num_layers=num_layers,aggr=aggr_type, bias=True)
        self.fc1 = nn.Linear(d1, d2)
        self.dense1_bn = nn.BatchNorm1d(d2)
        self.fc2 = nn.Linear(d2, d3)
        self.dense2_bn = nn.BatchNorm1d(d3)
        self.fc3 = nn.Linear(d3, num_classes)
        self.global_pool = global_mean_pool
        
    def forward(self, data):
        x, edge_index, batch_vector = data.x, data.edge_index, data.batch

        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # until here the output is for each node
        
        x = self.global_pool(x, batch_vector) # this makes the output to be graph level?
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = F.relu(self.dense2_bn(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class GGNN4(torch.nn.Module):
    def __init__(self, d1=50,d2=20,num_classes=6, num_layers=2, aggr_type='mean'):
        super(GGNN4, self).__init__()
        self.ggnn = GatedGraphConv(out_channels=d1, num_layers=num_layers,aggr=aggr_type, bias=True)
        self.fc1 = nn.Linear(d1, d2)
        self.dense1_bn = nn.BatchNorm1d(d2)
        self.fc2 = nn.Linear(d2, num_classes)
        self.global_pool = global_mean_pool
        
        
    def forward(self, data):
        x, edge_index, batch_vector = data.x, data.edge_index, data.batch

        print(edge_index)
        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # until here the output is for each node
        
        x = self.global_pool(x, batch_vector) # this makes the output to be graph level?
        #x = self.fc1(x)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        #x = self.pool1(x, batch )
        x = F.log_softmax(x, dim=1)
        #x = torch.argmax(x, dim=1)  # we output softmax to use the nll_loss
        
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

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, data):
        
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch        
        
        # output of meta is x,edge_attr, u
        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, u, batch)
        
        # idea1 is to cat x2, edge_attr2 and u2?
        # idea2 is to update edge_attr and u...
        data.x = x2
        data.edge_attr = edge_attr2
        data.u = u2

        # version using only u
        x = F.relu(self.dense1_bn(self.fc1(u2)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
        
    
    
class META2(torch.nn.Module):
    def __init__(self, d1=3, d2=50, d3=15, d4 =15,d5=10,num_classes=6):
        super(META2, self).__init__()

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

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, data):
        
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch        
        
        # output of meta is x,edge_attr, u
        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, u, batch)
        
        # idea1 is to cat x2, edge_attr2 and u2?
        # idea2 is to update edge_attr and u...
        data.x = x2
        data.edge_attr = edge_attr2
        data.u = u2
        
        # version using only x 
        x = self.global_pool(x2,batch) # separate by graph level
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

    
class META3(torch.nn.Module):
    def __init__(self, d1=3, d2=50, d3=15, d4 =15,d5=10,num_classes=6):
        super(META3, self).__init__()

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

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch        
        
        # output of meta is x,edge_attr, u
        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, u, batch)
        
        # idea1 is to cat x2, edge_attr2 and u2?
        # idea2 is to update edge_attr and u...
        data.x = x2
        data.edge_attr = edge_attr2
        data.u = u2


        # version using x and  u
        x = F.relu(torch.cat([x2,u2], dim=0))
        x = F.dropout(x, training=self.training) # until here the output is for each node
        ubatch = list(set([ elem.item() for elem in batch]))
        #print(ubatch)
        x = self.global_pool(x, torch.cat([batch,torch.LongTensor( ubatch).to(device) ],dim=0)) # this makes the output to be graph level?
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class META4(torch.nn.Module):
    def __init__(self, d1=3, d2=50, d3=15, d4 =15,d5=10,num_classes=6):
        super(META4, self).__init__()

        self.edge_mlp = Seq(Lin(d1*3, d2), ReLU(), Lin(d2, d3))
        self.node_mlp = Seq(Lin(d1*6, d2), ReLU(), Lin(d2, d3))
        self.global_mlp = Seq(Lin(16, d2), ReLU(), Lin(d2, d3))
        
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

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, data):
        
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch        
        
        # output of meta is x,edge_attr, u
        x2, edge_attr2, u2 =  self.op(x, edge_index, edge_attr, u, batch)
        
        # idea1 is to cat x2, edge_attr2 and u2?
        # idea2 is to update edge_attr and u...
        data.x = x2
        data.edge_attr = edge_attr2
        data.u = u2

    
        # version using x and  u and edge_attr
        x = F.relu(torch.cat([x2,u2], dim=0))
        #x = x2
        x = F.dropout(x, training=self.training) # until here the output is for each node
        
        x = self.global_pool(x, torch.cat([batch, ],dim=0)) # this makes the output to be graph level?
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
class META5(torch.nn.Module):
    """
        Not using edge attribute
    """
    def __init__(self, d1=3, d2=50, d3=15):
        super(META5, self).__init__()

        self.edge_mlp = Seq(Lin(d1*2, d2), ReLU(), Lin(d2, d3))
        self.node_mlp = Seq(Lin(d1, d2), ReLU(), Lin(d2, d3))
        self.global_mlp = Seq(Lin(2, d2), ReLU(), Lin(d2, d3))

        def edge_model(source, target, edge_attr, u):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            out = torch.cat([source, target], dim=1)
            #print("edge_model")
            #print(out.size())
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            row, col = edge_index
            out = torch.cat([x[col]], dim=1)
            out = self.node_mlp(out)
            return scatter_mean(out, row, dim=0, dim_size=x.size(0))

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            
            return self.global_mlp(out)

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        #print("Forward: ")
        #print(x.size())
        return self.op(x, edge_index, edge_attr, u, batch)