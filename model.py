import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import BatchNorm1d, Dropout, Linear, SELU, ReLU
from torch_geometric.utils import add_self_loops, scatter

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # model specifications as per table 8 in <https://arxiv.org/pdf/1910.10685>
        input_channels = 11
        # 152 cols in dravnieks, 146 output features - unsure where 138 came from in research paper
        output_channels = 146
        pool_dim = 175
        hidden_channels = [15, 20, 27, 36]
        fc_channels = [96, 63]

        # GCN Layers
        self.conv1 = GCNConv(input_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GCNConv(hidden_channels[2], hidden_channels[3])

        # Readout Layers
        self.read1 = torch.nn.Linear(hidden_channels[0], pool_dim)
        self.read2 = torch.nn.Linear(hidden_channels[1], pool_dim)
        self.read3 = torch.nn.Linear(hidden_channels[2], pool_dim)
        self.read4 = torch.nn.Linear(hidden_channels[3], pool_dim)
        
        # Fully Connected Layers
        self.fc1 = Linear(pool_dim, fc_channels[0])
        self.fc2 = Linear(fc_channels[0], fc_channels[1])
        
        # BatchNorm Layers
        self.bn1 = BatchNorm1d(fc_channels[0])
        self.bn2 = BatchNorm1d(fc_channels[1])
        
        # Dropout Layer
        self.dropout = Dropout(0.47)
        
        # Prediction Layer
        self.prediction = Linear(fc_channels[1], output_channels)

        # Activation Layers
        self.activate_selu = SELU()
        self.activate_relu = ReLU()
        
    def forward(self, x, edge_index, batch):
        readout = 0

        # First message passing layer
        x = self.conv1(x, edge_index)
        x = self.activate_selu(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.read1(x), dim=-1)
        
        # Second message passing layer
        x = self.conv2(x, edge_index)
        x = self.activate_selu(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.read2(x), dim=-1)
        
        # Third message passing layer
        x = self.conv3(x, edge_index)
        x = self.activate_selu(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.read3(x), dim=-1)
        
        # Fourth message passing layer
        x = self.conv4(x, edge_index)
        x = self.activate_selu(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.read4(x), dim=-1)

        # Readout layer
        x = global_add_pool(readout, batch=batch)
        
        # First fully connected layer
        x = self.fc1(x)
        x = self.activate_relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        x = self.activate_relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        # Prediction Layer
        x = self.prediction(x)
        x = torch.sigmoid(x)
        
        return x

    @staticmethod
    def max_graph_pool(x, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        x = scatter(x[row], col, dim=0, reduce="max")
        return x