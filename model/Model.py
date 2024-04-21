import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv,GATConv
import copy
from transformers import RobertaModel,RobertaTokenizer
from transformers import AutoTokenizer,AutoModel,TFAutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GATConv(in_feats, hid_feats,heads=3,concat=False)
        self.conv2 = GATConv(hid_feats+in_feats, out_feats,heads=3,concat=False)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).cuda()
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).cuda()
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GATConv(in_feats, hid_feats,heads=3,concat=False)
        self.conv2 = GATConv(hid_feats+in_feats, out_feats,heads=3,concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).cuda()
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).cuda()
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class GraphBranch(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(GraphBranch, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 2)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        return x


class TextBranch(nn.Module):

    def __init__(self, hidden_size=768, num_classes=4):
        super(TextBranch, self).__init__()
        self.bert_weights = "./weights/twitter-roberta-base-sep2022"
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(self.bert_weights)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, data):
        length = int(data.source_length[0])
        context = data.source.reshape(-1,length)
        mask = data.source_mask.reshape(-1,length)
        out = self.bert(context, attention_mask=mask).pooler_output   # output_hidden_states

        return out


class Model(th.nn.Module):
    def __init__(self, gcninC, gcnhidC, gcnoutC, berthidC,num_classes=2):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.catChannel = gcnhidC+berthidC
        self.Bert = TextBranch(berthidC,num_classes)
        self.Gcn = GraphBranch(gcninC, gcnhidC, gcnoutC)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.catChannel, self.catChannel//4)
        self.fc2 = nn.Linear(self.catChannel//4, self.num_classes)

    def forward(self, data):
        text_feature = self.Bert(data)
        graph_feature = self.Gcn(data)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        text_feature = fuse_weights[0] * text_feature
        graph_feature= fuse_weights[1] * graph_feature
        x=  torch.cat((text_feature, graph_feature), dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        out = F.log_softmax(x, dim=1)

        return out