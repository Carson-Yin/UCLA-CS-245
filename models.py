import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from transformer import TransformerEncoderLayer

hiddenimports = collect_submodules('fbprophet')
datas = collect_data_files('fbprophet')



class Attention(nn.Module):
    def __init__(self, ftlen):
        super(Attention, self).__init__()
        self.w_mat = nn.Linear(ftlen, int(ftlen / 2))
        self.tanh = nn.Tanh()
        self.v_mat = nn.Linear(int(ftlen / 2), 1)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, LSTM_out):
        # print(LSTM_out.size())
        x = self.w_mat(LSTM_out)
        # print(x.size())
        x = self.tanh(x)
        x = self.v_mat(x)
        # print(x.size())
        x = self.softmax(x)
        # print(x.size())
        return x

class MPNN_TSFM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_TSFM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.tsfm1 = TransformerEncoderLayer(2*nhid, 1, nhid, dropout)
        self.tsfm2 = TransformerEncoderLayer(2*nhid, 1, nhid, dropout)

        
        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(4*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
       # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        # print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        # print(x.shape)
        #print("------")
        x = x.permute([1, 0, 2])
        out1 = self.tsfm1(x)    
        out2 = self.tsfm2(out1)

        #print(self.rnn2._all_weights)
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)
                
        x = torch.cat([out1[:,-1,:], out2[:,-1,:], skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
       # print("--------------------")
        
        
        return x

# class MPNN_LSTM_Attn2(nn.Module):
#     def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
#         super(MPNN_LSTM_Attn2, self).__init__()
#         self.window = window
#         self.n_nodes = n_nodes
#         #self.batch_size = batch_size
#         self.nhid = nhid
#         self.nfeat = nfeat
#         self.conv1 = GCNConv(nfeat, nhid)
#         self.conv2 = GCNConv(nhid, nhid)
        
#         self.bn1 = nn.BatchNorm1d(nhid)
#         self.bn2 = nn.BatchNorm1d(nhid)
        
#         self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
#         self.rnn2 = nn.LSTM(nhid, nhid, 1)
#         self.attn1 = Attention(nhid)
#         self.attn2 = Attention(nhid)
        
#         #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
#         self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
#         self.fc2 = nn.Linear( nhid, nout)
        
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
        
#     def forward(self, adj, x):
#         lst = list()
#        # print("--------------------")
#         weight = adj.coalesce().values()
#         adj = adj.coalesce().indices()
#        # print(x.shape)
#         skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
#        # print(skip.shape)
#         skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
#         x = self.relu(self.conv1(x, adj,edge_weight=weight))
#         x = self.bn1(x)
#         x = self.dropout(x)
#         lst.append(x)
        
#         x = self.relu(self.conv2(x, adj,edge_weight=weight))
#         x = self.bn2(x)
#         x = self.dropout(x)
#         lst.append(x)
        
#         x = torch.cat(lst, dim=1)
        
#         #--------------------------------------
#         #print(x.shape)
#         x = x.view(-1, self.window, self.n_nodes, x.size(1))
#         #print(x.shape)
#         #print(x.shape)
#         x = torch.transpose(x, 0, 1)
#         x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
#         #print(x.shape)
#         #print("------")
#         out1, _ = self.rnn1(x)
#         weights1 = self.attn1(out1)        
#         out2, _ = self.rnn2(out1)
#         weights2 = self.attn2(out2)
#         out1 = torch.sum(out1 * weights1, dim=0)
#         out2 = torch.sum(out2 * weights2, dim=0)

#         #print(self.rnn2._all_weights)
#         #print(skip.shape)
#         #print(x.shape)
#         #skip = skip.view(skip.size(0),-1)
#         skip = skip.reshape(skip.size(0),-1)
#         #print(x.shape)
#         #print(skip.shape)
                
#         x = torch.cat([out1, out2, skip], dim=1)
#         #--------------------------------------
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x)).squeeze()
#         x = x.view(-1)
#        # print("--------------------")
        
        
#         return x
class MPNN_LSTM_Attn3(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM_Attn3, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn = nn.LSTM(2*nhid, nhid, 2)
        self.attn = Attention(nhid)
        
        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        for m in self.modules():
            if m.isinstance(nn.Linear):
                m.weight.data.normal_(0.01)           
        
    def forward(self, adj, x, return_weights=False):
        lst = list()
        print(adj.size())
        print(x.size())
       # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        #print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        #print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        #print(x.shape)
        #print("------")
        x, _ = self.rnn(x)
        weights = self.attn(x)

        x = torch.sum(x * weights, dim=0)
        
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)
                
        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
       # print("--------------------")
        
        if not return_weights:
            return x   
        else:
            return x, weights         

class MPNN_LSTM_Attn(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM_Attn, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn = nn.LSTM(2*nhid, nhid, 2)
        self.attn = Attention(nhid)
        
        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x, return_weights=False):
        lst = list()
       # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        #print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        #print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        #print(x.shape)
        #print("------")
        x, _ = self.rnn(x)
        weights = self.attn(x)

        x = torch.sum(x * weights, dim=0)
        
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)
                
        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
       # print("--------------------")
        
        
        if not return_weights:
            return x   
        else:
            return x, weights           
            
class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
       # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        #print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        #print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        #print(x.shape)
        #print("------")
        x, (hn1, cn1) = self.rnn1(x)
        
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        #print(self.rnn2._all_weights)
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)
                
        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
       # print("--------------------")
        
        
        return x
 



class MPNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MPNN, self).__init__()
        #self.n_nodes = n_nodes
    
        #self.batch_size = batch_size
        self.nhid = nhid
        
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid) 
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)
        #self.bn3 = nn.BatchNorm1d(nhid)
        #self.bn4 = nn.BatchNorm1d(nhid)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        #nn.init.zeros_(self.conv1.weight)
        #nn.init.zeros_(self.conv2.weight)
        #nn.init.zeros_(self.fc1.weight)
        #nn.init.zeros_(self.fc2.weight)
        
        
    def forward(self, adj, x):
        lst = list()
        #print(x.shape)
        #print(adj.shape)
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       
        #lst.append(ident)
        
        #x = x[:,mob_feats]
        #x = xt.index_select(1, mob_feats)
        lst.append(x)
        
        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        #print(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        #print(x.shape)
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        
        x = torch.cat(lst, dim=1)
                                   
        x = self.relu(self.fc1(x))
        #x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x)).squeeze() # 
        #x = self.bn4(x)
        
        x = x.view(-1)
        
        return x

    
    
    
class LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout,batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers= 2
        
        self.nfeat = nfeat 
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)
    
        self.linear = nn.Linear(nhid, self.nout)
        self.cell = ( nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))
      
        #self.hidden_cell = (torch.zeros(2,self.batch_size,self.nhid).to(device),torch.zeros(2,self.batch_size,self.nhid).to(device))
        #nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))
        
        
    def forward(self, adj, features):
        #adj is 0 here
        #print(features.shape)
        features = features.view(self.window,-1, self.n_nodes)#.view(-1, self.window, self.n_nodes, self.nfeat)
        #print(features.shape)
        #print("----")
        
        
        #------------------
        if(self.recur):
            #print(features.shape)
            #self.hidden_cell = 
            try:
                lstm_out, (hc,self.cell) = self.lstm(features,(torch.zeros(self.nb_layers,self.batch_size,self.nhid).cuda(),self.cell)) 
                # = (hc,cn)
            except:
                #hc = self.hidden_cell[0][:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                hc = torch.zeros(self.nb_layers,features.shape[1],self.nhid).cuda()                 
                cn = self.cell[:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                lstm_out, (hc,cn) = self.lstm(features,(hc,cn)) 
        else:
        #------------------
            lstm_out, (hc,cn) = self.lstm(features)#, self.hidden_cell)#self.hidden_cell 
            
        predictions = self.linear(lstm_out)#.view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
        #print(predictions.shape)
        return predictions[-1].view(-1)
