import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import ceil
import glob
import unidecode 
from datetime import date, timedelta
import pickle5 as pkl
from sklearn import preprocessing

import os
    
    
    
def read_meta_datasets(window):
    meta_labs = []
    meta_graphs = []
    meta_features = []
    meta_y = []


    #--------------- US
    labels = pd.read_csv('/content/gdrive/My Drive/pand_pred/cdc-pipeline/cdc_matrix/cdc_matrix.csv')
    labels[labels <0] = 0
    mob = pkl.load(open('/content/gdrive/My Drive/MedData/SafeGraphCov19/MobData/MobData_all.pkl', 'rb'))
    # mob data start from Dec 23 while label starts from Jan 22
    # sdate = date(2020, 1, 22)
    # edate = date(2020, 6, 15)
    # delta = edate - sdate
    # dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    # dates = [str(date) + '  12:00:00 AM'for date in dates]
    mob = mob[29:]
    Gs = []
    for d in range(mob.shape[0]):
        mat = mob[d]
        G = nx.DiGraph()
        nodes = set([i for i in range(51)])
        G.add_nodes_from(nodes)
        for x in range(51):
            for y in range(51):
                G.add_edge(x, y, weight=mat[x, y])
        Gs.append(G)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    labels = labels[labels.columns[1:]]
    labels[labels < 0] = 0    
    meta_labs.append(labels)
    # print(labels.shape, len(gs_adj))
    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels, "US", window)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.iloc[node,i])

    meta_y.append(y)

    os.chdir("/content/gdrive/My Drive/pand_pred/pandemic_tgnn-master/output")

    return meta_labs, meta_graphs, meta_features, meta_y
    
def generate_new_adj(gs_adj, window):
    new = []
    gs_adj = [i.reshape([1, 51, 51]) for i in gs_adj]
    for i in range(len(gs_adj)):
        if i < window - 1:
            pad = np.tile(gs_adj[0], (window-1-i, 1, 1))
            new.append(np.concatenate([pad] + gs_adj[:i+1], axis=0))
        else:
            new.append(np.concatenate(gs_adj[i-window+1:i+1], axis=0))
    return new 

def generate_graphs_tmp(dates,country):
    Gs = []
    for date in dates:
        d = pd.read_csv("graphs/"+country+"_"+date+".csv",header=None)
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)
        
    return Gs

def generate_new_features(Gs, labels, country, window=7, scaled=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
  

    #--- one hot encoded the region
    departments_name_to_id = dict()
    for node in nodes:
        departments_name_to_id[node] = len(departments_name_to_id)

    n_departments = len(departments_name_to_id)
    n_departments=0
    #print(n_departments)
    for idx,G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        
        H = np.zeros([G.number_of_nodes(),window+n_departments]) #+3+n_departments])#])#])
        me = labs.iloc[:, :(idx)].mean(1)
        sd = labs.iloc[:, :(idx)].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
            #---- Past cases      
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.iloc[node, :(idx)] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.iloc[node, :(idx)]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.iloc[node, (idx-window):(idx)] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.iloc[node, (idx-window):(idx)]
      
        features.append(H)
        
    return features

def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample, pred_tables = []):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    #n_nodes = Gs[0].number_of_nodes()
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #---- fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):                
                adj_tmp.append(Gs[k-1].T)  
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]
            
            
            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    
                    if(len(pred_tables)>0):
                        # for one sample after last known sample (test_sample) we know the predictions of shift 0,
                        # for two samples we know the predictions of shift 1 etc.. 
                        y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = pred_tables[(val+shift)- test_sample][:,val+shift]
                    else:
                        y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
                        
            else:
                #y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        #adj_lst.append(adj_tmp.to(device))
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst

def generate_batches_lstm(n_nodes, y, idx, window, shift, batch_size, device,test_sample):
    """
    Generate batches for graphs for the LSTM
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()
    pred_tables=[]
    
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        #step = n_nodes#*window
        step = n_nodes*1

        adj_tmp = list()
        #features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))
        features_tmp = np.zeros((window, n_nodes_batch))#features.shape[1]))
        
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)
        
        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]
            
            # keep the past information from val-window until val-1
            for e2,k in enumerate(range(val-window,val)):
               
                if(k==0): 
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])#features#[k]
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])#.reshape([n_nodes,1])#

            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    if(len(pred_tables)>0):
                        # for one sample after last known sample (test_sample) we know the predictions of shift 0,
                        # for two samples we know the predictions of shift 1 etc.. 
                        y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = pred_tables[(val+shift)- test_sample][:,val+shift]
                    else:
                        y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
            else:
                #y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]       
            #y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
            #for k in range(n_nodes):
            #    y_tmp[(n_nodes*e1)+k] = np.mean([y[v][k] for v in range(val,val+shift_targets)])
                
        adj_fake.append(0)
        
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))
        
    return adj_fake, features_lst, y_lst




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count