import random
import numpy as np
from scipy.sparse.linalg import lsmr
import time
import networkx as nx
import os
import gc
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from networkx.algorithms import centrality
from torch_geometric.data import Data


# 得到营养级
def get_levels(A):
    w_in = A.sum(axis=0)  # 计算入度
    w_out = A.sum(axis=1)  # 计算出度
    u = w_in + w_out
    v = w_in - w_out
    Lambda = np.diag(u) - A - A.T
    h = lsmr(Lambda, v)[0]
    h = h - min(h)  # 保证营养级从0开始
    del Lambda, w_in, w_out, u, v
    gc.collect()
    return h


def calc_troph_incoh(A, h):
    F = 0
    if (A.sum == 0):
        return F
    idx = np.nonzero(A)
    for i in range(len(idx[0])):
        x = idx[0][i]
        y = idx[1][i]
        F = F + (h[y] - h[x] - 1) ** 2
    F = F / A.sum()
    del idx
    gc.collect()
    return F


class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.05):  # dropout=0.6
        super(ResidualGATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = lambda x: x

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.dropout(x)
        return x + res


class DeepGATNet(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features, heads_per_layer, mlp_dims):
        super(DeepGATNet, self).__init__()
        assert len(hidden_dims) == len(heads_per_layer), "Hidden dimensions and heads per layer counts must match."
        self.layers = nn.ModuleList()

        # 添加GAT层
        current_dim = in_features
        for dim, heads in zip(hidden_dims, heads_per_layer):
            self.layers.append(ResidualGATLayer(current_dim, dim, heads))
            current_dim = dim

        # 添加最后一层GATConv，不使用残差连接
        self.layers.append(GATConv(current_dim, out_features, heads=1, concat=False))

        # 添加全连接层（多层感知器）
        self.mlp = nn.Sequential(
            nn.Linear(out_features, mlp_dims[0]),
            nn.ELU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ELU(),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.Sigmoid()  # 最后一层使用Sigmoid激活函数
        )

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:  # 前面的GAT层
            x = layer(x, edge_index)
        x = self.layers[-1](x, edge_index)  # 最后一层GAT
        x = self.mlp(x)
        return x.squeeze()


def load_DND(model, optimizer, filepath):
    print(device)
    checkpoint = torch.load(filepath, map_location=device)
    # print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # 切换到评估模式
    return model, optimizer


def DND_features(G):
    features = {
        "in_degree": [degree for node, degree in G.in_degree()],
        "out_degree": [degree for node, degree in G.out_degree()],
        "betweenness": list(centrality.betweenness_centrality(G).values()),
        "pagerank": list(nx.pagerank(G).values()),
    }
    features_dict = features
    features = np.array([features_dict[key] for key in sorted(features_dict.keys())]).T

    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    edge_index = np.array(adj_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)  # 使用计算的特征
    return Data(x=x, edge_index=edge_index)




def findfile(directory):
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            filenames.append(fileName)
    return filenames



def ATAD_analysis(dir):
    filenames = findfile(dir)
    epoch = 0
    for file in filenames:
        print('epoch:', epoch)
        print(file)
        filename = file
        g = nx.read_graphml(dir + filename)  # 读取graphhml形式储存的图
        g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
        nodes = list(g.nodes)
        N = len(nodes)
        print("N",N)

        if dir == "data/large_scale_SF_network/":
            # 找到所有强连通分量
            sccs = list(nx.strongly_connected_components(g))
            # 找到最大的强连通分量
            largest_scc = max(sccs, key=len)
            # 创建极大强连通子图
            new_g = g.subgraph(largest_scc).copy()
            # 更新 g 为极大强连通子图
            g = new_g


        nodes = list(g.nodes)
        N = len(nodes)
        print("N",N)
        edges = list(g.edges)
        L= len(edges)
        print("L", L)



        # ATAD方法选取节点
        print("ATAD")
        start_time = time.time()
        ATD_nodes = []
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        while A.sum():
            h = get_levels(A)
            h = h.reshape((h.shape[0], 1))
            h_h = h.T - h
            h_h = np.where(h_h >= 0, 1, h_h)
            h_h = np.power((h_h - 1), 2)
            h_h_A = h_h * A
            if h_h_A.sum() == 0:
                break
            up = h_h_A.sum(axis=1) + h_h_A.sum(axis=0)
            selected = np.argsort(up)[::-1]
            d = selected[0]
            ATD_nodes.append(nodes[d])
            A[:, d] = 0
            A[d, :] = 0
        ATD_nodes = ATD_nodes + list(set(g.nodes()) - set(ATD_nodes))
        print(ATD_nodes)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"ATAD运行时间：{run_time} 秒")


        # TAD方法选取节点
        print("TAD")
        start_time = time.time()
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        h = get_levels(A)
        h = h.reshape((h.shape[0], 1))
        h_h = h.T - h
        h_h = np.where(h_h >= 0, 1, h_h)
        h_h = np.power((h_h - 1), 2)
        h_h_A = h_h * A
        up = h_h_A.sum(axis=1) + h_h_A.sum(axis=0)
        selected = np.argsort(up)[::-1]
        back_nodes = np.array(nodes)[selected]
        back_nodes = back_nodes.tolist() + list(set(g.nodes) - set(back_nodes))
        end_time = time.time()
        run_time = end_time - start_time
        print(f"TAD运行时间：{run_time} 秒")

        # 随机方法选取节点：
        print("Rand")
        start_time = time.time()    # 记录开始时间
        nodes_rand = list(g.nodes)
        random.shuffle(nodes_rand)
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"Rand运行时间：{run_time} 秒")


        # 适应度方法选取节点：
        print("HDA")
        start_time = time.time()    # 记录开始时间
        adapt_degree=[]
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        nodes=list(g.nodes())
        while A.sum():
            D = A.sum(axis=0) + A.sum(axis=1)
            d=np.argmax(D)
            adapt_degree.append(nodes[d])
            A[:, d] = 0  # 以前是行置零
            A[d,:] = 0
        adapt_degree = adapt_degree + list(set(g.nodes()) - set(adapt_degree))
        print(adapt_degree)
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"HDA运行时间：{run_time} 秒")


        # 度方法选取节点
        print("HD")
        start_time = time.time()    # 记录开始时间
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        D=A.sum(axis=0)+A.sum(axis=1)
        d = sorted(range(N), key=lambda k: D[k], reverse=True)
        degree=np.array(list(g.nodes))[d]
        degree=degree.tolist()+list(set(g.nodes())-set(degree))
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"HD运行时间：{run_time} 秒")



        # DND方法取节点
        print("DND")
        start_time = time.time()
        mlp_dims = [100, 50, 1]
        model_DND = DeepGATNet(in_features=4, hidden_dims=[40, 30, 20, 10], out_features=mlp_dims[0], heads_per_layer=[5, 5, 5, 5], mlp_dims=mlp_dims).to(device)
        optimizer_DND = torch.optim.Adam(model_DND.parameters(), lr=0.000085)
        model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND, '../Data/other/model_checkpoint_DND.pth')
        features_DND = DND_features(g)
        with torch.no_grad():
            model_DND.eval()
            out = model_DND(features_DND.x.to(device), features_DND.edge_index.to(device))
        sorted_indices_DND = torch.argsort(out, descending=True)
        DND_nodes = np.array(list(g.nodes))[sorted_indices_DND.tolist()]
        end_time = time.time()          # 记录结束时间
        run_time = end_time - start_time
        print(f"DND运行时间：{run_time} 秒")



        # FINDER方法取节点
        print("FINDER")
        start_time = time.time()    # 记录开始时间
        idx2 = np.load('../Data/other/FINDER_selected/' + file+'.npy').astype(int)
        nodes = [str(item) for item in idx2]
        idx2 = nodes + list(set(g.nodes()) - set(nodes))
        print(idx2)
        #print(idx2)
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"FINDER运行时间：{run_time} 秒")

        # 入度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
        print("ID")
        start_time = time.time()    # 记录开始时间
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        D = A.sum(axis=0)
        d = sorted(range(N), key=lambda k: D[k], reverse=True)
        # print(d)
        degree_in = np.array(list(g.nodes))[d]
        degree_in = degree_in.tolist() + list(set(g.nodes()) - set(degree_in))
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"ID运行时间：{run_time} 秒")

        # 出度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
        print("OD")
        start_time = time.time()    # 记录开始时间
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        D = A.sum(axis=1)
        d = sorted(range(N), key=lambda k: D[k], reverse=True)
        # print(d)
        degree_out = np.array(list(g.nodes))[d]
        degree_out = degree_out.tolist() + list(set(g.nodes()) - set(degree_out))
        end_time = time.time()          # 记录结束时间
        # 计算运行时间
        run_time = end_time - start_time
        print(f"OD运行时间：{run_time} 秒")


        g1 = g.copy()
        g2 = g.copy()
        g3 = g.copy()
        g4 = g.copy()
        g5 = g.copy()
        g6 = g.copy()
        g7 = g.copy()
        g8 = g.copy()
        g9 = g.copy()

        strong_list_ATD = []
        strong_list_TAD = []
        strong_list_rand = []
        strong_list_adapt_degree = []
        strong_list_degree = []
        strong_list_DND = []
        strong_list_FINDER = []
        strong_list_ID = []
        strong_list_OD = []


        for i in range(len(ATD_nodes)):

            strong_ATD = max(nx.strongly_connected_components(g1), key=len)
            strong_list_ATD.append(len(strong_ATD) / N)
            edges = list(g1.in_edges(ATD_nodes[i])) + list(g1.out_edges(ATD_nodes[i]))
            g1.remove_edges_from(edges)

            strong_TAD = max(nx.strongly_connected_components(g2), key=len)
            strong_list_TAD.append(len(strong_TAD) / N)
            edges = list(g2.in_edges(back_nodes[i])) + list(g2.out_edges(back_nodes[i]))
            g2.remove_edges_from(edges)


            strong_rand = max(nx.strongly_connected_components(g3),key=len)
            strong_list_rand.append(len(strong_rand) / N)
            edges = list(g3.in_edges(nodes_rand[i]))+list(g3.out_edges(nodes_rand[i]))
            g3.remove_edges_from(edges)

            strong_adapt_degree = max(nx.strongly_connected_components(g4), key=len)
            strong_list_adapt_degree.append(len(strong_adapt_degree) / N)
            edges = list(g4.in_edges(adapt_degree[i])) + list(g4.out_edges(adapt_degree[i]))
            g4.remove_edges_from(edges)


            strong_degree = max(nx.strongly_connected_components(g5),key=len)
            strong_list_degree.append(len(strong_degree) / N)
            edges=list(g5.in_edges(degree[i]))+list(g5.out_edges(degree[i]))
            g5.remove_edges_from(edges)


            strong_DND= max(nx.strongly_connected_components(g6), key=len)
            strong_list_DND.append(len(strong_DND) / N)
            edges = list(g6.in_edges(str(DND_nodes[i]))) + list(g6.out_edges(str(DND_nodes[i])))
            g6.remove_edges_from(edges)


            strong_FINDER= max(nx.strongly_connected_components(g7), key=len)
            strong_list_FINDER.append(len(strong_FINDER) / N)
            edges = list(g7.in_edges(str(idx2[i]))) + list(g7.out_edges(str(idx2[i])))
            g7.remove_edges_from(edges)


            strong_degree_in = max(nx.strongly_connected_components(g8),key=len)
            strong_list_ID.append(len(strong_degree_in) / N)
            edges=list(g8.in_edges(degree_in[i]))+list(g8.out_edges(degree_in[i]))
            g8.remove_edges_from(edges)


            strong_degree_out = max(nx.strongly_connected_components(g9),key=len)
            strong_list_OD.append(len(strong_degree_out) / N)
            edges=list(g9.in_edges(degree_out[i]))+list(g9.out_edges(degree_out[i]))
            g9.remove_edges_from(edges)

        if dir == "data/large_scale_SF_network/":
            np.save('../Data/ND_results/NPY' + filename + '_ATD_GSCC.npy', strong_list_ATD)
            np.save('../Data/ND_results/NPY/'+filename+'_TAD_GSCC.npy',strong_list_TAD)
            np.save('../Data/ND_results/NPY/'+filename+'_Rand_GSCC.npy',strong_list_rand)
            np.save('../Data/ND_results/NPY/'+filename+'_HDA_GSCC.npy',strong_list_adapt_degree)
            np.save('../Data/ND_results/NPY/'+filename+'_HD_GSCC.npy',strong_list_degree)
            np.save('../Data/ND_results/NPY/' + filename + '_DND_GSCC.npy', strong_list_DND)
            np.save('../Data/ND_results/NPY/' + filename + '_FD_GSCC.npy', strong_list_FINDER)
            np.save('../Data/ND_results/NPY/' + filename + '_ID_GSCC.npy', strong_list_ID)
            np.save('../Data/ND_results/NPY/' + filename + '_OD_GSCC.npy', strong_list_OD)
        else:
            np.save('../Data/ND_results/NPY/' + filename + '_ATD.npy', strong_list_ATD)
            np.save('../Data/ND_results/NPY/' + filename + '_TAD.npy', strong_list_TAD)
            np.save('../Data/ND_results/NPY/' + filename + '_Rand.npy', strong_list_rand)
            np.save('../Data/ND_results/NPY/' + filename + '_HDA.npy', strong_list_adapt_degree)
            np.save('../Data/ND_results/NPY/' + filename + '_HD.npy', strong_list_degree)
            np.save('../Data/ND_results/NPY/' + filename + '_DND.npy', strong_list_DND)
            np.save('../Data/ND_results/NPY/' + filename + '_FD.npy', strong_list_FINDER)
            np.save('../Data/ND_results/NPY/' + filename + '_ID.npy', strong_list_ID)
            np.save('../Data/ND_results/NPY/' + filename + '_OD.npy', strong_list_OD)

        epoch += 1
        pass



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ### synthetic SF networks
    use_dir = "../Data/Networks/SF_network/"
    ATAD_analysis(use_dir)

    ### synthetic ER networks
    use_dir = "../Data/Networks/ER_network/"
    ATAD_analysis(use_dir)

    ### real-world networks
    use_dir = "../Data/Networks/real_network/"
    ATAD_analysis(use_dir)

    ### synthetic large-scale SF networks
    use_dir = "../Data/Networks/large_scale_SF_network/"
    ATAD_analysis(use_dir)
