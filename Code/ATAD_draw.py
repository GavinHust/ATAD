import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import powerlaw
import os
from scipy.sparse.linalg import lsmr
import gc
from numpy import array
from numpy import float32
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline


def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames

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



# 绘制拓扑图并标出 剩余的节点和边
def draw_topology(g, node_remove, node_remain, lsc_nodes,  ax, label):
    g_copy = g.copy()
    g = g_copy

    # 计算每个节点的度
    degrees = dict(g.degree())
    min_deg = min(degrees.values())
    max_deg = max(degrees.values())
    min_size = 100  # 最小像素
    max_size = 100  # 最大像素
    node_sizes = [min_size + (max_size - min_size) * (degrees[n] - min_deg) / (max_deg - min_deg + 1e-9) for n in g.nodes()]

    print("node_remove", node_remove)
    print("lsc_nodes", lsc_nodes)


    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    h = get_levels(A)
    print(h)
    pos = {'1': array([0.40768704, 0.05536604], dtype=float32), '2': array([0.5585349 , 0.28730518], dtype=float32), '3': array([0.58035058, 0.3039123 ], dtype=float32),
           '4': array([0.42639955, 0.6238122 ], dtype=float32), '5': array([0.23037745, 0.7897336 ], dtype=float32), '6': array([0.35093885, 0.6008161 ], dtype=float32),
           '7': array([0.42396854, 0.70864516], dtype=float32), '8': array([0.27753468, 0.9042672 ], dtype=float32), '9': array([0.25990486, 0.11892466], dtype=float32),
           '10': array([0.50530015, 0.20224823], dtype=float32), '11': array([0.62420267, 0.44914708], dtype=float32), '12': array([0.2897948 , 0.95739913], dtype=float32),
           '13': array([0.28333325, 0.50830996], dtype=float32), '14': array([0.53002354, 0.83503467], dtype=float32), '15': array([0.43733164, 0.7622905 ], dtype=float32),
           '16': array([0.38692224, 0.3622959 ], dtype=float32), '17': array([0.32457455, 0.5419849 ], dtype=float32), '18': array([0.22815547, 0.14499034], dtype=float32),
           '19': array([0.37889196, 0.18748127], dtype=float32), '20': array([0.38898148, 0.2400382 ], dtype=float32), '21': array([0.48665485, 0.56854475], dtype=float32),
           '22': array([0.2727648 , 0.17905511], dtype=float32), '23': array([0.4830777, 0.7200128], dtype=float32), '24': array([0.23106402, 0.5828253 ], dtype=float32),
           '25': array([0.6076776 , 0.90985346], dtype=float32), '26': array([0.256698  , 0.83572954], dtype=float32), '27': array([0.3511214 , 0.50632554], dtype=float32),
           '28': array([0.44449838, 0.06964476], dtype=float32), '29': array([0.5857957, 0.3913265], dtype=float32), '30': array([0.40817073, 0.17090505], dtype=float32),
           '31': array([0.32234514, 0.50998056], dtype=float32), '32': array([0.34074414, 0.4569195 ], dtype=float32), '33': array([0.56860098, 0.2565999 ], dtype=float32),
           '34': array([0.25109364, 0.0883537 ], dtype=float32), '35': array([0.6530169, 0.751329 ], dtype=float32), '36': array([0.60824253, 0.63344276], dtype=float32),
           '37': array([0.42870685, 0.9581425 ], dtype=float32), '38': array([0.3798569 , 0.11820124], dtype=float32), '39': array([0.38744941, 0.8951406 ], dtype=float32),
           '40': array([0.33208214, 0.7457428 ], dtype=float32), '41': array([0.45372217, 0.14775296], dtype=float32), '42': array([0.4557098 , 0.82382685], dtype=float32),
           '43': array([0.4616964 , 0.65852606], dtype=float32), '44': array([0.31356623, 0.9322602 ], dtype=float32), '45': array([0.52140648, 0.8665293 ], dtype=float32),
           '46': array([0.44316916, 0.1330757 ], dtype=float32), '47': array([0.5863102 , 0.11271576], dtype=float32), '48': array([0.5242527, 0.2838917], dtype=float32),
           '49': array([0.27124674, 0.6666988 ], dtype=float32), '50': array([0.5196445 , 0.28960174], dtype=float32), '51': array([0.53076876, 0.96579397], dtype=float32),
           '52': array([0.51422437, 0.21779506], dtype=float32), '53': array([0.46126464, 0.9261059 ], dtype=float32), '54': array([0.33413284, 0.90447366], dtype=float32),
           '55': array([0.32049135, 0.4564994 ], dtype=float32), '56': array([0.54785835, 0.24734345], dtype=float32), '57': array([0.63227504, 0.807594  ], dtype=float32),
           '58': array([0.6117755, 0.6862416], dtype=float32), '59': array([0.55078106, 0.04930931], dtype=float32), '60': array([0.65899247, 0.25851497], dtype=float32),
           '61': array([0.431912  , 0.59497964], dtype=float32)}
    new_pos = {}
    for node_id, coords in pos.items():
        node_index = int(node_id) - 1
        if h[node_index] >1.31:
            new_pos[node_id] = np.array([coords[0], h[node_index] * 1.2], dtype=np.float32)
        elif h[node_index] >1.2:
            new_pos[node_id] = np.array([coords[0], h[node_index]*1.1], dtype=np.float32)
        else:
            new_pos[node_id] = np.array([coords[0], h[node_index]], dtype=np.float32)


    pos = new_pos
    print(pos)


    node_colors = []
    if label == "Original Network":
        for node in g.nodes():
            if node in lsc_nodes[0]:
                node_colors.append('#4B0082')
            elif node in lsc_nodes[1]:
                node_colors.append('#800080')
            elif node in lsc_nodes[2]:
                node_colors.append('#9370DB')
            elif node in lsc_nodes[3]:
                node_colors.append('#DDA0DD')
            else:
                node_colors.append('dimgray')
    else:
        for node in g.nodes():
            if node in node_remove:
                #node_colors.append('black')
                node_colors.append('gray')
                x, y = pos[node]
                ax.scatter(x, y, marker='x', color='black', s=120, linewidths=2, alpha=0.8, zorder=5)
            elif node in lsc_nodes:
                node_colors.append('purple')
            else:
                node_colors.append('dimgray')



    if label == "Original Network":
        #nx.draw_networkx_nodes(g, pos, node_color='dimgray', node_size=node_sizes, ax=ax, alpha=1)
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=1)
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='black', alpha=0.2, style='solid', ax=ax, connectionstyle='arc3,rad=0.1')
    else:
        # 绘制节点
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=1)
        # 绘制边
        lsc_edges = [(u, v) for u, v in g.edges() if u in lsc_nodes and v in lsc_nodes]
    #     edges = [(edge[0], edge[1]) for edge in g.edges() if (edge[0] in node_remove or edge[1] in node_remove) and h[int(edge[0])-1]>h[int(edge[1])-1]]
    #     print("len(edges)", len(edges))
        nx.draw_networkx_edges(g, pos, edgelist=lsc_edges, edge_color='purple', alpha=0.5, style='solid', ax=ax, connectionstyle='arc3,rad=0.23')
    #     nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color='black', alpha=0.3, style='dashed', ax=ax)


    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if label == "Original Network":
        #ax.set_title(label,  fontsize=14, y=0.95)
        pass
    else:
        ax.set_title(label, fontsize=14, x=0.12, y=1, ha='left')
        ax.text(1.02, -0.15, f"removed: {len(node_remove)} nodes\nresidual GSCC: {len(lsc_nodes)} nodes", transform=ax.transAxes, fontsize=12, horizontalalignment='right',  verticalalignment='bottom')
    #nx.draw_networkx_labels(g, pos, labels={n: str(n) for n in g.nodes()}, font_size=15, font_color='black', ax=ax)



def draw_realexample():
    dir = "../Data/Networks/real_network/"
    network_name = "9_11_HIJACKERS_ASSOCIATES1"         #8

    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    nodes = list(g.nodes)
    N = len(nodes)
    print("节点数为：", N)
    print("边数为：", len(list(g.edges)))

    print(nx.strongly_connected_components(g))
    print(len(max(nx.strongly_connected_components(g), key=len)))
    nodes = list(g.nodes)
    N = len(nodes)
    print(N)


    # ATAD方法选取节点
    print("ATAD")
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


    # TAD方法选取节点
    print("TAD")
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
    TAD_nodes = back_nodes.tolist() + list(set(g.nodes) - set(back_nodes))



    print("FINDER")
    idx2 = np.load('../Data/other/FINDER_selected/' + network_name + '.npy').astype(int)
    nodes = [str(item) for item in idx2]
    DNetKey_nodes = nodes + list(set(g.nodes()) - set(nodes))


    print(ATD_nodes)
    print(TAD_nodes)
    print(DNetKey_nodes)


    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 9, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, :5])
    ax1 = fig.add_subplot(gs[0, 5:])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[1, 3:6])
    ax4 = fig.add_subplot(gs[1, 6:])

    nodes_nums_to_remove = 8

    #fig, ax1 = plt.subplots(1, 1, figsize=(4, 5))
    #lsc_nodes = max(nx.strongly_connected_components(g), key=len)
    #draw_topology(g, [], nodes, lsc_nodes, ax[0][0], "Original Network")
    #lsc_nodes = max(nx.strongly_connected_components(g), key=len)
    #draw_topology(g, [], nodes, lsc_nodes, ax1, "Original Network")
    ax0.axis('off')
    ax0.text(-0.0, 1.06, 'a', transform=ax0.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')


    g_copy = g.copy()
    nodes_remove = ATD_nodes[:nodes_nums_to_remove]
    g_copy.remove_nodes_from(nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    ax_inset = ax2.inset_axes([0.05, 0.12, 0.9, 0.8])
    ax_inset.axis('off')
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    draw_topology(g, ATD_nodes[:nodes_nums_to_remove], ATD_nodes[nodes_nums_to_remove:], lsc_nodes, ax_inset, "ATAD")
    ax2.text(0.05, 0.985, 'c :', transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')



    g_copy = g.copy()
    nodes_remove = TAD_nodes[:nodes_nums_to_remove]
    g_copy.remove_nodes_from(nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    ax_inset = ax3.inset_axes([0.05, 0.12, 0.9, 0.8])
    ax_inset.axis('off')
    ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    draw_topology(g, TAD_nodes[:nodes_nums_to_remove], TAD_nodes[nodes_nums_to_remove:], lsc_nodes, ax_inset, "TAD")
    ax3.text(0.05, 0.985, 'd :', transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')


    g_copy = g.copy()
    nodes_remove = DNetKey_nodes[:nodes_nums_to_remove]
    g_copy.remove_nodes_from(nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    ax_inset = ax4.inset_axes([0.05, 0.12, 0.9, 0.8])
    ax_inset.axis('off')
    ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    draw_topology(g, DNetKey_nodes[:nodes_nums_to_remove], DNetKey_nodes[nodes_nums_to_remove:], lsc_nodes, ax_inset, "FINDER")
    ax4.text(0.05, 0.985, 'e :', transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')




    atd = np.load('../Data/ND_results/NPY/' + network_name + '_ATD.npy')
    tad = np.load('../Data/ND_results/NPY/' + network_name + '_TAD.npy')
    adpdegree = np.load('../Data/ND_results/NPY/' + network_name + '_HDA.npy')
    finder = np.load('../Data/ND_results/NPY/' + network_name + '_FD.npy')
    dnd = np.load('../Data/ND_results/NPY/' + network_name + '_DND.npy')

    x = [_ / len(atd) for _ in range(len(atd))]

    ax1.plot(x, atd, color='#403990', lw=2.2)
    ax1.plot(x, tad, color="#888888", lw=1.8)
    ax1.plot(x, adpdegree, color="#FBDD85", lw=1.8)
    ax1.plot(x, finder, color="#00FF00", lw=1.8)
    ax1.plot(x, dnd, color="#80A6E2", lw=1.8)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_ylabel('GSCC', fontsize=14)
    ax1.set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax1.text(-0.15, 1.06, 'b', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax1.legend(["ATAD", "TAD", 'HDA', 'FINDER', "DND"], fontsize=11, bbox_to_anchor=(0.89, 0.5), loc='lower right', ncol=1, handlelength=1.5, labelspacing=0.5, frameon=False)
    #ax1.spines['top'].set_visible(False)
    #ax1.spines['right'].set_visible(False)


    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.3)
    plt.show()


    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # #lsc_nodes = max(nx.strongly_connected_components(g), key=len)
    # #draw_topology(g, [], nodes, lsc_nodes, ax[0][0], "Original Network")
    # lsc_nodes = max(nx.strongly_connected_components(g), key=len)
    #
    # scc_with_sizes = [(comp, len(comp)) for comp in nx.strongly_connected_components(g)]
    # print("scc_with_sizes", scc_with_sizes)
    #
    # draw_topology(g, [], nodes, lsc_nodes, ax, "Original Network")
    # ax.axis('off')
    # plt.tight_layout()
    # fig.subplots_adjust(left =0, right=1, top =1, bottom=0)
    # plt.show()


    scc_with_sizes = [(comp, len(comp)) for comp in nx.strongly_connected_components(g)]
    print("scc_with_sizes", scc_with_sizes)
    lsc_nodes_all = []
    for lsc_nodes, length in scc_with_sizes:
        if length >= 3:
            print(length)
            lsc_nodes_all.append((lsc_nodes))
    lsc_nodes_all.sort(key=len, reverse=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_topology(g, [], nodes, lsc_nodes_all, ax, "Original Network")
    ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(left =0, right=1, top =1, bottom=0)
    plt.show()





def ATD_show():

    G_0 = nx.DiGraph()
    nodes = list(range(20))
    G_0.add_nodes_from(nodes)
    # 手动指定边
    edges = [
        (0, 3), (0, 7), (0, 11), (0, 15), (0, 18),
        (1, 2), (1, 6), (1, 10), (1, 14), (2, 1),
        (2, 5), (2, 9), (3, 0), (3, 4), (3, 8),
        (4, 2), (4, 7), (5, 3), (5, 12), (6, 1),
        (6, 16), (7, 5), (7, 13), (8, 4), (8, 17),
        (9, 6), (9, 19), (10, 8), (11, 9), (12, 10),
    ]
    G_0.add_edges_from(edges)    # 添加边到图中
    G = G_0.copy()



    # 转换为邻接矩阵
    A = nx.to_numpy_array(G)
    pos = {0: (-0.06, -0.04), 1: (-0.18, 0.31), 2: (0.15, 0.2), 3: (-0.19, -0.2), 4: (0.16, 0.06), 5: (-0.04, -0.27), 6: (-0.05, 0.36), 7: (0.12, -0.18), 8: (-0.25, -0.03), 9: (-0.06, 0.18),
           10: (-0.24, 0.21), 11: (-0.17, 0.06), 12: (0.1, -0.05), 13: (-0.05, -0.15), 14: (-0.15, 0.19), 15: (-0.03, 0.12), 16: (0.1, 0.3), 17: (-0.22, 0.12), 18: (0.06, 0.15), 19: (-0.18, -0.08)}
    fig, ax1 = plt.subplots(1, 1, figsize=(4.2, 4))

    edge_rad = {(0, 3): 0.15, (0, 7): -0.15, (0, 11): -0.15, (0, 15): 0.15, (0, 18): 0.15, (1, 2): -0.15, (1, 6): 0.15,
     (1, 10): 0.15, (1, 14): -0.15, (2, 1): -0.15, (2, 5): 0.15, (2, 9): 0.15, (3, 0): 0.15, (3, 4): 0.15,
     (3, 8): -0.15, (4, 2): 0.15, (4, 7): -0.15, (5, 3): -0.15, (5, 12): 0.15, (6, 1): 0.15, (6, 16): -0.15,
     (7, 5): -0.15, (7, 13): -0.15, (8, 4): 0.3, (8, 17): 0.15, (9, 6): -0.15, (9, 19): 0.15, (10, 8): 0.15,
     (11, 9): -0.15, (12, 10): -0.15}

    nx.draw_networkx_nodes(G_0, pos, node_color='#A2B5CD', node_size=300, ax=ax1)
    for u, v in G_0.edges():
        rad = edge_rad[(u, v)]  # 已经算好 ±0.15
        nx.draw_networkx_edges(G_0, pos,
                               edgelist=[(u, v)],
                               arrows=True,
                               arrowsize=12,
                               width=1.5,
                               edge_color='#404040',
                               connectionstyle=f'arc3,rad={rad}',
                               ax=ax1)
    ax1.set_axis_off()
    ax1.text(0, 1.0, 'a', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    plt.show()




    h = get_levels(A)
    h = h.reshape((h.shape[0], 1))
    h_h = h.T - h
    h_h = np.where(h_h >= 0, 1, h_h)
    h_h = np.power((h_h - 1), 2)
    h_h_A = h_h * A
    up = h_h_A.sum(axis=1) + h_h_A.sum(axis=0)
    fig, ax2 = plt.subplots(1, 1, figsize=(4, 5))
    pos2 = {0: (0, -1), 1: (0.01, 2.82), 2: (0.03, 1.65), 3: (0, 0.50), 4: (-0.03, 1.86), 5: (0.05, 1.77), 6: (0.05, 4.34),
           7: (0.05, 2.80), 8: (0, 3.33), 9: (-0.04, 3.51),
           10: (-0.05, 4.46), 11: (-0.05, 1.47), 12: (0.03, 3.03), 13: (-0.02, 5.40), 14: (0.02, 5.52), 15: (-0.02, 3),
           16: (0, 6.84), 17: (-0.05, 6.33), 18: (-0.05, 3), 19: (0.05, 6.51)}
    edge_rad2 = {(0, 3): 0.15, (0, 7): 0.15, (0, 11): -0.2, (0, 15): -0.1, (0, 18): -0.1, (1, 2): -0.15, (1, 6): 0.15,
                (1, 10): -0.1, (1, 14): 0.15, (2, 1): -0.15, (2, 5): 0.15, (2, 9): 0.15, (3, 0): 0.2, (3, 4): 0-.15,
                (3, 8): -0.15, (4, 2): 0.15, (4, 7): -0.15, (5, 3): -0.15, (5, 12): 0.15, (6, 1): 0.15, (6, 16): 0.15,
                (7, 5): -0.15, (7, 13): -0.15, (8, 4): 0.15, (8, 17): -0.15, (9, 6): -0.15, (9, 19): 0.15, (10, 8): -0.15,
                (11, 9): -0.15, (12, 10): -0.15}
    red_edges = []
    black_edges = []
    for u, v in G_0.edges():
        if h[u] > h[v]:
            red_edges.append((u, v))
        else:
            black_edges.append((u, v))

    nx.draw_networkx_nodes(G_0, pos2, node_color='#A2B5CD', node_size=300, ax=ax2)
    nx.draw_networkx_nodes(G_0, pos2, node_color='#A2B5CD', node_size=300, ax=ax2)
    for u, v in black_edges:
        rad = edge_rad2[(u, v)]
        nx.draw_networkx_edges(G_0, pos2,
                               edgelist=[(u, v)],
                               arrows=True,
                               arrowsize=12,
                               width=1.5,
                               edge_color='#404040',
                               connectionstyle=f'arc3,rad={rad}',
                               ax=ax2)
    for u, v in red_edges:
        rad = edge_rad2[(u, v)]  # 已经算好 ±0.15
        nx.draw_networkx_edges(G_0, pos2,
                               edgelist=[(u, v)],
                               arrows=True,
                               arrowsize=12,
                               width=2,
                               edge_color='red',
                               connectionstyle=f'arc3,rad={rad}',
                               ax=ax2)
    ax2.set_axis_off()
    plt.show()




    up_min, up_max = up.min(), up.max()
    log_up = np.log(up - up_min + 1)
    log_max = np.log(up_max - up_min + 1)
    node_sizes = 100 + 500 * (log_up / log_max)
    fig, ax3 = plt.subplots(1, 1, figsize=(4.2, 4))
    nx.draw_networkx_nodes(G_0, pos, node_color='#A2B5CD', node_size=node_sizes, ax=ax3)
    ax3.set_axis_off()
    max_node = int(np.argmax(up))  # 节点编号
    x, y = pos[max_node]  # 坐标
    ax3.plot(x, y, 'rx',  markersize=node_sizes[max_node] / 20, markeredgewidth=3)
    for u, v in G_0.edges():
        rad = edge_rad[(u, v)]  # 已经算好 ±0.15
        nx.draw_networkx_edges(G_0, pos,
                               edgelist=[(u, v)],
                               arrows=True,
                               arrowsize=12,
                               width=1.5,
                               edge_color='#404040',
                               connectionstyle=f'arc3,rad={rad}',
                               ax=ax3)
    ax3.text(0, 1.0, 'c', transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    plt.show()




    fig, ax4 = plt.subplots(1, 1, figsize=(4.2, 4))
    G_removed = G_0.copy()
    G_removed.remove_node(max_node)
    lscc = max(nx.strongly_connected_components(G_removed), key=len)
    print(lscc)
    node_colors = ['#606060' if n == max_node else '#A2B5CD' for n in G_0.nodes()]
    nx.draw_networkx_nodes(G_0, pos, node_color=node_colors, node_size=node_sizes, ax=ax4)
    for u, v in G_0.edges():
        if u != max_node and v != max_node:
            rad = edge_rad[(u, v)]
            nx.draw_networkx_edges(G_0, pos,
                                   edgelist=[(u, v)],
                                   arrows=True,
                                   arrowsize=12,
                                   width=1.5,
                                   edge_color='#404040',
                                   connectionstyle=f'arc3,rad={rad}',
                                   ax=ax4)
    ax4.set_axis_off()
    ax4.text(0, 1.0, 'd', transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    plt.show()






def draw_synthetic(network_name):
    if network_name=='ER':
        unit_topics = ["ATAD", "TAD",  'FINDER', "DND", 'HDA', "HD", 'ID', "OD"]
        dir = "../Data/Networks/ER_network/"
        file_pre = "ER"
        fileNames = findfile(dir, file_pre)
        fileNames=fileNames[30:]+fileNames[:30]
        network_names = fileNames  # 调整为参数从小到大的顺序
        lamb = [r'$ER_{\langle k \rangle=3}$', r'$ER_{\langle k \rangle=6}$', r'$ER_{\langle k \rangle=9}$', r'$ER_{\langle k \rangle=12}$']

        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)
        D = ['3','6','9','12']
        color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30
        for j, col in enumerate(axes[0]):
            atd = np.zeros(1000)
            back = np.zeros(1000)  # 初始化不同方法的平均GSCC曲线
            degree = np.zeros(1000)
            adpdegree = np.zeros(1000)
            finder = np.zeros(1000)
            dnd = np.zeros(1000)
            id = np.zeros(1000)
            od = np.zeros(1000)

            for epoch in range(num):  # 计算平均GSCC曲线
                network_name = network_names[j * num + epoch]
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_ATD.npy')
                atd += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_TAD.npy')
                back += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_FD.npy')
                finder += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_DND.npy')
                dnd += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_HDA.npy')
                adpdegree += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_HD.npy')
                degree += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_ID.npy')
                id += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_OD.npy')
                od += temp / (num)

            x = [_ / len(back) for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            col.set_title(lamb[j], y=0.8, x=0.5)
            col.plot(x, atd, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, back, color="#888888", lw=1.2)
            col.plot(x, finder, color="#00FF00", lw=1.2)
            col.plot(x, dnd, color="#80A6E2", lw=1.2)
            col.plot(x, adpdegree, color="#FBDD85", lw=1.2)
            col.plot(x, degree, color="#00FFFF", lw=1.2)
            col.plot(x, id, color="#F46F43", lw=1.2)
            col.plot(x, od, color="#CF3D3E", lw=1.2)
            col.tick_params(labelsize=10)
            col.set_ylim(-0.05, 1.05)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=12, ha='center')
        fig.text(0.02, 0.75, 'GSCC', va='center', fontsize=12, rotation='vertical')
        col.legend(["ATAD", "TAD",  'FINDER', "DND", 'HDA', "HD", 'ID', "OD"], prop={'size': 10},
                   bbox_to_anchor=(0.35, 1.17), loc=1, ncol=9, borderaxespad=0)

        for j, col in enumerate(axes[1]):
            atd_auc = 0
            back_auc = 0
            finder_auc = 0
            DND_auc = 0
            adpDegree_auc = 0
            degree_auc = 0
            id_auc = 0
            od_auc = 0

            atda = []
            ba = []
            fa = []
            dnda = []
            aa = []
            da = []
            ia = []
            oa = []


            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]

                atd = np.load('../Data/ND_results/NPY/' + name + '_ATD.npy')
                #atd = atd / atd[0]
                back = np.load('../Data/ND_results/NPY/' + name + '_TAD.npy')
                #back = back / back[0]
                finder = np.load('../Data/ND_results/NPY/' + name + '_FD.npy')
                #finder = finder / finder[0]
                dnd = np.load('../Data/ND_results/NPY/' + name + '_DND.npy')
                #dnd = dnd / dnd[0]
                adpdegree = np.load('../Data/ND_results/NPY/' + name + '_HDA.npy')
                #adpdegree = adpdegree / adpdegree[0]
                degree = np.load('../Data/ND_results/NPY/' + name + '_HD.npy')
                #degree = degree / degree[0]
                id = np.load('../Data/ND_results/NPY/' + name + '_ID.npy')
                #id = id / id[0]
                od = np.load('../Data/ND_results/NPY/' + name + '_OD.npy')
                #od = od / od[0]


                atd[atd < 0.1] = 0
                back[back < 0.1] = 0
                finder[finder < 0.1] = 0
                dnd[dnd < 0.1] = 0
                adpdegree[adpdegree < 0.1] = 0
                degree[degree < 0.1] = 0
                id[id < 0.1] = 0
                od[od < 0.1] = 0

                max_val = max(back)
                max_val = 1     #AUC
                print(max_val)
                if max_val >= 0.1:
                    atd_auc += atd.sum() / (1000 * max_val)
                    back_auc += back.sum() / (1000 * max_val)
                    finder_auc += finder.sum() / (1000 * max_val)
                    DND_auc += dnd.sum() / (1000 * max_val)
                    adpDegree_auc += adpdegree.sum() / (1000 * max_val)
                    degree_auc += degree.sum() / (1000 * max_val)
                    id_auc += id.sum() / (1000 * max_val)
                    od_auc += od.sum() / (1000 * max_val)

                    atda.append(atd.sum() / (1000 * max_val))
                    ba.append(back.sum() / (1000 * max_val))
                    fa.append(finder.sum() / (1000 * max_val))
                    dnda.append(dnd.sum() / (1000 * max_val))
                    aa.append(adpdegree.sum() / (1000 * max_val))
                    da.append(degree.sum() / (1000 * max_val))
                    ia.append(id.sum() / (1000 * max_val))
                    oa.append(od.sum() / (1000 * max_val))

                else:
                    atd_auc += 0
                    back_auc += 0
                    finder_auc += 0
                    DND_auc += 0
                    adpDegree_auc += 0
                    degree_auc += 0
                    id_auc += 0
                    od_auc += 0

                    atda.append(0)
                    ba.append(0)
                    fa.append(0)
                    dnda.append(0)
                    aa.append(0)
                    da.append(0)
                    ia.append(0)
                    oa.append(0)


            std = [np.std(atda, ddof=1), np.std(ba, ddof=1), np.std(fa, ddof=1), np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1),
                   np.std(ia, ddof=1), np.std(oa, ddof=1)]
            temp = [atd_auc / num, back_auc / num, finder_auc / num, DND_auc / num, adpDegree_auc / num, degree_auc / num,
                    id_auc / num, od_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics, fontsize=10, rotation=60)
            if j==0:
                col.set_ylim(0, 0.05)
            else:
                col.set_ylim(0, 0.4)
            font1 = {
                'weight': 'normal',
                'size': 12,
            }
            col.set_title(r'$ER_{\langle k \rangle=' + D[j] + '}$', font1, y=1.0, x=0.5)
        fig.text(0.02, 0.25, 'AUC', va='center', fontsize=12, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'a', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        axes[1][0].text(-0.27, 1.1, 'b', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        plt.show()


    if network_name=='SF':
        unit_topics = ["ATAD", "TAD",  'FINDER', "DND", 'HDA', "HD", 'ID', "OD"]
        dir = "../Data/Networks/SF_network/"
        file_pre = "SF_1000"
        fileNames = findfile(dir, file_pre)
        network_names = findfile(dir, file_pre)
        lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.5}$', r'$SF_{\lambda=2.8}$', r'$SF_{\lambda=3.2}$']
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)

        D=['2.2','2.5','2.8','3.2']
        color = ['#403990', "#888888",  "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30


        for j, col in enumerate(axes[0]):

            atd = np.zeros(1000)
            back = np.zeros(1000)  # 初始化不同方法的平均GSCC曲线
            degree = np.zeros(1000)
            adpdegree = np.zeros(1000)
            finder = np.zeros(1000)
            dnd = np.zeros(1000)
            id = np.zeros(1000)
            od = np.zeros(1000)

            for epoch in range(num):  # 计算平均GSCC曲线
                network_name = network_names[j * num + epoch]
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_ATD.npy')
                atd += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_TAD.npy')
                back += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_FD.npy')
                finder += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_DND.npy')
                dnd += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_HDA.npy')
                adpdegree += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_HD.npy')
                degree += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_ID.npy')
                id += temp / (num)
                temp = np.load('../Data/ND_results/NPY/' + network_name + '_OD.npy')
                od += temp / (num)


            x = [_ / len(back) for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            col.set_title(lamb[j], y=0.8, x=0.5)
            col.plot(x, atd, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, back, color="#888888", lw=1.2)
            col.plot(x, finder, color="#00FF00", lw=1.2)
            col.plot(x, dnd, color="#80A6E2", lw=1.2)
            col.plot(x, adpdegree, color="#FBDD85", lw=1.2)
            col.plot(x, degree, color="#00FFFF", lw=1.2)
            col.plot(x, id, color="#F46F43", lw=1.2)
            col.plot(x, od, color="#CF3D3E", lw=1.2)
            col.tick_params(labelsize=10)
            col.set_ylim(-0.05, 1.05)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=12, ha='center')
        fig.text(0.02, 0.75, 'GSCC', va='center', fontsize=12, rotation='vertical')
        col.legend(["ATAD", "TAD",  'FINDER', "DND", 'HDA', "HD", 'ID', "OD"], prop={'size': 10},
                   bbox_to_anchor=(0.35, 1.17), loc=1, ncol=9, borderaxespad=0)

        for j, col in enumerate(axes[1]):
            atd_auc = 0
            back_auc = 0
            finder_auc = 0
            DND_auc = 0
            adpDegree_auc = 0
            degree_auc = 0
            id_auc = 0
            od_auc = 0

            atda = []
            ba = []
            fa = []
            dnda = []
            aa = []
            da = []
            ia = []
            oa = []

            # 用来收集 30 次实验的 AUC
            data_dict = {alg: [] for alg in unit_topics}

            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                atd = np.load('../Data/ND_results/NPY/' + name + '_ATD.npy')
                back = np.load('../Data/ND_results/NPY/' + name + '_TAD.npy')
                finder = np.load('../Data/ND_results/NPY/' + name + '_FD.npy')
                dnd = np.load('../Data/ND_results/NPY/' + name + '_DND.npy')
                adpdegree = np.load('../Data/ND_results/NPY/' + name + '_HDA.npy')
                degree = np.load('../Data/ND_results/NPY/' + name + '_HD.npy')
                id = np.load('../Data/ND_results/NPY/' + name + '_ID.npy')
                od = np.load('../Data/ND_results/NPY/' + name + '_OD.npy')


                atd[atd < 0.1] = 0
                back[back < 0.1] = 0
                finder[finder < 0.1] = 0
                dnd[dnd < 0.1] = 0
                adpdegree[adpdegree < 0.1] = 0
                degree[degree < 0.1] = 0
                id[id < 0.1] = 0
                od[od < 0.1] = 0

                max_val = max(back)
                max_val=1   #auc
                print(max_val)
                if max_val >= 0.1:
                    atd_auc += atd.sum() / (1000 * max_val)
                    back_auc += back.sum() / (1000 * max_val)
                    finder_auc += finder.sum() / (1000 * max_val)
                    DND_auc += dnd.sum() / (1000 * max_val)
                    adpDegree_auc += adpdegree.sum() / (1000 * max_val)
                    degree_auc += degree.sum() / (1000 * max_val)
                    id_auc += id.sum() / (1000 * max_val)
                    od_auc += od.sum() / (1000 * max_val)

                    atda.append(atd.sum() / (1000 * max_val))
                    ba.append(back.sum() / (1000 * max_val))
                    fa.append(finder.sum() / (1000 * max_val))
                    dnda.append(dnd.sum() / (1000 * max_val))
                    aa.append(adpdegree.sum() / (1000 * max_val))
                    da.append(degree.sum() / (1000 * max_val))
                    ia.append(id.sum() / (1000 * max_val))
                    oa.append(od.sum() / (1000 * max_val))
                else:
                    atd_auc += 0
                    back_auc += 0
                    finder_auc += 0
                    DND_auc += 0
                    adpDegree_auc += 0
                    degree_auc += 0
                    id_auc += 0
                    od_auc += 0

                    atda.append(0)
                    ba.append(0)
                    fa.append(0)
                    dnda.append(0)
                    aa.append(0)
                    da.append(0)
                    ia.append(0)
                    oa.append(0)



            std = [np.std(atda, ddof=1), np.std(ba, ddof=1), np.std(fa, ddof=1), np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1),
                   np.std(ia, ddof=1), np.std(oa, ddof=1)]
            temp = [atd_auc / num, back_auc / num, finder_auc / num, DND_auc / num, adpDegree_auc / num, degree_auc / num,
                    id_auc / num, od_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics,fontsize=10, rotation=60)
            col.set_ylim(0, 0.4)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.02, 0.25, 'AUC', va='center', fontsize=12, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'a', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        axes[1][0].text(-0.27, 1.1, 'b', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        plt.show()







def draw_real():

    network_names = [
        ['FoodWebs_little_rock', "FoodWebs_Weddel_sea", "FoodWebs_reef"],
        ['Neural_net_celegans_neural', 'Neural_rhesus_brain_1'],
        ["Trade_net_trade_basic", 'Trade_net_trade_food'],
        ['p2p-Gnutella06', 'p2p-Gnutella08'],
        ['Social-leader2Inter', "Social_net_moreno_highschool"]
    ]

    # 数据矩阵
    data = np.zeros(shape=(len(network_names), len(network_names)))
    method_colors = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85"]
    # 网络类别标签
    network_labels = ['Food \n Webs', 'Neural', 'Trade', 'P2P', 'Social']
    method_labels = ['ATAD', 'TAD', 'HDA', 'FINDER', 'DND']

    # 填充数据矩阵
    for col, network_class in enumerate(network_names):
        for epoch, network in enumerate(network_class):
            print(network_class, "内的第", epoch, "号网络")
            print('网络名称：', network)
            atd = np.load('../Data/ND_results/NPY/' + network + '_ATD.npy')
            back = np.load('../Data/ND_results/NPY/' + network + '_TAD.npy')
            finder = np.load('../Data/ND_results/NPY/' + network + '_FD.npy')
            dnd = np.load('../Data/ND_results/NPY/' + network + '_DND.npy')
            adpdegree = np.load('../Data/ND_results/NPY/' + network + '_HDA.npy')

            print("AUC", atd.sum() / len(atd))
            data[0, col] += (atd.sum() / len(atd))/len(network_class)
            data[1, col] += (back.sum() / len(atd))/len(network_class)
            data[3, col] += (finder.sum() / len(atd))/len(network_class)
            data[4, col] += (dnd.sum() / len(atd))/len(network_class)
            data[2, col] += (adpdegree.sum() / len(atd))/len(network_class)
            print(data[0, col], data[1, col], data[2, col],data[3, col],data[4, col])

    # 绘制热图
    fig, ax = plt.subplots(figsize=(8, 7))
    # 每列单独归一化并绘制
    for j in range(data.shape[1]):
        cmap = LinearSegmentedColormap.from_list(f'cmap_{j}', ['#403990', "white"])
        col_data = data[:, j]
        col_min, col_max = col_data.min(), col_data.max()
        if col_max == col_min:
            col_normalized = np.zeros_like(col_data)
        else:
            col_normalized = (col_data - col_min) / (col_max - col_min)
        col_matrix = col_normalized.reshape(-1, 1)
        x = [j, j + 1]
        y = np.arange(data.shape[0] + 1)
        ax.pcolormesh(x, y, col_matrix, cmap=cmap, vmin=-0.1, vmax=1.5,
                      edgecolors='white', linewidth=1)

    ax.invert_yaxis()
    # 设置坐标轴
    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=13)
    ax.set_yticklabels(method_labels, fontsize=13)
    ax.text(-0.12, 1.06, chr(97), transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='top', ha='left')
    # 添加数值标注
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5, i + 0.5, f'{data[i, j]:.3f}',
                    ha='center', va='center', color='black', fontsize=15)

    ax.set_title("Mean AUC across network categories", fontsize=14)

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('global', ['#403990', "white"]), norm=norm)
    sm.set_array([])
    #cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.05, shrink=1)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.1, pad=0.05, shrink=1, aspect=15)

    cbar.set_ticks([])  # 移除所有刻度值
    # 添加自定义文字说明（深=小，浅=大）
    cbar.ax.text(0.5, -0.01, 'min: 0', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=12)
    cbar.ax.text(0.5, 1.01, 'max: 1', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    #plt.subplots_adjust(right=1)
    plt.show()



    fig, axes = plt.subplots(2,1, figsize=(5, 7))

    network_names = ["Neural_net_celegans_neural", "p2p-Gnutella06"]
    names = ["Neural_celegans", "p2p06"]
    labels = ["ATAD", "TAD", "HDA", "FINDER", "DND"]
    for i in range(2):
        network_name = network_names[i]
        name = names[i]
        ax = axes[i]

        atd = np.load('../Data/ND_results/NPY/' + network_name + '_ATD.npy')
        back = np.load('../Data/ND_results/NPY/' + network_name + '_TAD.npy')
        finder = np.load('../Data/ND_results/NPY/' + network_name + '_FD.npy')
        dnd = np.load('../Data/ND_results/NPY/' + network_name + '_DND.npy')
        adpdegree = np.load('../Data/ND_results/NPY/' + network_name + '_HDA.npy')



        x = [_ / len(back) for _ in range(len(back))]
        show_n = int(len(back) * 0.6)

        ax.set_title(name, y=1.0, x=0.5, size=14)
        ax.plot(x[:show_n], atd[:show_n], color='#403990', lw=1.8)
        ax.plot(x[:show_n], back[:show_n], color="#888888", lw=1.2)
        ax.plot(x[:show_n], adpdegree[:show_n], color="#FBDD85", lw=1.2)
        ax.plot(x[:show_n], finder[:show_n], color="#00FF00", lw=1.2)
        ax.plot(x[:show_n], dnd[:show_n], color="#80A6E2", lw=1.2)
        ax.tick_params(labelsize=10)
        ax.set_ylabel('GSCC', fontsize=12)
        ax.set_xlabel('Fraction of Nodes Removed', fontsize=12)
        ax.text(-0.13, 1.06, chr(98 + i), transform=ax.transAxes, fontsize=15, fontweight='bold',
                       va='top', ha='left')
    # 在图下方添加图例
    axes[1].legend(labels, loc='lower right', bbox_to_anchor=(0.95, 0.4), ncol=1, fontsize=10, frameon=False)
              #ncol=1, handlelength=1.5, labelspacing=0.25, frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.35)

    plt.show()





def draw_soc():
    dir = "../Data/Networks/large_scale_SF_network/"
    file_pre = "SF_10000_"
    network_names = findfile(dir, file_pre)


    method = ["ATD", "TAD", 'HDA', 'FD', "DND"]
    colors = ['#403990', "#888888", "#FBDD85", "#00FF00", "#80A6E2"]
    x_common = np.linspace(0, 1, 10000)


    fig, ax = plt.subplots(2, 3, figsize=(14, 8))


    # 图a
    network_name = "SF_10000_2.8_13.3"
    atd = np.load('../Data/ND_results/NPY/' + network_name + '_ATD_GSCC.npy')
    back = np.load('../Data/ND_results/NPY/' + network_name + '_TAD_GSCC.npy')
    finder = np.load('../Data/ND_results/NPY/' + network_name + '_FD_GSCC.npy')
    dnd = np.load('../Data/ND_results/NPY/' + network_name + '_DND_GSCC.npy')
    adpdegree = np.load('../Data/ND_results/NPY/' + network_name + '_HDA_GSCC.npy')
    #print(len(atd))
    x = [_ / len(back) for _ in range(len(back))]

    def highlight_max_drop(ax, x, y, color, lw=1.8, highlight_lw=3.2):
        # 计算单步下降值
        drops = np.diff(y)
        max_drop_index = np.argmin(drops)  # 找到最大下降点的索引
        max_drop_x = x[max_drop_index]
        max_drop_y = y[max_drop_index]
        next_drop_y = y[max_drop_index + 1]
        # 绘制整条曲线
        ax.plot(x, y, color=color, lw=lw)
        # 加粗最大单步下降部分
        ax.plot([max_drop_x, x[max_drop_index + 1]], [max_drop_y, next_drop_y], color=color, lw=highlight_lw)

    highlight_max_drop(ax[0][0], x, atd, color='#403990', lw=2.2, highlight_lw=5)
    highlight_max_drop(ax[0][0], x, back, color="#888888", lw=1.8, highlight_lw=5)
    highlight_max_drop(ax[0][0], x, adpdegree, color="#FBDD85", lw=1.8, highlight_lw=5)
    highlight_max_drop(ax[0][0], x, finder, color="#00FF00", lw=1.8, highlight_lw=5)
    highlight_max_drop(ax[0][0], x, dnd, color="#80A6E2", lw=1.8, highlight_lw=5)

    # 手动添加一个虚拟曲线用于生成图例
    ax[0][0].plot([], [], color='#403990', lw=2.2, label='ATAD')
    ax[0][0].plot([], [], color="#888888", lw=1.8, label='TAD')
    ax[0][0].plot([], [], color="#FBDD85", lw=1.8, label='HDA')
    ax[0][0].plot([], [], color="#00FF00", lw=1.8, label='FINDER')
    ax[0][0].plot([], [], color="#80A6E2", lw=1.8, label='DND')

    ax[0][0].tick_params(axis='both', labelsize=12)
    ax[0][0].set_ylabel('GSCC', fontsize=14)
    ax[0][0].set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax[0][0].set_xlim(0, 0.6)
    ax[0][0].text(-0.2, 1.05, 'a', transform=ax[0][0].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax[0][0].legend(fontsize=11,
              bbox_to_anchor=(0.99, 0.5), loc='lower right',
              ncol=1, handlelength=1.5, labelspacing=0.5, frameon=False)





    # 图b
    #dir = "data/"
    network_name = "SF_10000_2.8_13.3"
    #network_name = "SF_1000_2.2_0.1_0.13_10.66"

    atd = np.load('../Data/ND_results/NPY/' + network_name + '_ATD_GSCC_backedge.npy')
    back = np.load('../Data/ND_results/NPY/' + network_name + '_TAD_GSCC_backedge.npy')
    finder = np.load('../Data/ND_results/NPY/' + network_name + '_FD_GSCC_backedge.npy')
    dnd = np.load('../Data/ND_results/NPY/' + network_name + '_DND_GSCC_backedge.npy')
    adpdegree = np.load('../Data/ND_results/NPY/' + network_name + '_HDA_GSCC_backedge.npy')
    print(atd)
    print(back)
    print(finder)
    print(dnd)
    print(adpdegree)
    print(len(atd))
    x = [_ / len(atd) for _ in range(len(atd))]

    ax[0][1].plot(x, atd, color='#403990', lw=2.2)
    ax[0][1].plot(x, back, color="#888888", lw=1.8)
    ax[0][1].plot(x, adpdegree, color="#FBDD85", lw=1.8)
    ax[0][1].plot(x, finder, color="#00FF00", lw=1.8)
    ax[0][1].plot(x, dnd, color="#80A6E2", lw=1.8)
    ax[0][1].tick_params(axis='both', labelsize=12)
    ax[0][1].set_ylabel('backward-link ratio', fontsize=14)
    ax[0][1].set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax[0][1].set_xlim(0, 1)
    ax[0][1].text(-0.2, 1.05, 'b', transform=ax[0][1].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax[0][1].legend(["ATAD", "TAD", 'HDA', 'FINDER', "DND"], fontsize=11,
                    bbox_to_anchor=(0.99, 0.5), loc='lower right',
                   ncol=1, handlelength=1.5, labelspacing=0.5, frameon=False)



    # 图c
    network_name = "SF_10000_2.8_13.3"
    atd = np.load('../Data/ND_results/NPY/' + network_name + '_ATD_GSCC_F.npy')
    back = np.load('../Data/ND_results/NPY/' + network_name + '_TAD_GSCC_F.npy')
    finder = np.load('../Data/ND_results/NPY/' + network_name + '_FD_GSCC_F.npy')
    dnd = np.load('../Data/ND_results/NPY/' + network_name + '_DND_GSCC_F.npy')
    adpdegree = np.load('../Data/ND_results/NPY/' + network_name + '_HDA_GSCC_F.npy')
    atd = np.nan_to_num(atd, nan=0.0)
    back = np.nan_to_num(back, nan=0.0)
    finder = np.nan_to_num(finder, nan=0.0)
    dnd = np.nan_to_num(dnd, nan=0.0)
    adpdegree = np.nan_to_num(adpdegree, nan=0.0)
    print(atd[5500:5580])
    print(back[5500:5580])
    print(adpdegree[5500:5580])
    #print(dnd[5500:5700])
    print(finder[5500:5580])


    print(len(atd))
    print(len(atd))
    x = [_ / len(back) for _ in range(len(back))]

    ax[0][2].plot(x, atd, color='#403990', lw=2.2)
    ax[0][2].plot(x, back, color="#888888", lw=1.8)
    ax[0][2].plot(x, adpdegree, color="#FBDD85", lw=1.8)
    ax[0][2].plot(x, finder, color="#00FF00", lw=1.8)
    ax[0][2].plot(x, dnd, color="#80A6E2", lw=1.8)
    ax[0][2].tick_params(axis='both', labelsize=12)
    ax[0][2].set_ylabel('F$_0$', fontsize=14)
    ax[0][2].set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax[0][2].set_xlim(0, 1)
    ax[0][2].text(-0.2, 1.05, 'c', transform=ax[0][2].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax[0][2].legend(["ATAD", "TAD", 'HDA', 'FINDER', "DND"], fontsize=11,
                    bbox_to_anchor=(0.99, 0.5), loc='lower right',
                    ncol=1, handlelength=1.5, labelspacing=0.5, frameon=False)






    #图d
    network_name = "SF_10000_2.8_13.3"
    g = nx.read_graphml("../Data/Networks/" + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    sccs = list(nx.strongly_connected_components(g))
    largest_scc = max(sccs, key=len)
    new_g = g.subgraph(largest_scc).copy()
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)

    color_map = {"ATD": "#403990", "TAD": "#888888", 'HDA':"#FBDD85", 'FD':"#00FF00", "DND": "#80A6E2"}

    # 3) 统一坐标范围（按需要再调）
    xlim = (1, 3e3)
    ylim = (1e-3, 1)
    ax[1][0].set_title("Avalanche size CCDF", fontsize=14, y=0.88)

    # 幂律分布拟合函数
    def power_law(x, alpha, c):
        return c * x ** (-alpha)

    def fit_power_law(x_data, y_data):
        # 只对正数进行拟合
        mask = (x_data > 0) & (y_data > 0)
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        try:
            # 使用对数空间的线性回归
            log_x = np.log(x_fit)
            log_y = np.log(y_fit)
            # 线性拟合
            slope, intercept = np.polyfit(log_x, log_y, 1)
            alpha = -slope  # 指数
            c = np.exp(intercept)  # 系数

            return alpha, c, x_fit
        except:
            return None, None, x_fit

    for m in ["ATD", "TAD", 'HDA', 'FD', "DND"]:
        back = np.load(f'../Data/ND_results/NPY/{network_name}_{m}_GSCC.npy') * node_count
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta[back_deta != 0]

        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        back_deta_raw = np.repeat(back_unique, np.round(back_p * len(back_deta)).astype(int))

        # 画散点
        print("pdf", back_p)
        ccdf = np.concatenate(([1.], 1 - np.cumsum(back_p)))[:-1]
        print("ccdf", ccdf)
        if m == "ATD":
            name = "ATAD"
        elif m == "FD":
            name = "FINDER"
        else:
            name = m

            # 散点图标签
        ax[1][0].scatter(back_unique, ccdf, color=color_map[m], s=20, alpha=0.6, label=name)

    for m in ["ATD"]:
        back = np.load(f'../Data/ND_results/NPY/{network_name}_{m}_GSCC.npy') * node_count
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta[back_deta != 0]
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        back_p = back_counts / back_counts.sum()
        ccdf = np.concatenate(([1.], 1 - np.cumsum(back_p)))[:-1]
        # 拟合幂律分布
        alpha, c, x_fit = fit_power_law(back_unique, ccdf)
        if alpha is not None:
            # 生成拟合曲线
            x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 100)
            y_fit = power_law(x_smooth, alpha, c)

            # 画拟合曲线
            #ax.plot(x_smooth, y_fit, color=color_map[m], linestyle='--', linewidth=2, alpha=0.8, label=f'{m} fit: α={alpha:.2f}')
            ax[1][0].plot(x_smooth, y_fit, color=color_map[m], linestyle='--', linewidth=2, alpha=0.8)
    # 统一坐标
    ax[1][0].set_xscale("log")
    ax[1][0].set_yscale("log")
    ax[1][0].set_xlim(*xlim)
    ax[1][0].set_ylim(*ylim)
    ax[1][0].set_ylabel(r"$P(S \geq s)$", rotation=90, labelpad=10, fontsize=14)
    ax[1][0].set_xlabel('size s', fontsize=14)
    ax[1][0].tick_params(axis='both', labelsize=12)  # 同时设置x和y
    ax[1][0].legend(bbox_to_anchor=(0.99, 0.4), loc='lower right', fontsize=11, frameon=False)
    #plt.suptitle('Difference of deta_GSCC in STLD', fontsize=14)

    ax[1][0].text(-0.2, 1.05,  'd', transform=ax[1][0].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')




    # color_map = {"ATD": "#403990", "TAD": "#888888", 'HDA': "#FBDD85", 'FD': "#00FF00", "DND": "#80A6E2"}
    # method_names = {"ATD": "ATAD", "FD": "DNetKey"}
    # # 坐标范围
    # xlim = (1e-4, 1)  # 归一化后范围调整为比例
    # ylim = (1e-3, 1)
    # xlim = (1, 5e3)
    # ylim = (1e-3, 1)
    # def process_single_network(network_name_local, method):
    #     print(network_name_local)
    #     # 加载网络
    #     g = nx.read_graphml(dir + network_name_local)
    #     g.remove_edges_from(nx.selfloop_edges(g))
    #     largest_scc = max(nx.strongly_connected_components(g), key=len)
    #     g = g.subgraph(largest_scc).copy()
    #     node_count = g.number_of_nodes()
    #
    #     # 加载并处理数据
    #     back = np.load(f'result/{network_name_local}_{method}_GSCC.npy') * node_count
    #     back_deta = -np.diff(back)
    #     back_deta = np.round(back_deta, 5)
    #     back_deta = back_deta[back_deta != 0]
    #
    #     #return back_deta / node_count
    #     return back_deta
    #
    # # 为每种方法聚合所有网络的数据
    # all_data = {m: [] for m in method}
    #
    # # 遍历所有网络和方法，收集数据
    # for network in network_names:
    #     for m in method:
    #         normalized_data = process_single_network(network, m)
    #         if len(normalized_data) > 0:
    #             all_data[m].extend(normalized_data.tolist())
    #
    # # 绘制每种方法的平均CCDF
    # for m in method:
    #     if not all_data[m]:
    #         continue
    #
    #     # 合并所有网络的归一化数据
    #     combined_data = np.array(all_data[m])
    #
    #     # 计算CCDF
    #     unique_vals, counts = np.unique(combined_data, return_counts=True)
    #     probs = counts / len(combined_data)
    #     ccdf = np.concatenate(([1.], 1 - np.cumsum(probs)))[:-1]
    #
    #     # 名称映射
    #     name = method_names.get(m, m)
    #
    #     # 散点图
    #     ax[1][0].scatter(unique_vals, ccdf, color=color_map[m], s=30, alpha=0.6, label=name)
    #
    #     # 幂律拟合（可选）
    #     def fit_power_law(x_data, y_data):
    #         mask = (x_data > 0) & (y_data > 0)
    #         log_x, log_y = np.log(x_data[mask]), np.log(y_data[mask])
    #         slope, intercept = np.polyfit(log_x, log_y, 1)
    #         return -slope, np.exp(intercept), x_data[mask]
    #
    #     alpha, c, x_fit = fit_power_law(unique_vals, ccdf)
    #     if alpha is not None:
    #         x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 100)
    #         if m == "ATD":
    #             ax[1][0].plot(x_smooth, c * x_smooth ** (-alpha), '--', color=color_map[m],
    #                     alpha=0.8, linewidth=2)
    #
    # # 美化
    # ax[1][0].set_xscale("log")
    # ax[1][0].set_yscale("log")
    # ax[1][0].set_xlim(*xlim)
    # ax[1][0].set_ylim(*ylim)
    # ax[1][0].set_xlabel('Relative avalanche size s/N', fontsize=14)
    # ax[1][0].set_ylabel(r"$P(S \geq s)$", fontsize=14)
    # ax[1][0].legend(loc='best', fontsize=11, frameon=False)
    # ax[1][0].tick_params(axis='both', labelsize=12)





    # 图e
    all_delta_curves = {m: [] for m in method}
    def diff_vec(v):
        return v[:-1] - v[1:]
    for network_name in network_names:
        for m in method:
            data = np.load(f'../Data/ND_results/NPY/{network_name}_{m}_GSCC.npy')
            print(data)
            delta_data = diff_vec(data)
            print(delta_data)
            x_original = np.linspace(0, 1, len(data))
            x_diff = x_original[:-1]
            spl = make_interp_spline(x_diff, delta_data, k=3)
            interpolated = spl(x_common[:-1])
            # 截断：防止插值产生负值
            interpolated = np.maximum(interpolated, 0)
            all_delta_curves[m].append(interpolated)

    for i, m in enumerate(method):
        avg_curve = np.mean(all_delta_curves[m], axis=0)
        if m == "ATD":
            ax[1][1].plot(x_common[:-1], avg_curve, color=colors[i], lw=2.2)
        else:
            ax[1][1].plot(x_common[:-1], avg_curve, color=colors[i], lw=1.8)
    ax[1][1].set_ylim([0, 0.02])
    ax[1][1].set_yticks([0.00, 0.005, 0.01, 0.015, 0.02])
    ax[1][1].set_yticklabels(['0.00', '0.005', '0.01', '0.015', '0.02'])
    ax[1][1].tick_params(axis='both', labelsize=12)
    ax[1][1].set_ylabel('$\Delta GSCC$', fontsize=14)
    ax[1][1].set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax[1][1].text(-0.2, 1.05, 'e', transform=ax[1][1].transAxes, fontsize=15, fontweight='bold',
            va='top', ha='left')
    ax[1][1].legend(["ATAD", "TAD", 'HDA', 'FINDER', "DND"], fontsize=11,
              bbox_to_anchor=(0.99, 0.5), loc='lower right',
              ncol=1, handlelength=1.5, labelspacing=0.5, frameon=False)







    # 图f
    # 保存和加载 p_values 的目录
    save_dir = "../Data/p_values/"

    def draw_fun_avg(path, color):
        all_p_values = []
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边

            sccs = list(nx.strongly_connected_components(g))
            largest_scc = max(sccs, key=len)
            new_g = g.subgraph(largest_scc).copy()
            g = new_g

            node_count = g.number_of_nodes()

            save_path = os.path.join(save_dir, f"{network_name}_{path}_GSCC.npy")
            if os.path.exists(save_path):
                print(f"加载已保存的 p_values: {save_path}")
                p_values = np.load(save_path)
            else:
                back = np.load('result/' + network_name + "_" + path + '_GSCC.npy')
                back = back * node_count
                p_values = []

                for i in range(len(back)):
                    back_deta = -np.diff(back[:i + 1])
                    back_deta = np.round(back_deta, 5)  # 防止多余的小数
                    back_deta = back_deta[back_deta != 0]

                    # 检查数据量是否足够
                    if len(back_deta) < 2:
                        p_values.append(0)  # 如果数据不足，记录为 NaN
                        continue

                    # 计算变化量的频率分布
                    back_unique, back_counts = np.unique(back_deta, return_counts=True)
                    back_p = back_counts / back_counts.sum()
                    back_deta_raw = np.repeat(back_unique, np.round(back_p * len(back_deta)).astype(int))
                    # 拟合截断幂律分布
                    fit = powerlaw.Fit(back_deta_raw, xmin=min(back_deta), discrete=True)
                    # print("alpha", fit.power_law.alpha)
                    # print(f"最优 xmin: {fit.xmin}")
                    # print(f"最优 xmax: {fit.xmax}")
                    # print("D:", fit.power_law.D)

                    # 蒙特卡洛模拟计算 p 值
                    D_obs = fit.power_law.D
                    D_sim = []
                    for _ in range(1000):  # 模拟1000次
                        sim_data = fit.power_law.generate_random(len(back_deta))
                        sim_fit = powerlaw.Fit(sim_data, xmin=min(back_deta), discrete=True)
                        D_sim.append(sim_fit.power_law.D)

                    p_value = np.sum(np.array(D_sim) >= D_obs) / len(D_sim)
                    p_values.append(p_value)
                    print(f"次数:{i}")
                    print(f"KS检验的p值: {p_value:.4f}")

                    """
                    # 判断KS检验是否通过
                    if p_value > 0.05:
                        print("KS检验通过，拟合的截断幂律分布与数据一致")
                    else:
                        print("KS检验未通过，拟合的截断幂律分布与数据不一致")
                    """
                # 保存 p_values
                np.save(save_path, p_values)
                print(f"保存 p_values 到: {save_path}")

            # 归一化横坐标到0-1
            x_original = np.linspace(0, 1, len(p_values))
            spl = make_interp_spline(x_original, p_values, k=3)
            interpolated = spl(x_common)
            all_p_values.append(interpolated)

        # 计算平均和标准差
        avg_p_values = np.mean(all_p_values, axis=0)
        std_p_values = np.std(all_p_values, axis=0)  # 计算标准差
        # 对平均值和标准差进行平滑
        spl_avg = make_interp_spline(x_common, avg_p_values, k=3)
        spl_std = make_interp_spline(x_common, std_p_values, k=3)
        ax[1][2].plot(x_common, avg_p_values, color=color, lw=0.8, alpha=0.2)

        #x_smooth = np.linspace(0, 1, 300)
        #y_smooth = spl_avg(x_smooth)
        #std_smooth = spl_std(x_smooth)
        x_smooth = x_common
        y_smooth = spl_avg(x_smooth)
        std_smooth = spl_std(x_smooth)
        # 绘制阴影区域（代表方差范围）
        ax[1][2].fill_between(x_smooth,
                        y_smooth - std_smooth,
                        y_smooth + std_smooth,
                        color=color, alpha=0.15, linewidth=0)
        if path == 'ATD':
            ax[1][2].plot(x_smooth, y_smooth, label='ATAD', color=color, lw=2.2)
        elif path == 'FD':
            ax[1][2].plot(x_smooth, y_smooth, label='FINDER', color=color, lw=2.0)
        else:
            ax[1][2].plot(x_smooth, y_smooth, label=path, color=color, lw=2.0)

    for method_name, color in zip(method, colors):
        print(method_name)
        draw_fun_avg(method_name, color)

    ax[1][2].set_xlim([0.01, 1])
    ax[1][2].set_ylim([-0.01, 1.05])
    ax[1][2].set_xticks(np.arange(0.1, 1, 0.2))
    ax[1][2].set_xlabel('Fraction of Nodes Removed', fontsize=14)
    ax[1][2].set_ylabel('$p$-value', rotation=90, fontsize=14, labelpad=10)
    ax[1][2].tick_params(axis='both', labelsize=12)  # 同时设置x和y
    #ax[1][2].legend(fontsize=12, bbox_to_anchor=(0.99, 0.55), loc='lower right', ncol=1, handlelength=1.5, labelspacing=0.25, frameon=False)
    back = np.load('../Data/ND_results/NPY/' + network_name + "_" + 'ATD_GSCC.npy')
    x_values = np.linspace(0, 1, len(back))
    ax[1][2].plot(x_values, [0.05] * len(x_values), linestyle='--', color='black')
    ax[1][2].text(-0.18, 0.11, 'y=0.05', transform=ax[1][2].transAxes, fontsize=12, va='top', ha='left')
    ax[1][2].text(-0.2, 1.05, 'f', transform=ax[1][2].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')




    plt.tight_layout()
    fig.subplots_adjust(wspace=0.32, hspace=0.32)
    plt.show()






def draw():
    dir = "../Data/Networks/large_scale_SF_network/"
    file_pre = "SF_10000_"
    network_names = findfile(dir, file_pre)
    print(network_names)


    method = ["ATD", "TAD", 'HDA', 'FD', "DND"]

    x_common = np.linspace(0, 1, 10000)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))



    color_map = {"ATD": "#403990", "TAD": "#888888", 'HDA': "#FBDD85", 'FD': "#00FF00", "DND": "#80A6E2"}
    method_names = {"ATD": "ATAD", "FD": "FINDER"}
    # 坐标范围
    xlim = (1e-4, 1)  # 归一化后范围调整为比例
    ylim = (1e-3, 1)
    xlim = (1, 5e3)
    ylim = (1e-3, 1)
    def process_single_network(network_name_local, method):
        print(network_name_local)
        # 加载网络
        g = nx.read_graphml(dir + network_name_local)
        g.remove_edges_from(nx.selfloop_edges(g))
        largest_scc = max(nx.strongly_connected_components(g), key=len)
        g = g.subgraph(largest_scc).copy()
        node_count = g.number_of_nodes()

        # 加载并处理数据
        back = np.load(f'../Data/ND_results/NPY/{network_name_local}_{method}_GSCC.npy') * node_count
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta[back_deta != 0]

        #return back_deta / node_count
        return back_deta

    # 为每种方法聚合所有网络的数据
    all_data = {m: [] for m in method}

    # 遍历所有网络和方法，收集数据
    for network in network_names:
        for m in method:
            normalized_data = process_single_network(network, m)
            if len(normalized_data) > 0:
                all_data[m].extend(normalized_data.tolist())

    # 绘制每种方法的平均CCDF
    for m in method:
        if not all_data[m]:
            continue

        # 合并所有网络的归一化数据
        combined_data = np.array(all_data[m])

        # 计算CCDF
        unique_vals, counts = np.unique(combined_data, return_counts=True)
        probs = counts / len(combined_data)
        ccdf = np.concatenate(([1.], 1 - np.cumsum(probs)))[:-1]

        # 名称映射
        name = method_names.get(m, m)

        # 散点图
        ax.scatter(unique_vals, ccdf, color=color_map[m], s=30, alpha=0.6, label=name)

        # # 幂律拟合（可选）
        # def fit_power_law(x_data, y_data):
        #     mask = (x_data > 0) & (y_data > 0)
        #     log_x, log_y = np.log(x_data[mask]), np.log(y_data[mask])
        #     slope, intercept = np.polyfit(log_x, log_y, 1)
        #     return -slope, np.exp(intercept), x_data[mask]
        #
        # alpha, c, x_fit = fit_power_law(unique_vals, ccdf)
        # if alpha is not None:
        #     x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 100)
        #     if m == "ATD":
        #         ax.plot(x_smooth, c * x_smooth ** (-alpha), '--', color=color_map[m],
        #                 alpha=0.8, linewidth=2)

    # 美化
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('size s', fontsize=14)
    ax.set_ylabel(r"$P(S \geq s)$", fontsize=14)
    ax.legend(loc='best', fontsize=11, frameon=False)
    ax.tick_params(axis='both', labelsize=12)




    plt.tight_layout()
    fig.subplots_adjust(wspace=0.32)
    plt.show()



if __name__ == '__main__':

    ATD_show()                 # Fig.1

    draw_realexample()         # Fig.2

    draw_synthetic("SF")        # Fig.3
    draw_synthetic("ER")        # Fig.S1

    draw_real()                 # Fig.4

    draw_soc()                  # Fig.5

    draw()                      # Fig.S2

