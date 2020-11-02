import os
import numpy as np
import pandas
import json
#read file mtx

def readMTXFile(mtxFile):
    lines = open(mtxFile).readlines()
    data_dict={}
    col_tag=False; row_tag = False
    for i in range(len(lines)):
        line = lines[i].rstrip()
        if line.startswith("gamma:"):
            gamma = float(line.split("gamma: ")[-1])
            tmp_list = []
        elif line.startswith("col_memship:"):
            col_tag = True
            continue
        elif line.startswith("run_time:"):
            continue
        elif col_tag == True:
            col_memship = list(map(int, lines[i].strip().split(" ")))
            col_num = len(col_memship)
     
            col_tag = False
        elif line.startswith("row_memship:"):
            row_tag = True
            continue
        elif row_tag == True:
            row_memship = list(map(int, lines[i].strip().split(" ")))
            row_num = len(row_memship)
            heatmap = np.asarray(tmp_list).reshape((col_num, row_num)).T
            data_dict[gamma] = {
                "col_memship":col_memship,
                "row_memship":row_memship,
                "heatmap" : heatmap
            }
            row_tag = False
        else:
            tmp_list.append(float(line.strip()))
    return data_dict


def create_tree(key, data):
    gammas = [i for i in data.keys()] 
    gamma0_mat = data[gammas[0]]['heatmap']
    nrow = len(gamma0_mat)
    ncol = len(gamma0_mat[0])
    sons_dict = {}
    query = 34
    for j in range(len(gammas)):
        gamma = gammas[j]
        m = data[gamma][key]
        new_tree = {}
        if j!=0:
            if key == "row_memship":
                ncol = len(data[gamma]['heatmap'][0]) 
            else:
                nrow = len(data[gamma]['heatmap'])
            new_heatmap = np.zeros((nrow, ncol))
        for i in range(len(m)):
            father = m[i]
            me = i
            faga = f'{father}_{gamma}'
            if (faga not in sons_dict) and (father not in new_tree.keys()):
                new_tree[father]={"name":[f'{father}_{gamma}'], "children":[]}
                sons_dict[f'{father}_{gamma}']=[]
            if j == 0: # the first gamma index
                new_tree[father]["children"].append({"name":[me]}) # leaf nodes
                sons_dict[f'{father}_{gamma}'].append(me) 
                if me == query:
                    print(f'{father}_{gamma}')
                    query = f'{father}_{gamma}'
            else:
                last_gamma = gammas[j-1]
                new_tree[father]["children"].append(tree[me])
                sons_dict[f'{father}_{gamma}'].extend(sons_dict[f'{me}_{last_gamma}'])
                sons_indices = list(set(sons_dict[f'{me}_{last_gamma}']))
                if key=="row_memship":
                    mylist = [data[gamma]['heatmap'][me, :] for ii in range(len(sons_indices))]
                    new_heatmap[sons_indices, ] = np.vstack(mylist) 
                else:
                    mylist = [data[gamma]['heatmap'][:, me] for ii in range(len(sons_indices))]
                    new_heatmap[:, sons_indices] = np.column_stack(mylist)     
        tree = new_tree.copy()
        if j!=0:
            data[gamma]['heatmap'] = new_heatmap
    tree = {"name":"root", "children":[tree[iii] for iii in tree.keys()] }
    return tree, data
