# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 
import os
import numpy as np
import math
import sys 
import pdb
from collections import defaultdict
import time
from shutil import copyfile 
import gc  
from scipy import sparse
import argparse

saved_model_path = './best_model/' 
dataset_base_path='../data/ml/' 
# parser = argparse.ArgumentParser(description="Please select a similarity to train the model.", add_help=False)
# parser.add_argument("--similarity", default=0.8, required=True, type=float, help="Options: 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
#
# args = parser.parse_args()
similarity = 0.7
users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]#gender

##movieLens-1M 
user_num=6040#user_size
item_num=3952#item_size 
factor_num=64 
batch_size=2048*8


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--runid', type=str, default = "fgada_bpr_ml")
args = parser.parse_args()

run_id=args.runid



path_save_model_base=saved_model_path+run_id
if (os.path.exists(path_save_model_base)):
    print("Model:",run_id)
else:
    print('Model_Save_Path does not exist',path_save_model_base)

training_user_set,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)
testing_user_set,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True)

########################### TESTING ##################################### 
np.seterr(divide='ignore',invalid='ignore')
def get_real_data(idx_pre_s,idx_pre_e):
    all_row = []
    all_col = []
    all_data =[]

    test_row = []
    test_col = []
    test_data =[]

    user_id_test =[]
    one_data_len = idx_pre_e - idx_pre_s
    
    for u_i in range(idx_pre_s,idx_pre_e):   
        user_id_test.append(u_i) 
        if u_i not in testing_user_set:
            continue
        else:
            all_one = list(set(training_user_set[u_i])|set(testing_user_set[u_i]))
        
        test_one = list(testing_user_set[u_i])

        all_len = len(all_one)
        test_len = len(test_one)

        all_row = all_row + [u_i-idx_pre_s]*all_len
        all_col = all_col + all_one
        all_data = all_data + [1]*all_len

        test_row = test_row + [u_i-idx_pre_s]*test_len
        test_col = test_col + test_one
        test_data = test_data + [1]*test_len

    all_idx = sparse.csr_matrix((all_data, (all_row, all_col)), shape=(one_data_len, item_num))    
    test_idx = sparse.csr_matrix((test_data, (test_row, test_col)), shape=(one_data_len, item_num))   
    return all_idx.toarray(),test_idx.toarray()
 
    
    
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def all_metrics(top_k): 
    ndcg_max=[0]*top_k
    ndcg_each=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        ndcg_each[i_topK] = 1.0/math.log(i_topK+2)
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg
        
    print('--------test processing, Top K:',top_k,'-------')
    count, best_hr = 0, 0
    for epoch in ['best']: 
    
        PATH_model_u=path_save_model_base+'/user_emb'+str(similarity)+'.npy'
        user_e = np.load(PATH_model_u)
        PATH_model_i=path_save_model_base+'/item_emb'+str(similarity)+'.npy'
        item_e = np.load(PATH_model_i) 
        
        rank_topk=np.zeros([user_num,top_k],dtype=int)
        
        HR, NDCG = 0, 0
        HR_num, NDCG_num = 0, 0
        test_start_time = time.time()
        # ######test and val###########
        idx_all = []
        len_split = int(user_num/3000)
        for i_g in range(len_split):
            idx_pre_s = i_g*3000
            idx_pre_e = (i_g+1)*3000
            idx_all.append([idx_pre_s,idx_pre_e])
        idx_all.append([idx_pre_e,user_e.shape[0]])
        
        i_pre=0
        for idx_pre_s,idx_pre_e in idx_all:  
            i_pre+=1
            s_time = time.time() 
            pre_all = np.matmul(user_e[idx_pre_s:idx_pre_e,:], item_e.T) 
            all_data,test_data = get_real_data(idx_pre_s,idx_pre_e)
            len_test_data =test_data.sum(-1) 

  
            x1=np.where(all_data>0,-1000,pre_all)  
            x2=np.where(test_data>0,pre_all,-1000) 
            
            pre_rank = np.concatenate((x1,x2),axis=-1)
            del x1, x2,pre_all,all_data,test_data
            gc.collect() 
            
            indices_part = np.argpartition(pre_rank, -top_k)[:,-top_k:] 
            values_part = np.partition(pre_rank, -top_k)[:,-top_k:]
            indices_sort = np.argsort(-values_part)
            indice1=np.take_along_axis(indices_part, indices_sort, axis=-1) 
            
            del pre_rank,indices_part,values_part,indices_sort
            gc.collect() 
            
            indice2 = np.where(indice1>item_num,1,0)  
            rank_topk[idx_pre_s:idx_pre_e,:]=indice1  
        
            
            len_large = np.where(len_test_data<top_k,len_test_data,top_k)
            max_hr = len_large
            len_large_ndcg = len_large-1
            max_ndcg = np.array(ndcg_max)[len_large_ndcg]
            
            hr_topK = indice2.sum(-1)
            ndcg_topK = (indice2*ndcg_each).sum(-1) 
            hr_t1 = hr_topK/max_hr
            hr_t = hr_t1[~np.isnan(hr_t1)] 
            ndcg_t=ndcg_topK/max_ndcg
            e4_time = time.time() - s_time 
            HR+=hr_t.sum(-1)
            NDCG+=ndcg_t.sum(-1)
            HR_num+=hr_t.shape[0]
            NDCG_num+=ndcg_t.shape[0] 
        hr_test=round(HR/HR_num,4)
        ndcg_test=round(NDCG/NDCG_num,4)    
        elapsed_time = time.time() - test_start_time     
        if i_pre<len_split:
            str_print_evl="part user test-"
        else:
            str_print_evl=""
        str_print_evl+=" Top K:"+str(top_k)+" test"+" HR:"+str(hr_test)+' NDCG:'+str(ndcg_test) 
        # print(str_print_evl)
        
        DP_res = dict()
        EO_res = dict()
        for v_id in range(item_num*2):
            DP_res[v_id]=[]
            EO_res[v_id]=[]
        for u_id in range(user_num):
            label_gender = users_features[u_id]
            for v_id in rank_topk[u_id]:
                DP_res[v_id].append(label_gender)
                if v_id >item_num:
                    EO_res[v_id].append(label_gender)
        DP=[] 
        male_num=users_features.sum()
        female_num=(1-users_features).sum() 
        for v_id in DP_res:
            one_data = np.array(DP_res[v_id])
            if len(one_data)<1:
                continue
            res_v = one_data.sum()-(1-one_data).sum()
  
            res2_v = one_data.sum()/len(one_data)-(1-one_data).sum()/len(one_data)
            DP.append(np.abs(res2_v))
        DP_test =round(np.array(DP).mean() ,6)
        
        
        EO=[] 
        male_num=users_features.sum()
        female_num=(1-users_features).sum() 
        for v_id in EO_res:
            one_data = np.array(EO_res[v_id])
            if len(one_data)<1:
                continue
            res_v = one_data.sum()-(1-one_data).sum()
    #         res2_v = one_data.sum()/male_num-(1-one_data).sum()/female_num
            res2_v = one_data.sum()/len(one_data)-(1-one_data).sum()/len(one_data)
            EO.append(np.abs(res2_v)) 
        EO_test =round(np.array(EO).mean(),6)
        elapsed_time = time.time() - test_start_time
        str_print_evl+= '\t DP:'+str(DP_test)+' EO:'+str(EO_test)
        print(str_print_evl)
        

for top_k_v in [10,20,30,40,50]:
    all_metrics(top_k_v)

