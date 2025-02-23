# -- coding:UTF-8
import gc
import torch
# print(torch.__version__)
import torch.nn as nn 
from sklearn.neighbors import KDTree
import argparse
import os
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm
import math
import sys
import random
import tensorflow as tf
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))
#为特定GPU设置种子，生成随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2022)

 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.autograd as autograd 

from sklearn import metrics
from sklearn.metrics import f1_score
import pdb
import copy 
from collections import defaultdict
import time
import data_utils 
from shutil import copyfile    

dataset_base_path='../data/ml/' 

##movieLens-1M
user_num=6040#user_size
item_num=3952#item_size 
factor_num=64
batch_size=2048*100#10
top_k=20
num_negative_test_val=-1##all

run_id="fgada_bpr_ml"
print('Model:',run_id)
dataset='ml'

path_save_model_base='./best_model/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)



train_dict,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)
test_dict,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True) 
users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]#gender

np.seterr(divide='ignore',invalid='ignore')

class FairData(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,similarity,gcn_user_embs=None,gcn_item_embs=None):
        super(FairData, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors. 
        """ 
        self.users_features = torch.cuda.LongTensor(users_features) 
        self.user_num = user_num
        self.factor_num =factor_num
        self.similarity=similarity
        
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        self.noise_item = nn.Embedding(item_num, factor_num)    
        nn.init.normal_(self.noise_item.weight, std=0.01)  
        
#         self.zero_noise_item = torch.zeros(item_num, factor_num*2).cuda()
        
        self.min_clamp=-1
        self.max_clamp=1


    def build_generator(self,input_shape):
        generator = tf.keras.Sequential()
        generator.add(layers.Dense(64, activation='relu', input_shape=input_shape))
        generator.add(layers.Dense(64, activation='relu'))
        generator.add(layers.Dense(input_shape[0]))
        return generator

    # 定义判别器模型
    def build_discriminator(self,input_shape):
        discriminator = tf.keras.Sequential()
        discriminator.add(layers.Dense(64, activation='relu', input_shape=input_shape))
        discriminator.add(layers.Dense(64, activation='relu'))
        discriminator.add(layers.Dense(1, activation='sigmoid'))
        return discriminator

    def compile_models(self, gen_optimizer, dis_optimizer, loss_fn):
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.loss_fn = loss_fn

    def generate_fake_embeddings(self,embedding1, embedding2, similarity_threshold):
        emb1_len = embedding1.shape[0]
        emb2_len = embedding2.shape[0]
        # print(emb1_len,emb2_len)
        # 将embedding1和embedding2从CUDA设备复制到CPU
        embedding1_cpu = embedding1.cpu()
        embedding2_cpu = embedding2.cpu()

        # 将CPU上的Tensor转换为NumPy数组
        embedding1_np = embedding1_cpu.detach().numpy()
        embedding2_np = embedding2_cpu.detach().numpy()

        # 构建KD树
        kdtree = spatial.cKDTree(embedding1_np)

        # 批量查询相似度
        distances, _ = kdtree.query(embedding2_np, k=1)
        # # 构建KD树
        # kdtree = spatial.cKDTree(embedding1)
        #
        # # 批量查询相似度
        # distances, _ = kdtree.query(embedding2, k=1)
        similarities = 1 / (1 + distances)
        normalized_similarities = (similarities - 0) / (1 - 0)  # 归一化处理，将相似度映射到0到1之间

        # 将NumPy数组转换为PyTorch张量
        normalized_similarities_tensor = torch.from_numpy(normalized_similarities).to(embedding1.device)

        # 筛选出符合条件的emb2索引
        selected_indices = torch.where(normalized_similarities_tensor < similarity_threshold)[0]

        if len(selected_indices) == 0:
            sorted_indices = torch.argsort(normalized_similarities_tensor)[:min(emb1_len, emb2_len)]
        else:
            sorted_indices = selected_indices

            # 计算fake_embeddings
        fake_embeddings = embedding2[sorted_indices]
        # fake_embeddings = (embedding2[sorted_indices] * (1 - normalized_similarities_tensor[sorted_indices]).unsqueeze(
        #     1)) + \
        #                   (embedding1 * normalized_similarities_tensor[sorted_indices].unsqueeze(1).repeat(1,
        #                                                                                                    embedding1.size(
        #                                                                                                        1)))
        # fake_embeddings = fake_embeddings.expand_as(embedding2[sorted_indices])

        fake_tensors = [item.clone().detach().to(device='cuda:0') for item in fake_embeddings]
        fake_embeddings_tensor = torch.stack(fake_tensors)  # 将列表中的张量堆叠成一个更高维度的张量

        fake_embeddings_cpu = fake_embeddings_tensor.cpu().detach().numpy()  # 将 CUDA 设备上的张量转移到主机内存并转换为 NumPy 数组

        # print(fake_embeddings_cpu)
        #     male_end_idx = male_len%avg_len+avg_len
        #     male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1,avg_len, self.factor_num)
        #     male_noise_i_mean = torch.mean(male_noise_i_reshape,axis=1)
        #     male_noise_len = male_noise_i_mean.shape[0]
        #     if male_noise_len > female_len:
        #         female_like = male_noise_i_mean[:female_len]
        #     else:
        #         expand_len = int(female_len/male_noise_len)+1
        #         female_like = male_noise_i_mean.repeat(expand_len,1)[:female_len]0

        # fake_emb_reshape = fake_embeddings.view(-1, 1, self.factor_num)
        # fake_emb_reshape_tensor = fake_emb_reshape.to(embedding1.device)
        #
        # fake_emb_i_mean = torch.mean(fake_emb_reshape_tensor, dim=1)
        # fake_emb_len = fake_emb_i_mean.shape[0]

        fake_emb_reshape=fake_embeddings_cpu.reshape(-1,1, self.factor_num)
        fake_emb_reshape_tensor = torch.from_numpy(fake_emb_reshape)  # 将 NumPy 数组转换为 PyTorch 张量

        fake_emb_i_mean = torch.mean(fake_emb_reshape_tensor, axis=1)  # 在指定维度上计算张量的均值
        # fake_emb_i_mean = torch.mean(fake_emb_reshape,axis=1)
        fake_emb_len = fake_emb_i_mean.shape[0]


        if fake_emb_len >= emb1_len:

            fake_like = fake_emb_i_mean[:emb1_len]
        else:
            expand_len = int(emb1_len/fake_emb_len)+1
            fake_like = fake_emb_i_mean.repeat(expand_len,1)[:emb1_len]

        return fake_like

    # 构建KD树
    # kdtree = KDTree(embedding1_np)
    #
    # fake_embeddings = []
    # for emb2 in embedding2_np:
    #     distances, _ = kdtree.query([emb2], k=1)
    #     similarity = 1 / (1 + distances[0])
    #     normalized_similarity = (similarity - 0) / (1 - 0)  # 归一化处理，将相似度映射到0到1之间
    #     # print(normalized_similarity)
    #     if normalized_similarity < similarity_threshold:
    #         fake_embeddings.append(torch.tensor(emb2).to(embedding1.device))

    # 将NumPy数组转换为CUDA上的PyTorch张量
    # fake_cuda= torch.from_numpy(numpy_array).cuda()fake_cuda = torch.from_numpy(numpy_array).cpu().numpy()
    # def fake_pos(self,male_noise_i_emb, female_noise_i_emb):
    #     male_len = male_noise_i_emb.shape[0]
    #     female_len = female_noise_i_emb.shape[0]
    #
    #     avg_len = 1
    #     male_end_idx = male_len%avg_len+avg_len
    #     male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1,avg_len, self.factor_num)
    #     male_noise_i_mean = torch.mean(male_noise_i_reshape,axis=1)
    #     male_noise_len = male_noise_i_mean.shape[0]
    #     if male_noise_len > female_len:
    #         female_like = male_noise_i_mean[:female_len]
    #     else:
    #         expand_len = int(female_len/male_noise_len)+1
    #         female_like = male_noise_i_mean.repeat(expand_len,1)[:female_len]
    #
    #
    #     female_end_idx = female_len%avg_len+avg_len
    #     female_noise_i_emb_reshape = female_noise_i_emb[:-female_end_idx].reshape(-1,avg_len,self.factor_num)
    #     female_noise_i_mean = torch.mean(female_noise_i_emb_reshape,axis=1)
    #     female_noise_len = female_noise_i_mean.shape[0]
    #     if female_noise_len > male_len:
    #         male_like = female_noise_i_mean[:male_len]
    #     else:
    #         expand_len = int(male_len/female_noise_len)+1
    #         male_like = female_noise_i_mean.repeat(expand_len,1)[:male_len]
    #
    #     return male_like,female_like
        
    def forward(self, adj_pos,u_batch,i_batch,j_batch):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 将模型移动到设备上
        u_batch = u_batch.to(device)
        i_batch =i_batch.to(device)
        j_batch = j_batch.to(device)

        # filter gender
        user_emb = self.embed_user.weight 
        item_emb = self.embed_item.weight
        # noise_emb =  self.embed_item.weight
        # noise_emb = torch.clamp(noise_emb, min=self.min_clamp, max=self.max_clamp)
        # noise_emb += self.noise_item.weight
        similarity=self.similarity
        #get gender attribute
        gender = F.embedding(u_batch, torch.unsqueeze(self.users_features, 1)).reshape(-1)
        male_gender = gender.type(torch.BoolTensor).cuda().to(device)
        female_gender = (1 - gender).type(torch.BoolTensor).cuda().to(device)
        
        u_emb = F.embedding(u_batch,user_emb)
        i_emb = F.embedding(i_batch,item_emb)  
        j_emb = F.embedding(j_batch,item_emb)
        # noise_i_emb2 = F.embedding(i_batch,noise_emb)
        # len_noise = int(i_emb.size()[0]*0.4)
        # add_emb = torch.cat((i_emb[:-len_noise],noise_i_emb2[-len_noise:]),0)
        #
        # noise_j_emb2 = F.embedding(j_batch,noise_emb)
        # len_noise = int(j_emb.size()[0]*0.4)
        # add_emb_j = torch.cat((noise_j_emb2[-len_noise:],j_emb[:-len_noise]),0)

        #according gender attribute, selecting embebdding

        male_i_batch = torch.masked_select(i_batch, male_gender)
        female_i_batch = torch.masked_select(i_batch, female_gender)
        male_i_emb = F.embedding(male_i_batch,item_emb)
        female_i_emb = F.embedding(female_i_batch,item_emb)
        male_i_emb = male_i_emb.to('cuda:0')
        female_i_emb = female_i_emb.to('cuda:0')
        male_like_emb=self.generate_fake_embeddings(male_i_emb,female_i_emb,similarity)
        # male_like_generator = self.build_generator(male_like_emb.shape[1:])
        # male_like_discriminator = self.build_discriminator(male_i_emb.shape[1:])
        # Generate embeddings
        generated_male = self.build_generator(male_like_emb.shape[1:])
        len_male_noise = int(male_i_emb.size()[0] * 0.4)
        male_like_emb = male_like_emb.to('cuda:0')
        # if 0<male_like_emb.shape[0]<len_male_noise:
        #     add_male_i = torch.cat((male_like_emb[-male_like_emb.shape[0]:], male_i_emb[:-male_like_emb.shape[0]]), 0)
        # elif male_like_emb.shape[0]==0:
        #     add_male_i=male_i_emb
        # else:
        #
        add_male_i = torch.cat((male_like_emb[-(len_male_noise-1):], male_i_emb[:(1-len_male_noise)]), 0)
        female_like_emb = self.generate_fake_embeddings(female_i_emb, male_i_emb,similarity)
        # female_like_generator = self.build_generator(female_like_emb.shape[1:])
        # female_like_discriminator = self.build_discriminator(female_i_emb.shape[1:])
        # male_like_emb, female_like_emb = self.fake_pos(male_noise_i_emb,female_noise_i_emb)
        generated_female = self.build_generator(female_like_emb.shape[1:])

        # Discriminator training
        with tf.GradientTape() as tape:
            real_output_male = self.build_discriminator(male_i_emb.shape[1:])
            fake_output_male = self.build_discriminator(generated_male.shape[1:])
            real_output_female = self.build_discriminator(male_i_emb.shape[1:])
            fake_output_female = self.build_discriminator(generated_female.shape[1:])

            dis_loss_male = self.loss_fn(tf.ones_like(real_output_male), real_output_male) + \
                            self.loss_fn(tf.zeros_like(fake_output_male), fake_output_male)

            dis_loss_female = self.loss_fn(tf.ones_like(real_output_female), real_output_female) + \
                              self.loss_fn(tf.zeros_like(fake_output_female), fake_output_female)

        gradients = tape.gradient([dis_loss_male, dis_loss_female], self.discriminator.trainable_variables)
        self.dis_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Generator training
        with tf.GradientTape() as tape:
            generated_male = self.build_generator(generated_male.shape[1:])
            generated_female = self.build_generator(generated_female.shape[1:])
            fake_output_male = self.build_discriminator(generated_male.shape[1:])
            fake_output_female = self.build_discriminator(generated_female.shape[1:])

            gen_loss_male = self.loss_fn(tf.ones_like(fake_output_male), fake_output_male)
            gen_loss_female = self.loss_fn(tf.ones_like(fake_output_female), fake_output_female)
        gradients = tape.gradient([gen_loss_male, gen_loss_female], self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        len_female_noise = int(female_i_emb.size()[0] * 0.4)
        female_like_emb = female_like_emb.to('cuda:0')
        # if 0<female_like_emb.shape[0] < len_female_noise:
        #     add_female_i = torch.cat((female_like_emb[-female_like_emb.shape[0]:], female_i_emb[:-female_like_emb.shape[0]]), 0)
        # elif female_like_emb.shape[0]==0:
        #     add_female_i=female_i_emb
        # else:
        #
        add_female_i = torch.cat((female_i_emb[:(1-len_female_noise)],female_like_emb[-(len_female_noise-1):]), 0)

        male_j_batch = torch.masked_select(j_batch, male_gender)
        female_j_batch = torch.masked_select(j_batch, female_gender)
        male_j_emb = F.embedding(male_j_batch,item_emb) 
        female_j_emb = F.embedding(female_j_batch,item_emb)  
        
        male_u_batch = torch.masked_select(u_batch, male_gender)
        female_u_batch = torch.masked_select(u_batch, female_gender)
        male_u_emb = F.embedding(male_u_batch,user_emb) 
        female_u_emb = F.embedding(female_u_batch,user_emb)
        
        prediction_female_neg = (female_u_emb * female_j_emb).sum(dim=-1)
        prediction_female_add = (female_u_emb * add_female_i).sum(dim=-1)
        loss_female_add = -((prediction_female_add - prediction_female_neg).sigmoid().log().mean())
        prediction_male_neg = (male_u_emb * male_j_emb).sum(dim=-1)
        prediction_male_add = (male_u_emb * add_male_i).sum(dim=-1)
        loss_male_add = -((prediction_male_add - prediction_male_neg).sigmoid().log().mean())
        loss_add=loss_male_add+loss_female_add
        l2_regulization_female = 0.01*(female_u_emb**2+add_female_i**2+female_j_emb**2).sum(dim=-1).mean()
        l2_regulization_male = 0.01 * (male_u_emb ** 2 + add_male_i ** 2 + male_j_emb ** 2).sum(dim=-1).mean()
        l2_regulization= (l2_regulization_female+ l2_regulization_male)/2
        # print('l2_regulization')

        # male_u_emb = male_u_emb.to('cuda:0')
        # male_like_emb = male_like_emb.to('cuda:0')
        # female_u_emb = female_u_emb.to('cuda:0')
        # female_like_emb = female_like_emb.to('cuda:0')
        # prediction_neg_male = (male_u_emb * male_j_emb).sum(dim=-1)
        # prediction_pos_male = (male_u_emb * male_like_emb).sum(dim=-1)
        # loss_fake_male = -((prediction_pos_male - prediction_neg_male).sigmoid().log().mean())
        # prediction_neg_female = (female_u_emb * female_j_emb).sum(dim=-1)
        # prediction_pos_female = (female_u_emb * female_like_emb).sum(dim=-1)
        # loss_fake_female = -((prediction_pos_female - prediction_neg_female).sigmoid().log().mean())
        # loss_fake = loss_fake_male + loss_fake_female
        # l2_regulization2 = 0.01*(male_like_emb**2).sum(dim=-1).mean()+ 0.01*(female_like_emb**2).sum(dim=-1).mean()
        # print('l2_regulization2')

        loss_task = 1*loss_add + l2_regulization
        # loss_add_item = loss_fake + l2_regulization2
        # all_loss = [loss_task, l2_regulization,loss_add_item]
        #
        # all_loss = [male_gen_loss,male_disc_loss,female_gen_loss,female_disc_loss]
        # print('all_loss')

        return loss_task,l2_regulization,dis_loss_male, dis_loss_female, gen_loss_male, gen_loss_female

        # return all_loss
    # Detach the return variables
    def embed(self, adj_pos): 
        u_emb = self.embed_user.weight 
        i_emb = self.embed_item.weight  
        return u_emb.detach(),i_emb.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))
      
    
############ Model #############
# parser = argparse.ArgumentParser(description="Please select a similarity to train the model.", add_help=False)
# parser.add_argument("--similarity", default=0.8, required=True, type=float, help="Options: 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
#
# args = parser.parse_args()
# similarity = args.similarity
similarity =0.7
model = FairData(user_num, item_num, factor_num,users_features,similarity)
model=model.to('cuda')

# task_optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

task_optimizer = torch.optim.Adam(list(model.embed_user.parameters()) + \
                            list(model.embed_item.parameters()) ,lr=0.001)
noise_optimizer = torch.optim.Adam(list(model.noise_item.parameters()),lr=0.001)

gen_optimizer = tf.keras.optimizers.Adam()
dis_optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile_models(gen_optimizer, dis_optimizer, loss_fn)

############ dataset #############
train_dataset = data_utils.BPRData(
        train_dict=train_dict,num_item=item_num, num_ng=5 ,is_training=0, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4)

testing_dataset_loss = data_utils.BPRData(
        train_dict=test_dict,num_item=item_num, num_ng=5 ,is_training=1, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)

######################################################## TRAINING #####################################


print('--------training processing-------')
count, best_hr = 0, 0  
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
trained_end = 0 

for epoch in tqdm(range(150)):
    model.train()
    start_time = time.time()
    print('negative samping, need some minute')
    train_loader.dataset.ng_sample()
    print('negative samping, end')


    loss_current = [[],[],[],[]]
    #GAN
    for user_batch,  itemi_batch,itemj_batch, in tqdm(train_loader):

        loss_task,l2_regulization,dis_loss_male, dis_loss_female, gen_loss_male, gen_loss_female=  model(1,user_batch,itemi_batch,itemj_batch)
        #
        # male_like_cpu= male_like_emb.cpu()
        # male_i_cpu = male_i_emb.cpu()
        # female_like_cpu= female_like_emb.cpu()
        # female_i_cpu = female_i_emb.cpu()
        # male_like_np = male_like_cpu.detach().numpy()
        # male_i_np = male_i_cpu.detach().numpy()
        # female_like_np = female_like_cpu.detach().numpy()
        # female_i_np = female_i_cpu.detach().numpy()
        # # 生成器训练步骤
        # with tf.GradientTape() as gen_mtape:
        #     # 通过生成器生成调整后的嵌入向量
        #     male_adjusted_embeddings = male_like_generator(male_like_np)
        #     # 计算生成器的损失函数
        #     male_gen_loss = mse_loss(male_adjusted_embeddings, male_i_np)
        # with tf.GradientTape() as gen_ftape:
        #     # 通过生成器生成调整后的嵌入向量
        #     female_adjusted_embeddings = female_like_generator(female_like_np)
        #     # 计算生成器的损失函数
        #     female_gen_loss = mse_loss(female_adjusted_embeddings, female_i_np)
        # male_gradients_of_generator = gen_mtape.gradient(male_gen_loss, male_like_generator.trainable_variables)
        # female_gradients_of_generator = gen_ftape.gradient(female_gen_loss, female_like_generator.trainable_variables)
        # male_generator_optimizer.apply_gradients(
        #     zip(male_gradients_of_generator, male_like_generator.trainable_variables))
        # female_generator_optimizer.apply_gradients(zip(female_gradients_of_generator, female_like_generator.trainable_variables))
        # with tf.GradientTape() as disc_mtape:
        #     # 使用真实嵌入向量作为输入
        #     male_real_predictions = male_like_discriminator(male_like_np)
        #     # 使用生成器生成的调整后的嵌入向量作为输入
        #     male_fake_embeddings = male_adjusted_embeddings
        #     male_fake_predictions = male_like_discriminator(male_fake_embeddings)
        #     # 计算判别器的损失函数
        #     male_disc_loss = bce_loss(tf.ones_like(male_real_predictions), male_real_predictions) + \
        #                      bce_loss(tf.zeros_like(male_fake_predictions), male_fake_predictions)
        # with tf.GradientTape() as disc_ftape:
        #     female_real_predictions = female_like_discriminator(female_like_np)
        #     # 使用生成器生成的调整后的嵌入向量作为输入
        #     female_fake_embeddings = female_adjusted_embeddings
        #     female_fake_predictions = female_like_discriminator(female_fake_embeddings)
        #     # 计算判别器的损失函数
        #     female_disc_loss = bce_loss(tf.ones_like(female_real_predictions), female_real_predictions) + \
        #                        bce_loss(tf.zeros_like(female_fake_predictions), female_fake_predictions)
        # male_gradients_of_discriminator = disc_mtape.gradient(male_disc_loss,
        #                                                       male_like_discriminator.trainable_variables)
        # female_gradients_of_discriminator = disc_ftape.gradient(female_disc_loss,
        #                                                         female_like_discriminator.trainable_variables)
        # male_discriminator_optimizer.apply_gradients(
        #     zip(male_gradients_of_discriminator, male_like_discriminator.trainable_variables))
        # female_discriminator_optimizer.apply_gradients(zip(female_gradients_of_discriminator, female_like_discriminator.trainable_variables))
        #
        # del loss_task, l2_regulization, male_i_emb, male_like_emb, male_like_generator, \
        #     male_like_discriminator, female_i_emb, female_like_emb, female_like_generator, \
        #     female_like_discriminator, male_like_cpu, male_i_cpu, female_like_cpu, female_i_cpu, \
        #     male_like_np, male_i_np, female_like_np, female_i_np, male_adjusted_embeddings, \
        #     male_gen_loss, gen_mtape, gen_ftape, male_gradients_of_generator, female_gradients_of_generator, \
        #     male_real_predictions, male_fake_embeddings, male_fake_predictions, male_disc_loss, disc_mtape, \
        #     female_real_predictions, female_fake_embeddings, female_fake_predictions, female_disc_loss, disc_ftape, \
        #     male_gradients_of_discriminator, female_gradients_of_discriminator
    for user_batch,  itemi_batch,itemj_batch, in tqdm(train_loader):
        get_loss =  model(1,user_batch,itemi_batch,itemj_batch)
        task_loss,relu_loss,dis_loss_male, dis_loss_female, gen_loss_male, gen_loss_female= get_loss
        loss_current[0].append(task_loss.item())
        loss_current[1].append(relu_loss.item())
        task_optimizer.zero_grad()
        task_loss.backward()
        task_optimizer.step()
    # for user_batch,  itemi_batch,itemj_batch, in tqdm(train_loader):
    #     user_batch = user_batch.cuda()
    #     itemi_batch = itemi_batch.cuda()
    #     itemj_batch = itemj_batch.cuda()
    #     get_loss =  model(1,user_batch,itemi_batch,itemj_batch)
    #     task_loss,relu_loss,noise_loss = get_loss
    #     loss_current[2].append(noise_loss.item())
    #     noise_optimizer.zero_grad()
    #     noise_loss.backward()
    #     noise_optimizer.step()
    loss_current=np.array(loss_current)
    elapsed_time = time.time() - start_time
    train_loss_task = round(np.mean(loss_current[0]),4) 
    train_loss_sample = round(np.mean(loss_current[1]),4) 
    # train_loss_noise = round(np.mean(loss_current[2]),4)
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1)) 
    if epoch==135:
        trained_end=1

    loss_str='loss' 
    loss_str+=' task:'+str(train_loss_task) 
    str_print_train +=loss_str
    print(run_id+' '+str_print_train)
    
    
    model.eval()

    f1_u_embedding,f1_i_emb= model.embed(1)
    user_e_f1 = f1_u_embedding.cpu().numpy() 
    item_e_f1 = f1_i_emb.cpu().numpy()  
    if trained_end == 1:#epoch==135:
        PATH_model=path_save_model_base+'/best_model'+str(similarity)+'.pt'
        torch.save(model.state_dict(), PATH_model)
        
        PATH_model_u_f1=path_save_model_base+'/user_emb'+str(similarity)+'.npy'
        np.save(PATH_model_u_f1,user_e_f1) 
        PATH_model_i_f1=path_save_model_base+'/item_emb'+str(similarity)+'.npy'
        np.save(PATH_model_i_f1,item_e_f1)

        print("Training end")
        os.system("python ./test.py --runid=\'"+run_id+"\'")
        exit()
    del f1_u_embedding, f1_i_emb, user_e_f1, item_e_f1, loss_current
    gc.collect()

