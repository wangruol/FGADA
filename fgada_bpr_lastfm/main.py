# -- coding:UTF-8
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm
import random
import torch
from torch.nn.functional import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2024) 


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
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import time
import data_utils
import tensorflow as tf

dataset_base_path='../data/lastfm/' 
user_reid, item_reid= np.load(dataset_base_path+'/u_i_reid.npy',allow_pickle=True)


##lastfm
user_num=len(user_reid)#  
item_num=len(item_reid)#
factor_num=64
batch_size=2048*100#10
top_k=20
num_negative_test_val=-1##all


run_id="fgada_bpr_lastfm"
print('Model:',run_id)
dataset='lastfm'

path_save_model_base='./best_model/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

train_dict,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)
test_dict,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True) 
val_dict,val_dict_count = np.load(dataset_base_path+'/val.npy',allow_pickle=True)  

users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]#gender

# print(train_dict_count,test_dict_count,val_dict_count)
np.seterr(divide='ignore',invalid='ignore')

# 判别器训练过程
def train_male_discriminator(male_like_emb,male_i_emb,female_i_emb,similarity):
    male_discriminator.zero_grad()

    # 虚假样本（由生成器生成）
    generated_output = male_generated(male_like_emb).detach()
    fake_outputs = male_discriminator(generated_output).to('cuda')
    fake_targets = torch.full((fake_outputs.size(0), fake_outputs.size(1)), 0, device='cuda')
    fake_targets = fake_targets.float()
    # 真实样本
    real_outputs = male_discriminator(male_i_emb).to('cuda')
    real_targets = torch.full((real_outputs.size(0), real_outputs.size(1)), similarity, device='cuda')
    real_targets = real_targets.float()
    real_loss = adversarial_loss(real_outputs, real_targets)
    fake_loss = adversarial_loss(fake_outputs, fake_targets)
    # 真实样本2
    real_outputs_2 = male_discriminator(female_i_emb).to('cuda')
    real_targets_2 = torch.full((real_outputs_2.size(0), real_outputs_2.size(1)), 1 - similarity, device='cuda')
    real_targets_2 = real_targets_2.float()
    # print("6")
    real_loss_2 = adversarial_loss(real_outputs_2, real_targets_2)
    # 损失函数
    discriminator_loss = real_loss + real_loss_2 + fake_loss
    discriminator_loss.backward(retain_graph=True)
    male_discriminator_optimizer.step()

    # print("smale_discriminator")
    return discriminator_loss.item()

# 生成器训练过程
def train_male_generator(male_like_emb,male_i_emb,similarity):
    male_generated.zero_grad()

    # 生成虚假样本
    fake_inputs = male_generated(male_like_emb).detach()

    # 希望虚假样本被判别器误分类为真实样本
    outputs = male_discriminator(fake_inputs).to('cuda')
    targets = torch.full((outputs.size(0), outputs.size(1)), similarity, device='cuda')
    targets = targets.float()

    generator_loss = adversarial_loss(outputs, targets)

    generator_loss.backward(retain_graph=True)
    male_generator_optimizer.step()
    # print("smale_generator")
    return generator_loss.item()
def train_female_discriminator(male_like_emb,male_i_emb,female_i_emb,similarity):
    female_discriminator.zero_grad()

    # 虚假样本（由生成器生成）
    generated_output = male_generated(male_like_emb).detach()
    fake_outputs = male_discriminator(generated_output).to('cuda')
    fake_targets = torch.full((fake_outputs.size(0), fake_outputs.size(1)), 0, device='cuda')
    fake_targets = fake_targets.float()
    # 真实样本
    real_outputs = male_discriminator(male_i_emb).to('cuda')
    real_targets = torch.full((real_outputs.size(0), real_outputs.size(1)), similarity, device='cuda')
    real_targets = real_targets.float()
    real_loss = adversarial_loss(real_outputs, real_targets)
    fake_loss = adversarial_loss(fake_outputs, fake_targets)
    # 真实样本2
    real_outputs_2 = male_discriminator(female_i_emb).to('cuda')
    real_targets_2 = torch.full((real_outputs_2.size(0), real_outputs_2.size(1)), 1 - similarity, device='cuda')
    real_targets_2 = real_targets_2.float()
    # print("6")
    real_loss_2 = adversarial_loss(real_outputs_2, real_targets_2)
    # 损失函数
    discriminator_loss = real_loss + real_loss_2 + fake_loss
    discriminator_loss.backward(retain_graph=True)
    male_discriminator_optimizer.step()

    # print("female_discriminatorr")
    return discriminator_loss.item()

# 生成器训练过程
def train_female_generator(male_like_emb,male_i_emb,similarity):
    female_generated.zero_grad()

    # 生成虚假样本
    fake_inputs = male_generated(male_like_emb).detach()

    # 希望虚假样本被判别器误分类为真实样本
    outputs = male_discriminator(fake_inputs).to('cuda')
    targets = torch.full((outputs.size(0), outputs.size(1)), similarity, device='cuda')
    targets = targets.float()

    generator_loss = adversarial_loss(outputs, targets)

    generator_loss.backward(retain_graph=True)
    male_generator_optimizer.step()

    # print("sfemale_generator")
    return generator_loss.item()


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Generator, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.Tanh()
        # )
        self.embed_dim = input_dim
        self.net =  nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            # nn.Sigmoid()
        )
        # self.batchnorm = nn.BatchNorm1d(self.embed_dim,track_running_stats=True)

    def forward(self, ents_emb):
        if not isinstance(ents_emb, torch.Tensor):
            ents_emb = torch.tensor(ents_emb, dtype=torch.float32)

        h3 = self.net(ents_emb)
        # print("Generator")
        return h3

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim, 1),
        #     nn.Sigmoid()
        # )
        self.embed_dim=input_dim
        self.out_dim=input_dim
        self.model = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 8), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.embed_dim / 8), int(self.embed_dim / 16), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.embed_dim / 16), self.out_dim, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.out_dim, self.out_dim, bias=True),
            )

    def forward(self, x, return_loss=True):
        scores = self.model(x)
        outputs = F.softmax(scores, dim=1)  # F.log_softmax(scores, dim=1)
        # outputs = scores
        # pdb.set_trace()
        # if return_loss:
        #     loss = self.criterion(outputs, labels)
        #     # if loss>100:
        #     #     print('age d process')
        #     #     pdb.set_trace()
        #     # pdb.set_trace()
        #     return loss
        # print("discriminatorr")
        return outputs
        # return self.model(x)

class FairData(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,similarity,b,gcn_user_embs=None,gcn_item_embs=None):
        super(FairData, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors. 
        """ 
        self.users_features = torch.cuda.LongTensor(users_features) 
        self.user_num = user_num
        self.factor_num =factor_num
        self.similarity = similarity
        self.b = b

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        self.noise_item = nn.Embedding(item_num, factor_num)    
        nn.init.normal_(self.noise_item.weight, std=0.01)  
        
        
        self.min_clamp=-1
        self.max_clamp=1

        self.optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    def fake_pos(self, embedding1, embedding2, similarity):
        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding1_tensor = embedding1.to(device)
        embedding2_tensor = embedding2.to(device)
        batch_size = 100
        filtered_embeddings = []

        # 计算 embedding2 的范数 (在 GPU 上)
        norm_embedding2 = torch.norm(embedding2_tensor, dim=1, keepdim=True)

        m, d = embedding1_tensor.shape
        for i in range(0, m, batch_size):
            # 取出当前批次的 embedding1
            batch_embedding1 = embedding1_tensor[i:i + batch_size]

            # 计算当前批次的 embedding1 范数 (在 GPU 上)
            norm_batch_embedding1 = torch.norm(batch_embedding1, dim=1, keepdim=True)

            # 通过矩阵乘法计算批次与 embedding2 的点积 (在 GPU 上)
            dot_products = torch.mm(batch_embedding1, embedding2_tensor.T)

            # 计算范数的乘积 (在 GPU 上)
            norms_product = torch.mm(norm_batch_embedding1, norm_embedding2.T)

            # 计算余弦相似度 (在 GPU 上)
            cosine_similarities = dot_products / norms_product

            # 对于每个 batch_embedding1，找到它与 embedding2 中的最大相似度
            max_similarities = torch.max(cosine_similarities, dim=1).values

            # 筛选相似度小于指定阈值的向量
            filtered_batch = batch_embedding1[max_similarities < similarity]

            # 添加到结果集中
            filtered_embeddings.append(filtered_batch)

        # 将批次结果组合成最终的结果
        return torch.cat(filtered_embeddings, dim=0).cpu()  # 将结果转移回 CPU（如需要）
        # embedding1_tensor = embedding1.to(device)
        # embedding2_tensor = embedding2.to(device)
        # embeddings1 = embedding1_tensor.view(-1, embedding1_tensor.shape[-1])
        # embeddings2 = embedding2_tensor.view(-1, embedding2_tensor.shape[-1])
        # norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        # norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # 准备收集结果
        # filtered_embeddings = []
        #
        # batch_size = 512
        # # 分批处理以避免内存问题
        # num_batches = (norm_embeddings2.size(0) + batch_size - 1) // batch_size
        # for i in range(num_batches):
        #     batch_start = i * batch_size
        #     batch_end = min(batch_start + batch_size, norm_embeddings2.size(0))
        #     batch_embeddings2 = norm_embeddings2[batch_start:batch_end]
        #
        #     # 计算当前批次的相似度
        #     similarities = torch.matmul(norm_embeddings1, batch_embeddings2.T)
        #
        #     # 找到最大相似度并应用阈值
        #     max_similarities, _ = similarities.max(dim=1)
        #     filtered_indices = (max_similarities < similarity_threshold).nonzero(as_tuple=True)[0]
        #     filtered_embeddings.append(embeddings1[filtered_indices])
        #
        #     # 合并所有批次的结果
        #     filtered_embeddings = torch.cat(filtered_embeddings, dim=0)
        #
        #     # 如有需要，转换为 NumPy 数组
        #     return filtered_embeddings.detach().cpu().numpy() if len(filtered_embeddings) > 0 else []

    def noise (self, male_noise_i_emb, female_noise_i_emb):
        male_len = male_noise_i_emb.shape[0]
        female_len = female_noise_i_emb.shape[0]

        avg_len = 1
        male_end_idx = male_len % avg_len + avg_len
        male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1, avg_len, self.factor_num)
        male_noise_i_mean = torch.mean(male_noise_i_reshape, axis=1)
        male_noise_len = male_noise_i_mean.shape[0]
        if male_noise_len > female_len:
            female_like = male_noise_i_mean[:female_len]
        else:
            expand_len = int(female_len / male_noise_len) + 1
            female_like = male_noise_i_mean.repeat(expand_len, 1)[:female_len]

        female_end_idx = female_len % avg_len + avg_len
        female_noise_i_emb_reshape = female_noise_i_emb[:-female_end_idx].reshape(-1, avg_len, self.factor_num)
        female_noise_i_mean = torch.mean(female_noise_i_emb_reshape, axis=1)
        female_noise_len = female_noise_i_mean.shape[0]
        if female_noise_len > male_len:
            male_like = female_noise_i_mean[:male_len]
        else:
            expand_len = int(male_len / female_noise_len) + 1
            male_like = female_noise_i_mean.repeat(expand_len, 1)[:male_len]

        return male_like, female_like
        
    def forward(self, adj_pos,u_batch,i_batch,j_batch):

        # filter gender
        user_emb = self.embed_user.weight
        item_emb = self.embed_item.weight
        noise_emb = self.embed_item.weight
        noise_emb = torch.clamp(noise_emb, min=self.min_clamp, max=self.max_clamp)
        noise_emb += self.noise_item.weight
        b = self.b
        #get gender attribute
        gender = F.embedding(u_batch, torch.unsqueeze(self.users_features, 1)).reshape(-1)
        male_gender = gender.type(torch.BoolTensor).cuda()
        female_gender = (1 - gender).type(torch.BoolTensor).cuda()

        u_emb = F.embedding(u_batch,user_emb)
        i_emb = F.embedding(i_batch,item_emb)
        j_emb = F.embedding(j_batch,item_emb)

        noise_i_emb2 = F.embedding(i_batch, noise_emb)
        len_noise = int(i_emb.size()[0] * b)
        add_emb = torch.cat((i_emb[:-len_noise], noise_i_emb2[-len_noise:]), 0)
        noise_j_emb2 = F.embedding(j_batch, noise_emb)
        len_noise = int(j_emb.size()[0] * b)
        add_emb_j = torch.cat((noise_j_emb2[-len_noise:], j_emb[:-len_noise]), 0)

        male_i_batch = torch.masked_select(i_batch, male_gender)
        female_i_batch = torch.masked_select(i_batch, female_gender)
        male_noise_i_emb = F.embedding(male_i_batch, noise_emb)
        female_noise_i_emb = F.embedding(female_i_batch, noise_emb)
        male_i_emb = F.embedding(male_i_batch, item_emb)
        female_i_emb = F.embedding(female_i_batch, item_emb)

        male_j_batch = torch.masked_select(j_batch, male_gender)
        female_j_batch = torch.masked_select(j_batch, female_gender)
        male_j_emb = F.embedding(male_j_batch, item_emb)
        female_j_emb = F.embedding(female_j_batch, item_emb)

        male_u_batch = torch.masked_select(u_batch, male_gender)
        female_u_batch = torch.masked_select(u_batch, female_gender)
        male_u_emb = F.embedding(male_u_batch, user_emb)
        female_u_emb = F.embedding(female_u_batch, user_emb)
        # ——————————————创新点开始——————————————————————————————————————————————————————
        # 相似度判断出输入embedding
        female_like_emb = self.fake_pos(male_i_emb, female_i_emb, similarity)
        male_like_emb = self.fake_pos(female_i_emb, male_i_emb, similarity)
        # female_like_emb = self.fake_pos(female_i_emb, male_i_emb, similarity)
        print("male_like_emb", male_like_emb, "female_like_emb", female_like_emb)
        male_i_emb_cpu = male_i_emb.to('cpu')
        female_i_emb_cpu = female_i_emb.to('cpu')
        if male_like_emb.size(0) != 0:
            for epoch in tqdm(range(100)):
                disc_loss = train_male_discriminator(male_like_emb, male_i_emb_cpu, female_i_emb_cpu, similarity)
                gen_loss = train_male_generator(male_like_emb, female_i_emb, similarity)
                print('Epoch {}: G_Loss: {:.4f}, D_Loss: {:.4f}'.format(epoch, gen_loss, disc_loss))

            # 保存模型
            torch.save(male_discriminator.state_dict(), 'bpr_ml_male_discriminator' + str(similarity)+ str(b)+ '.pth')
            torch.save(female_discriminator.state_dict(), 'bpr_ml_male_discriminator ' + str(similarity) + str(b)+ 'r.pth')
            male_gan_emb = male_generated(male_like_emb).detach()
            # male_gan_emb = torch.from_numpy(male_like_emb)

        if female_like_emb.size(0) != 0:
            for epoch in tqdm(range(100)):
                disc_loss = train_female_discriminator(female_like_emb, female_i_emb_cpu, male_i_emb_cpu, similarity)
                gen_loss = train_female_generator(female_like_emb, male_i_emb, similarity)
                print('Epoch {}: G_Loss: {:.4f}, D_Loss: {:.4f}'.format(epoch, gen_loss, disc_loss))

            torch.save(female_generated.state_dict(), 'bpr_ml_female_generator' + str(similarity)+ str(b)+ 'th')
            torch.save(female_discriminator.state_dict(), 'bpr_ml_female_discriminator ' + str(similarity) + str(b)+ 'r.pth')
            female_gan_emb = female_generated(female_like_emb).detach()
            # female_gan_emb = torch.from_numpy(female_like_emb)
            # 真假数据混合计算loss值
        len_male_add = int(male_i_emb.size()[0] * b)
        len_female_add = int(female_i_emb.size()[0] * b)
        male_noise_emb, female_noise_emb = self.noise(male_noise_i_emb, female_noise_i_emb)

        if male_like_emb.size(0) == 0:
            add_male_emb_i =male_noise_emb
        elif len_male_add > male_gan_emb.size()[0]:
            len_gan_add = int(male_gan_emb.size()[0])
            add_male_emb_i = torch.cat((male_i_emb[:-len_gan_add],male_gan_emb.to(male_i_emb.device)),0)
        else:
            add_male_emb_i=torch.cat((male_i_emb[:-len_male_add],male_gan_emb.to(male_i_emb.device)[0:len_male_add]),0)

        if female_like_emb.size(0) == 0:
            add_female_emb_i = female_noise_emb
        elif len_female_add > female_gan_emb.size()[0]:
            len_gan_add = int(female_gan_emb.size()[0])
            add_female_emb_i = torch.cat((female_i_emb[:-len_gan_add],female_gan_emb.to(female_i_emb.device)),0)
        else:
            add_female_emb_i=torch.cat((female_i_emb[:-len_female_add],female_gan_emb.to(female_i_emb.device)[0:len_female_add]),0)

        # prediction_neg_female = (female_u_emb * female_j_emb).sum(dim=-1)
        # prediction_add__female = (female_u_emb * add_female_emb_i).sum(dim=-1)
        # prediction_neg_male = (male_u_emb * male_j_emb).sum(dim=-1)
        # prediction_add__male = (male_u_emb * add_male_emb_i).sum(dim=-1)
        # loss_add_male = -((prediction_add__male - prediction_neg_male).sigmoid().log().mean())
        # loss_add_female = -((prediction_add__female - prediction_neg_female).sigmoid().log().mean())
        # l2_regulization = 0.01 * (u_emb ** 2 + j_emb ** 2).sum(dim=-1).mean()+0.01 * (add_male_emb_i ** 2).sum(dim=-1).mean() + 0.01 * (add_female_emb_i ** 2).sum(
        #     dim=-1).mean()
        # loss_task = loss_add_male+loss_add_female + l2_regulization
        #
        # # return结果
        # return loss_task
        prediction_neg = (u_emb * add_emb_j).sum(dim=-1)
        prediction_add = (u_emb * add_emb).sum(dim=-1)
        loss_add = -((prediction_add - prediction_neg).sigmoid().log().mean())
        l2_regulization = 0.01 * (u_emb ** 2 + add_emb ** 2 + j_emb ** 2).sum(dim=-1).mean()

        prediction_neg_male = (male_u_emb * male_j_emb).sum(dim=-1)
        prediction_pos_male = (male_u_emb * add_male_emb_i).sum(dim=-1)
        loss_fake_male = -((prediction_pos_male - prediction_neg_male).sigmoid().log().mean())
        prediction_neg_female = (female_u_emb * female_j_emb).sum(dim=-1)
        prediction_pos_female = (female_u_emb * add_female_emb_i).sum(dim=-1)
        loss_fake_female = -((prediction_pos_female - prediction_neg_female).sigmoid().log().mean())
        loss_fake = loss_fake_male + loss_fake_female
        l2_regulization2 = 0.01 * (add_male_emb_i ** 2).sum(dim=-1).mean() + 0.01 * (add_female_emb_i ** 2).sum(
            dim=-1).mean()

        loss_task = 1 * loss_add + l2_regulization
        loss_add_item = loss_fake + l2_regulization2
        all_loss = [loss_task, l2_regulization, loss_add_item]

        return all_loss

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

    
########### Model #############
parser = argparse.ArgumentParser(description="Please select a similarity to train the model.", add_help=False)
parser.add_argument("--similarity", default=0.2, required=True, type=float, help="Options: 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
parser.add_argument("--b", default=0.3, required=True, type=float, help="Options: 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")

args = parser.parse_args()
similarity = args.similarity
b=args.b
# similarity =0.4
model = FairData(user_num, item_num, factor_num,users_features,similarity,b)
model=model.to('cuda')

male_generated = Generator(64, 64, 128)
male_discriminator = Discriminator(64, 128)
male_generator_optimizer = torch.optim.Adam(male_generated.parameters(), lr=0.0002)
male_discriminator_optimizer = torch.optim.Adam(male_discriminator.parameters(), lr=0.0002)
female_generated = Generator(64, 64, 128)
female_discriminator = Discriminator(64, 128)
female_generator_optimizer = torch.optim.Adam(female_generated.parameters(), lr=0.0002)
female_discriminator_optimizer = torch.optim.Adam(male_discriminator.parameters(), lr=0.0002)
# 定义损失函数
adversarial_loss = nn.BCELoss()

print(male_generated,female_generated)


task_optimizer = torch.optim.Adam(list(model.embed_user.parameters()) + \
                            list(model.embed_item.parameters()) ,lr=0.001)
noise_optimizer = torch.optim.Adam(list(model.noise_item.parameters()),lr=0.001) 

############ dataset #############

train_dataset = data_utils.BPRData(
        train_dict=train_dict,num_item=item_num, num_ng=5 ,is_training=0, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)

testing_dataset_loss = data_utils.BPRData(
        train_dict=test_dict,num_item=item_num, num_ng=5 ,is_training=1, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)

val_dataset_loss = data_utils.BPRData(
        train_dict=val_dict,num_item=item_num, num_ng=5 ,is_training=2, data_set_count=val_dict_count)
val_loader_loss = DataLoader(val_dataset_loss,
        batch_size=val_dict_count, shuffle=False, num_workers=0)

######################################################## TRAINING #####################################


print('--------training processing-------')
count, best_hr = 0, 0  
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
trained_end = 0 

for epoch in tqdm(range(104)):
    model.train()  
    start_time = time.time()
    print('negative samping, about 3 minute')
    train_loader.dataset.ng_sample()
    print('negative samping is end')

    loss_current = [[], [], [], []]

    for user_batch, itemi_batch, itemj_batch, in train_loader:
        user_batch = user_batch.cuda()
        itemi_batch = itemi_batch.cuda()
        itemj_batch = itemj_batch.cuda()
        get_loss = model(1, user_batch, itemi_batch, itemj_batch)
        task_loss, relu_loss, noise_loss = get_loss
        loss_current[0].append(task_loss.item())
        loss_current[1].append(relu_loss.item())
        task_optimizer.zero_grad()
        task_loss.backward()
        task_optimizer.step()

    for user_batch, itemi_batch, itemj_batch, in train_loader:
        user_batch = user_batch.cuda()
        itemi_batch = itemi_batch.cuda()
        itemj_batch = itemj_batch.cuda()
        get_loss = model(1, user_batch, itemi_batch, itemj_batch)
        task_loss, relu_loss, noise_loss = get_loss
        loss_current[2].append(noise_loss.item())
        noise_optimizer.zero_grad()
        noise_loss.backward()
        noise_optimizer.step()
    loss_current = np.array(loss_current)
    elapsed_time = time.time() - start_time
    train_loss_task = round(np.mean(loss_current[0]), 4)
    train_loss_sample = round(np.mean(loss_current[1]), 4)
    train_loss_noise = round(np.mean(loss_current[2]), 4)
    str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1))
    if epoch==100:
        trained_end=1

    loss_str = 'loss'
    loss_str += ' task:' + str(train_loss_task)
    str_print_train += loss_str
    print(run_id + ' ' + str_print_train)
    
    model.eval()

    f1_u_embedding,f1_i_emb= model.embed(1)
    user_e_f1 = f1_u_embedding.cpu().numpy() 
    item_e_f1 = f1_i_emb.cpu().numpy()  
    
    if trained_end == 1:#epoch==82:
        PATH_model = path_save_model_base + '/best_model' +  str(similarity)+str(b) + '2n.pt'
        torch.save(model.state_dict(), PATH_model)

        PATH_model_u_f1 = path_save_model_base + '/user_emb' + str(similarity)+ str(b) + '2n.npy'
        np.save(PATH_model_u_f1, user_e_f1)
        PATH_model_i_f1 = path_save_model_base + '/item_emb' + str(similarity) + str(b)+ '2n.npy'
        np.save(PATH_model_i_f1, item_e_f1)

        print("Training end")
        os.system("python ./test.py --runid=\'"+run_id+"\' --b=\'"+str(b)+"\' ")
        exit()
    

    