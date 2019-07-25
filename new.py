#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from google.colab import drive
#drive.mount('/content/drive/')


# In[2]:


#cd drive/My Drive/CycleGAN


# In[3]:


#!pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#!pip install torchvision


# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy


# In[ ]:


EPOCHS = 50
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### 以下定义生成器与判别器的网络结构
# 
# - 生成器和判别器均使用孪生网络  
# - 生成器使用RNN方式，第一个step生成local文件，第二个step生成全局图像。  
# - 生成器step1和step2使用同一个骨干网络
# - 判别器最后一层不要用sigmoid

# In[ ]:


#每个local的输出的分支网络（如果需要的话）（如果输入也要分支也可以用这个）
class outBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(outBlock,self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels,32,kernel_size=(3,3),stride=1,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
        nn.Conv2d(32,64,kernel_size=1,stride=1,padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        self.block = nn.Sequential(
        nn.Conv2d(32,64,kernel_size = 1,stride = 1,padding = 0),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        self.sqush = nn.Conv2d(128,out_channels,kernel_size=3,stride=1,padding=1)
        
    def forward(self,x):
        x = self.layer1(x)
        x1 = self.shortcut(x)
        x2 = self.block(x)
        x = torch.cat((x1,x2),1)
        return self.sqush(x)


# In[ ]:


#注册钩子函数
class saveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[ ]:


class unetUpSampleBlock(nn.Module):
    """
    用于创建unet右侧的上采样层，采用转置卷积进行上采样（尺寸×2）
    self.tranConv将上一层进行上采样，尺寸×2
    self.conv，将左侧特征图再做一次卷积减少通道数，所以尺寸不变
    此时两者尺寸正好一致-----建立在图片尺寸为128×128的基础上，否则上采样不能简单的×2
    """
    def __init__(self,in_channels,feature_channels,out_channels,dp=False,ps=0.25):#注意，out_channels 是最终输出通道的一半。
        super(unetUpSampleBlock,self).__init__()
        self.tranConv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,bias=False)#输出尺寸正好为输入尺寸的两倍
        self.conv = nn.Conv2d(feature_channels,out_channels,1,bias=False) #这一层将传来的特征图再做一次卷积，将特征图通道数减半
        self.bn = nn.BatchNorm2d(out_channels*2) #将特征图与上采样再通道出相加后再一起归一化
        self.dp = dp
        if dp:
            self.dropout = nn.Dropout(ps,inplace=True)
            
    def forward(self,x,features):
        x1 = self.tranConv(x)
        x2 = self.conv(features)
        x = torch.cat([x1,x2],dim=1)
        x = self.bn(F.relu(x))
        return self.dropout(x) if self.dp else x


# In[ ]:


class Generator(nn.Module):
    #基于resnet50的UNet网络
    #NIR是可见光模式，3通道
    #主干网络为Unet，输入输出尺寸均为64×64
    def __init__(self,model,in_channels,out_channels):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True)
        )
        self.downsample = nn.Sequential(*list(model.children())[4:-2])
        #print(len(list(model.children())[4:-2]))
        self.features = [saveFeatures(list(self.downsample.children())[i]) for i in range(3)]
        self.up1 = unetUpSampleBlock(2048,1024,512) #feature:self.features[2]
        self.up2 = unetUpSampleBlock(1024,512,256)
        self.up3 = unetUpSampleBlock(512,256,128)
        self.up4 = unetUpSampleBlock(256,64,32) #feature:self.layer1的输出
        self.outlayer = nn.Conv2d(64,out_channels,3,1,1)
        
    def forward(self,batch):
        out_batch = []
        for i in batch:
            x1 = self.layer1(i)
            x = self.downsample(x1)
            x = self.up1(x,self.features[2].features)
            x = self.up2(x,self.features[1].features)
            x = self.up3(x,self.features[0].features)
            x = self.up4(x,x1)
            out_batch.append(self.outlayer(x))
        return out_batch


# In[10]:


m = models.resnet50(pretrained=True)
tem_paras = copy.deepcopy(m.layer1[0].downsample[0].state_dict())
m.layer1[0].downsample[0] = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
m.layer1[0].downsample[0].load_state_dict(tem_paras)
tem_paras = copy.deepcopy(m.layer1[0].conv2.state_dict())
m.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
m.layer1[0].conv2.load_state_dict(tem_paras)
del tem_paras


# In[ ]:


genernator_VIS2NIR = Generator(m,3,1)
genernator_NIR2VIS = Generator(m,1,3)
merge_NIR2NIR = outBlock(5,1)
merge_VIS2VIS = outBlock(15,3)


# In[12]:


discriminator_A_NIR = models.resnet34(pretrained=True)
discriminator_B_VIS = models.resnet34(pretrained=True)


# In[ ]:


discriminator_A_NIR.fc = nn.Linear(512,1,bias = True)
discriminator_B_VIS.fc = nn.Linear(512,1,bias = True)
#resnet降的倍数太多了，减少一个pool
discriminator_B_VIS.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
discriminator_A_NIR.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)


# In[ ]:


genernator_NIR2VIS = genernator_NIR2VIS.to(device)
genernator_VIS2NIR = genernator_VIS2NIR.to(device)
merge_NIR2NIR = merge_NIR2NIR.to(device)
merge_VIS2VIS = merge_VIS2VIS.to(device)
discriminator_A_NIR = discriminator_A_NIR.to(device)
discriminator_B_VIS = discriminator_B_VIS.to(device)


# ### 以下定义数据读取
# 
# - 分别读取一张图片即上面的头、胸、手、腿  
# - 全局图片为.resize((64,128))  
# - 头为.resize((32,16))
# - 胸部为.resize((64,64))  
# - 手臂为.resize((64,64))  
# - 腿部为.resize((64,128))  
# - 坐标文件存储在images_NIR.yml和images_VIS.yml两个文件上

# In[ ]:


from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import yaml
import torch.utils.data as data


# In[ ]:


def process(x1,y1,x2,y2):
    return x1,y1,x2,y2


# In[ ]:


class CustomDatasets(Dataset):
    def __init__(self,img_NIR_all,img_VIS_all,img_train_NIR_list,img_train_VIS_list,img_NIR_dir,img_VIS_dir):
        self.img_NIR_all = img_NIR_all
        self.img_VIS_all = img_VIS_all
        self.img_train_NIR_list = img_train_NIR_list
        self.img_train_VIS_list = img_train_VIS_list
        self.img_NIR_dir = img_NIR_dir
        self.img_VIS_dir = img_VIS_dir
        self.NIR_key = list(img_NIR_all.keys())
        self.VIS_key = list(img_VIS_all.keys())
        
    def __len__(self):
        return len(self.img_train_NIR_list)
    
    def __getitem__(self,idx):
        img_NIR_info = self.img_NIR_all[self.NIR_key[idx]]
        img_VIS_info = self.img_VIS_all[self.VIS_key[idx]]
        
        batch = {}
        
        name = self.NIR_key[idx].split('.')
        name = name[0][:-2]+'.'+name[1]
        batch['img_NIR'] = Image.open(os.path.join(self.img_NIR_dir,name)).convert('L').resize((64,128))
        #如果想要打乱NIR图像与VIS图像之间的关系的话只需重新随机选择一个idx即可
        name = self.VIS_key[idx].split('.')
        name = name[0][:-2]+'.'+name[1]
        batch['img_VIS'] = Image.open(os.path.join(self.img_VIS_dir,name)).convert('RGB').resize((64,128))
        
        batch['id_NIR'] = int(self.NIR_key[idx].split('_')[0])
        batch['id_VIS'] = int(self.VIS_key[idx].split('_')[0])
        
        batch['head_NIR'] = batch['img_NIR'].crop(process(**img_NIR_info['head'])).resize((32,16))
        batch['head_VIS'] = batch['img_VIS'].crop(process(**img_NIR_info['head'])).resize((32,16))
        
        batch['chest_NIR'] = batch['img_NIR'].crop(process(**img_NIR_info['chest'])).resize((64,64))
        batch['chest_VIS'] = batch['img_VIS'].crop(process(**img_NIR_info['chest'])).resize((64,64))
        
        batch['thigh_NIR'] = batch['img_NIR'].crop(process(**img_NIR_info['thigh'])).resize((64,64))
        batch['thigh_VIS'] = batch['img_VIS'].crop(process(**img_NIR_info['thigh'])).resize((64,64))
        
        batch['leg_NIR'] = batch['img_NIR'].crop(process(**img_NIR_info['leg'])).resize((64,128))
        batch['leg_VIS'] = batch['img_VIS'].crop(process(**img_NIR_info['leg'])).resize((64,128))
        
        totensor = transforms.ToTensor()
        for i in batch.keys():
            if i == 'id_NIR' or i == 'id_VIS':
                continue
            batch[i] = totensor(batch[i])
        return batch


# In[ ]:


def createDatasets(yaml_NIR,yaml_VIS,img_NIR_dir,img_VIS_dir,p_test=0.1):
    with open(yaml_NIR,'r') as rf:
        img_NIR_all = yaml.safe_load(rf.read())
    with open(yaml_VIS,'r') as rf:
        img_VIS_all = yaml.safe_load(rf.read())
        
    #假设img_NIR_all和img_VIS_all长度一致
    length = min(len(img_NIR_all),len(img_VIS_all))
    
    img_test_NIR_list = list(img_NIR_all.keys())[:int(length*p_test)]
    img_test_VIS_list = list(img_VIS_all.keys())[:int(length*p_test)]
    img_train_NIR_list = list(img_NIR_all.keys())[int(length*p_test):length]
    img_train_VIS_list = list(img_VIS_all.keys())[int(length*p_test):length]
    #return img_NIR_all,img_VIS_all,img_train_NIR_list,img_train_VIS_list,img_NIR_dir,img_VIS_dir
    return CustomDatasets(img_NIR_all,img_VIS_all,img_train_NIR_list,img_train_VIS_list,img_NIR_dir,img_VIS_dir),CustomDatasets(img_NIR_all,img_VIS_all,img_test_NIR_list,img_test_VIS_list,img_NIR_dir,img_VIS_dir)


# In[ ]:


trainSet,testSet = createDatasets('./images_NIR.yml','./images_VIS.yml','./data/trainB/','./data/trainA/')
train_loader = data.DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
test_loader = data.DataLoader(testSet,1,shuffle=True)


# In[ ]:


def concat_patch(b):
    size = b[0].size()#遮掩新生成的一个tensor，反向传播会在此停止。
    batch = torch.zeros(size[0],size[1]*5,size[2],size[3])
    batch = batch.to(device)
    batch[:,:size[1],:,:] = b[0]
    batch[:,size[1]:size[1]*2,:16,16:48] = b[1]
    batch[:,size[1]*2:size[1]*3,:64,:] = b[2]
    batch[:,size[1]*3:size[1]*4,:64,:] = b[3]
    #print(batch.size(),batch[:,size[1]*4,:,:].size(),b[4].size())
    batch[:,size[1]*4:,:,:] = b[4]
    return batch


# In[ ]:


def deconcat(batch):#如果上面合并时不生成一个新的tensor，直接操作会改变batch的形状，可以用这个转回来
    batch[1] = batch[1][:,:,:16,16:48]
    batch[2] = batch[2][:,:,:64,:]
    batch[3] = batch[3][:,:,:64,:]


# ### 以下定义训练及测试
# 
# - 训练过程：先生成5各VIS图，再把它们合并再生成更清楚的全局VIS图
# - 得到的全军图的size为[batchsize,channels,128,64]
# - 得到的headsize为[batchsize,channels,16,32] ---- 所有的图片均作了一个转置
# - 生成器和判别器的loss均不要取log

# 对于生成器，其前向传播过程为生成器A生成fake1（五张图片），再合并这五张图片然后再生成fake2（一张图片，最终要得到的就是这张），然后fake1的local+fake2传递给生成器B，同理生成fake3和fake4.  
# 同时，fake1和fake2交给判别器打分，我们希望生成的这些图片再判别器中获得的分数越高越好，所以定义损失为||socre-1||，然后反向传播，得到梯度。  
# 同时，fake3，fake4应该与原图越相似越好，所以定义loss: norm(fake-real)，反向传播得到梯度。  
# 两次得到的梯度信息合并然后更新生成器A的参数

# In[ ]:


#如果只是L1番薯，则loss会特别大，可以改用mean(abs(map))
def similarity_loss(real,fake):
    loss = 0
    for i,j in zip(real,fake):
        loss += torch.mean(torch.abs(i-j))
    return loss


# In[ ]:


def score_loss(discrinminator,fake):
    loss = 0
    for i in fake:
        loss += torch.pow(discrinminator(i.expand(-1,3,-1,-1))-1,2) 
    return loss


# In[ ]:


def genernator_train(genernator,merge,discriminator,optim,data_batch):#两个生成器的模型一起放在列表中传入，如[genernator_VIS,genernator_NIR]
    genernator[0].train()
    genernator[1].train()
    merge[0].train()
    merge[1].train()
    discriminator[0].eval()
    discriminator[1].eval()    
    
    VIS_real = [data_batch['img_VIS'],data_batch['head_VIS'],data_batch['chest_VIS'],data_batch['thigh_VIS'],data_batch['leg_VIS']]
    NIR_real = [data_batch['img_NIR'],data_batch['head_NIR'],data_batch['chest_NIR'],data_batch['thigh_NIR'],data_batch['leg_NIR']]
    
    VIS2NIR_fake1 = genernator[0](VIS_real)
    #print(VIS2NIR_fake1[0].shape,VIS2NIR_fake1[1].shape,VIS2NIR_fake1[2].shape,VIS2NIR_fake1[3].shape,VIS2NIR_fake1[4].shape)
    batch_fake = concat_patch(VIS2NIR_fake1)
    #print(VIS2NIR_fake1[0].shape,VIS2NIR_fake1[1].shape,VIS2NIR_fake1[2].shape,VIS2NIR_fake1[3].shape,VIS2NIR_fake1[4].shape)
    VIS2NIR_fake2 = merge[0](batch_fake)
    NIR2VIS_fake3 = genernator[1]([VIS2NIR_fake2,*VIS2NIR_fake1[1:]])
    #print(NIR2VIS_fake3[0].shape,NIR2VIS_fake3[1].shape,NIR2VIS_fake3[2].shape,NIR2VIS_fake3[3].shape,NIR2VIS_fake3[4].shape)
    batch_fake = concat_patch(NIR2VIS_fake3)
    NIR2VIS_fake4 = merge[1](batch_fake)
    loss_A_consistency = similarity_loss([NIR2VIS_fake4,*NIR2VIS_fake3],[VIS_real[0],*VIS_real])
    loss_A_discriminator = score_loss(discriminator[0],[VIS2NIR_fake2,*VIS2NIR_fake1])
    del VIS2NIR_fake1,VIS2NIR_fake2,NIR2VIS_fake3,NIR2VIS_fake4,batch_fake
    
    NIR2VIS_fake1 = genernator[1](NIR_real)
    batch_fake = concat_patch(NIR2VIS_fake1)
    NIR2VIS_fake2 = merge[1](batch_fake)
    VIS2NIR_fake3 = genernator[0]([NIR2VIS_fake2,*NIR2VIS_fake1[1:]])
    batch_fake = concat_patch(VIS2NIR_fake3)
    VIS2NIR_fake4 = merge[0](batch_fake)
    loss_B_consistency = similarity_loss([VIS2NIR_fake4,*VIS2NIR_fake3],[NIR_real[0],*NIR_real])
    loss_B_discriminator = score_loss(discriminator[1],[NIR2VIS_fake2,*NIR2VIS_fake1])
    del NIR2VIS_fake1,NIR2VIS_fake2,VIS2NIR_fake3,VIS2NIR_fake4
    del VIS_real,NIR_real
    
    loss_genernator = loss_A_consistency*2+loss_A_discriminator+loss_B_consistency*2+loss_B_discriminator
    optim[0].zero_grad()
    optim[1].zero_grad()
    optim[2].zero_grad()
    optim[3].zero_grad()
    #print(loss_genernator)
    loss_genernator.mean().backward()
    optim[0].step()
    optim[1].step()
    optim[2].step()
    optim[3].step()
    
    return loss_A_consistency,loss_A_discriminator,loss_B_consistency,loss_B_discriminator


# In[ ]:


def discriminator_loss(discriminator,fake,real):
    loss = 0
    for i,j in zip(fake,real):
        #print(j.expand(-1,3,-1,-1).shape)
        loss += (torch.pow(discriminator(j.expand(-1,3,-1,-1))-1,2)+torch.pow(discriminator(i.expand(-1,3,-1,-1)),2))
    return loss


# In[ ]:


def discriminator_train(discriminator,genernator,merge,optim,data_batch):#参数的传递方式同上
    discriminator[0].train()
    discriminator[1].train()
    genernator[0].eval()
    genernator[1].eval()
    merge[0].eval()
    merge[1].eval()
    VIS_real = [data_batch['img_VIS'],data_batch['head_VIS'],data_batch['chest_VIS'],data_batch['thigh_VIS'],data_batch['leg_VIS']]
    NIR_real = [data_batch['img_NIR'],data_batch['head_NIR'],data_batch['chest_NIR'],data_batch['thigh_NIR'],data_batch['leg_NIR']]
    
    VIS2NIR_fake1 = genernator[0](VIS_real)
    for i in VIS2NIR_fake1:
        i.detach()
    batch_fake = concat_patch(VIS2NIR_fake1)
    VIS2NIR_fake2 = merge[0](batch_fake).detach()
    loss = discriminator_loss(discriminator[0],[VIS2NIR_fake2,*VIS2NIR_fake1],[VIS_real[0],*VIS_real])
    del VIS2NIR_fake1,VIS2NIR_fake2
    optim[0].zero_grad()
    loss.mean().backward()
    optim[0].step()
    
    
    NIR2VIS_fake1 = genernator[1](NIR_real)
    for i in NIR2VIS_fake1:
        i.detach()
    batch_fake = concat_patch(NIR2VIS_fake1)
    NIR2VIS_fake2 = merge[1](batch_fake).detach()
    loss_1 = discriminator_loss(discriminator[1],[NIR2VIS_fake2,*NIR2VIS_fake1],[NIR_real[0],*NIR_real])
    del NIR2VIS_fake1,NIR2VIS_fake2,batch_fake,VIS_real,NIR_real
    optim[1].zero_grad()
    loss_1.mean().backward()
    optim[1].step()
    
    return loss,loss_1


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def test(genernator,merge,data_batch,epoch):
    genernator[0].eval()
    genernator[1].eval()
    merge[0].eval()
    merge[1].eval()
    transform = transforms.ToPILImage()
    for i in data_batch.keys():
        data_batch[i] = data_batch[i].to(device)
    with torch.no_grad():
        VIS_real = [data_batch['img_VIS'],data_batch['head_VIS'],data_batch['chest_VIS'],data_batch['thigh_VIS'],data_batch['leg_VIS']]
        NIR_real = [data_batch['img_NIR'],data_batch['head_NIR'],data_batch['chest_NIR'],data_batch['thigh_NIR'],data_batch['leg_NIR']]
    
        VIS2NIR_fake1 = genernator[0](VIS_real)
        batch_fake = concat_patch(VIS2NIR_fake1)
        VIS2NIR_fake2 = merge[0](batch_fake)
    
        NIR2VIS_fake3 = genernator[1]([VIS2NIR_fake2,*VIS2NIR_fake1[1:]])
        batch_fake = concat_patch(NIR2VIS_fake3)
        NIR2VIS_fake4 = merge[1](batch_fake)
    
    fig=plt.figure(figsize=(16, 4))
    columns = 6
    rows = 1
    fig.add_subplot(rows, columns, 1)
    #print(VIS_real[0].shape)
    plt.imshow(transform(VIS_real[0][0].cpu()))
    fig.add_subplot(rows, columns, 2)
    plt.imshow(transform(VIS2NIR_fake1[0][0].cpu()))
    fig.add_subplot(rows, columns, 3)
    plt.imshow(transform(VIS2NIR_fake2[0].cpu()))
    fig.add_subplot(rows, columns, 4)
    plt.imshow(transform(NIR2VIS_fake3[0][0].cpu()))
    fig.add_subplot(rows, columns, 5)
    plt.imshow(transform(NIR2VIS_fake4[0].cpu()))
    fig.add_subplot(rows, columns, 6)
    plt.imshow(transform(NIR_real[0][0].cpu()))
    plt.tight_layout()       
    plt.savefig('./process_image/image_VIS2NIR_A_%d.jpg'%(epoch+1))
    plt.show()
    
    del VIS2NIR_fake1,VIS2NIR_fake2,NIR2VIS_fake3,NIR2VIS_fake4,batch_fake
    with torch.no_grad():
        NIR2VIS_fake1 = genernator[1](NIR_real)
        batch_fake = concat_patch(NIR2VIS_fake1)
        NIR2VIS_fake2 = merge[1](batch_fake)
        VIS2NIR_fake3 = genernator[0]([NIR2VIS_fake2,*NIR2VIS_fake1[1:]])
        batch_fake = concat_patch(VIS2NIR_fake3)
        VIS2NIR_fake4 = merge[0](batch_fake)
    
    fig=plt.figure(figsize=(16, 4))
    columns = 6
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(transform(NIR_real[0][0].cpu()))
    fig.add_subplot(rows, columns, 2)
    plt.imshow(transform(NIR2VIS_fake1[0][0].cpu()))
    fig.add_subplot(rows, columns, 3)
    plt.imshow(transform(NIR2VIS_fake2[0].cpu()))
    fig.add_subplot(rows, columns, 4)
    plt.imshow(transform(VIS2NIR_fake3[0][0].cpu()))
    fig.add_subplot(rows, columns, 5)
    plt.imshow(transform(VIS2NIR_fake4[0].cpu()))
    fig.add_subplot(rows, columns, 6)
    plt.imshow(transform(VIS_real[0][0].cpu()))
    plt.tight_layout()       
    plt.savefig('./process_image/image_NIR2VIS_B_%d.jpg'%(epoch+1))
    plt.show()


# ### 以下为正式训练的流程

# In[ ]:


import torch.optim as optim


# In[ ]:


optimzer_gen_A_VIS2NIR = optim.RMSprop(genernator_VIS2NIR.parameters(),lr=0.0002)
optimzer_gen_B_NIR2VIS = optim.RMSprop(genernator_NIR2VIS.parameters(),lr=0.0002)
optimzer_mer_A_NIR2NIR = optim.RMSprop(merge_NIR2NIR.parameters(),lr=0.0002)
optimzer_mer_B_VIS2VIS = optim.RMSprop(merge_VIS2VIS.parameters(),lr=0.0002)
optimzer_dis_A = optim.Adam(discriminator_A_NIR.parameters(),lr = 0.001)
optimzer_dis_B = optim.Adam(discriminator_B_VIS.parameters(),lr=0.001)


# In[39]:


for epoch in range(EPOCHS):
    test([genernator_VIS2NIR,genernator_NIR2VIS],[merge_NIR2NIR,merge_VIS2VIS],next(iter(test_loader)),epoch)
    for batch in train_loader:
        for i in batch.keys():
            batch[i] = batch[i].to(device)
        loss_A_consistency,loss_A_discriminator,loss_B_consistency,loss_B_discriminator = genernator_train([genernator_VIS2NIR,genernator_NIR2VIS],[merge_NIR2NIR,merge_VIS2VIS],[discriminator_A_NIR,discriminator_B_VIS],[optimzer_gen_A_VIS2NIR,optimzer_gen_B_NIR2VIS,optimzer_mer_A_NIR2NIR,optimzer_mer_B_VIS2VIS],batch)
        loss,loss_1 = discriminator_train([discriminator_A_NIR,discriminator_B_VIS],[genernator_VIS2NIR,genernator_NIR2VIS],[merge_NIR2NIR,merge_VIS2VIS],[optimzer_dis_A,optimzer_dis_B],batch)
    print('epoch: {}/{},loss_A_consistency: {},loss_A_discriminator: {},loss_B_consistency: {},loss_B_discriminator: {}'.format(epoch+1,EPOCHS,loss_A_consistency,loss_A_discriminator.item(),loss_B_consistency,loss_B_discriminator.item()))
    print('discriminator_A_VIS_loss:{},discriminator_B_NIR_loss{}'.format(loss.item(),loss_1.item()))
    if epoch%10 == 0:
        torch.save(genernator_VIS2NIR.state_dict(), './process_image/genernator_VIS2NIR.pkl')
        torch.save(genernator_NIR2VIS.state_dict(), './process_image/genernator_NIR2VIS.pkl')
        torch.save(merge_NIR2NIR.state_dict(), './process_image/merge_NIR2NIR.pkl')
        torch.save(merge_VIS2VIS.state_dict(), './process_image/merge_VIS2VIS.pkl')
        torch.save(discriminator_A_NIR.state_dict(), './process_image/discriminator_A_NIR.pkl')
        torch.save(discriminator_B_VIS.state_dict(), './process_image/discriminator_B_VIS.pkl')


# In[ ]:





# In[ ]:


test([genernator_VIS2NIR,genernator_NIR2VIS],[merge_NIR2NIR,merge_VIS2VIS],next(iter(test_loader)))

