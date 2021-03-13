import sys
sys.path.append('..')
import numpy as np

import torch
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from torch import optim
from torch.utils.data import DataLoader

from imageio import imwrite
from skimage import img_as_ubyte

import unet.model as model
from unet.dataset import KidDataset

import random
# import unet.evaluation as eva

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skip_case = [160] # 第160号，数据比例【 512, 796】不能读取


x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5],[0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
            if epoch_loss < 0.001:
                break
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    return model

# 训练模型
def train():
    um = model.UNet(1, 1).to(device)
    batch_size = args.batch_size
    # 损失函数
    criterion = torch.nn.BCELoss() # 梯度下降 交叉熵应用于二分类时候的特殊形式
    # criterion = torch.nn.CrossEntropyLoss() #交叉熵损失函数
    # 梯度下降
    # optimizer = optim.Adam(um.parameters(),lr=0.0001)  # model.parameters():Returns an iterator over module parameters
    sum_dice = 0
    fo=open('dice.txt','w')
    # 使用十折交叉验证
    for k in range(10):
        optimizer = optim.Adam(um.parameters())
        for j in range(10):
            if k==j: #第j组做验证组
                continue
            for i in range(21):#使用前200组数据训练，使用200-210组数据进行验证
                case_id = 21*j + i
                if case_id in skip_case:
                    print('第%d数据已跳过' %case_id)
                    continue
                print("第%d_%d次训练,训练集%d" % (k,i,case_id))
                # 加载数据集
                kid_dataset = KidDataset(case_id, transform=x_transform, target_transform=y_transform)
                dataloader = DataLoader(kid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
                # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
                # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
                # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
                # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
                um=train_model(um, criterion, optimizer, dataloader)
                torch.save(um.state_dict(), 'weights_%d_%d.pth' % (k,batch_size))  # 返回模型的所有内容
        single_dice = 0
        for i in range(21): #第k组做验证
            single_dice = sum_dice+verify(um,21*k + i)
        print('第 %d 次验证，dice为 %.3f' %(k,single_dice/21))
        fo.write('第 %d 次验证，dice为 %.3f' %(k,single_dice/21))
        sum_dice = sum_dice + single_dice/21
    print('十折验证后，dice为 %.3f' %(k,sum_dice/10))
    fo.write('十折验证后，dice为 %.3f' %(k,sum_dice/10))
    fo.close()

def verify(model,case_id):
    kid_dataset = KidDataset(case_id, transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(kid_dataset)  # batch_size默认为1
    model.eval()
    with torch.no_grad():
        num = 0
        case_dice = 0
        for x, label in dataloaders:
            y = model(x)
            case_dice = case_dice + dice(torch.squeeze(y)>0.5,label)
            num = num + 1
        return case_dice / num


def dice(prec,label):
    # 平滑变量
    smooth = 1
    # 将宽高 reshape 到同一纬度
    input_flat = prec.view(1, -1)
    targets_flat = label.view(1, -1)
    # 计算交集
    intersection = input_flat * targets_flat
    d = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
    # 计算一个批次中平均每张图的损失
    return d.sum()

def test():
    um = model.UNet(1, 1)
    um.load_state_dict(torch.load(args.weight, map_location='cpu'))
    # print(um.state_dict())
    print('load dataset')
    # 这里计算dice，因此会有y_transform
    dic = 0
    for k in range(1,2):
        kid_dataset = KidDataset(k, transform=x_transform, target_transform=y_transform)
        dataloaders = DataLoader(kid_dataset)  # batch_size默认为1
        um.eval()
        with torch.no_grad():
            num = 0
            case_dice = 0
            # for x, _ in dataloaders:
            for x, label in dataloaders:
                print('start %d_%d picture' % (k,num))
                y = um(x)
                # print(y)
                y=torch.squeeze(y)
                # print(y)
                y=y>0.5
                # print(y)
                case_dice = case_dice + dice(y,label)
                print('dic',dic)
                # img_y = y.numpy()
                # print(img_y)
                # fpath_seg = ("./predict/{:05d}.png".format(num))
                # ubimg_y = img_as_ubyte(img_y)
                # imwrite(str(fpath_seg), ubimg_y)
                num = num + 1
            case_dice_avg = case_dice / num
            print("%d Dice is %.3f" %(k,case_dice_avg))
            dic = dic + case_dice_avg
        avg_dice = dic / 10
        print("Dice is %.3f" %avg_dice)

def prec():
    um = model.UNet(1, 1)
    um.load_state_dict(torch.load(args.weight, map_location='cpu'))
    # print(um.state_dict())
    print('load dataset')
    case_id = 220
    kid_dataset = KidDataset(case_id, transform=x_transform)
    dataloaders = DataLoader(kid_dataset)  # batch_size默认为1
    um.eval()
    with torch.no_grad():
        num = 0
        dic = 0
        for x, _ in dataloaders:
            print('print %d picture' % num)
            y = um(x)
            # print(y)
            y=torch.squeeze(y)
            # print(y)
            y=y>0.5
            # print(y)
            # dic = dic + dice(y,label)
            # print('dic',dic)
            img_y = y.numpy()
            # print(img_y)
            fpath_seg = ("./predict/{:05d}.png".format(num))
            ubimg_y = img_as_ubyte(img_y)
            imwrite(str(fpath_seg), ubimg_y)
            num = num + 1

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('action', type=str, help='train or test')  # 添加参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()

    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    elif args.action == 'prec':
        prec()
