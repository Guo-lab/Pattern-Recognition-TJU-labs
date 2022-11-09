import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

import matplotlib.pyplot as plt


# 数据集下载，dataloader构建
# 补充代码时需注意：path，transform，shuffle，batch_size
#### TODO ####
path = "../数据集/CIFAR10"
transform = transforms.Compose([
    #transforms.ColorJitter(),
    transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),  # default 0.5 
    transforms.ToTensor(),
    #transforms.RandomGrayscale(p=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = 32

cifar_trn = datasets.CIFAR10(root=path, train=True, transform=transform, download=True)
cifar_train = DataLoader(cifar_trn, batch_size=batch_size, shuffle=True)
#cifar_trn = datasets.CIFAR10(root=path, train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
#cifar_train = DataLoader(cifar_trn, batch_size=batch_size, shuffle=True)

#g = torch.Generator()
#g.manual_seed(0)
cifar_tst = datasets.CIFAR10(root=path, train=False, transform=transform, download=True)
cifar_test = DataLoader(cifar_tst, batch_size=batch_size, shuffle=True)#, generator=g)
#cifar_tst = datasets.CIFAR10(root=path, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
#cifar_test = DataLoader(cifar_tst, batch_size=batch_size, shuffle=True)#, generator=g)







# Residual block
# 补充代码时需注意：卷积核设置，bn写法，shortcut写法
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=[1,1]):
        super(ResBlk, self).__init__()
        #### TODO ####
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_out) 
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch_out)
        if ch_out == ch_in:
            self.extra = nn.Sequential()
        else:
            self.extra = nn.Sequential(
                # 1*1卷积和bn
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(ch_out)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

# ResNet18
# 补充代码时需注意：block构建
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        #### TODO ####
        self.conv1 = nn.Sequential(
            # 第一层卷积和bn
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 4 blocks
        self.blk1 = ResBlk(64,  64,  [1, 1])
        self.blk1_= ResBlk(64,  64,  [1, 1])
        self.blk2 = ResBlk(64,  128, [2, 1])
        self.blk2_= ResBlk(128, 128, [1, 1])
        self.blk3 = ResBlk(128, 256, [2, 1])
        self.blk3_= ResBlk(256, 256, [1, 1])
        self.blk4 = ResBlk(256, 512, [2, 1])
        self.blk4_= ResBlk(512, 512, [1, 1])
        # 全连接
        self.outlayer = nn.Linear(512, 10) # num_classes = 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk1_(x)
        x = self.blk2(x)
        x = self.blk2_(x)
        x = self.blk3(x)
        x = self.blk3_(x)
        x = self.blk4(x)
        x = self.blk4_(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)

# Combine figure parts in MNIST main.py
train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []
train_counter, test_counter = [], []
train_epoch, test_epoch = [], []

def train(total_epoch, train_log_interval, test_log_interval):
    # 补充损失函数设置和优化器设置
    #### TODO ####
    loss_fun = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    #optimizer = optim.ASGD(model.parameters(), lr=0.001)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    loss_list = []
    for epoch in range(total_epoch):
        running_loss = 0.0
        train_running_loss = 0.0
        
        train_total_correct = 0
        train_total_num = 0
        
        # 补充训练部分代码，注意课件里的7步
        #### TODO ####
        model.train()
        for batchidx, (inputs, labels) in enumerate(cifar_train):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: [Batch Size * class numbers], labels: [Batch Size]
            
            #// For Other loss
            #//outputs2 = outputs.argmax(dim=1).to(torch.float64)
            #//loss = loss_fun(outputs2, labels)
            #//loss.requires_grad_(True)
            loss = loss_fun(outputs, labels)
            
            train_running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_total_correct += torch.eq(pred, labels).float().sum().item()
            train_total_num += inputs.size(0)
            
            # 观察损失变化，下面5行代码不用改
            running_loss += loss.item()
            loss_list.append(loss.item())
            if (batchidx + 1) % 100 == 0:
                print('epoch = %d , batch = %d , loss = %.6f' % (epoch + 1, batchidx + 1, running_loss / 100))
                running_loss = 0.0
                
            loss.backward()
            optimizer.step()
            if (batchidx + 1) % train_log_interval == 0:
                ##### For Figure ####
                train_loss_list.append((train_running_loss / train_log_interval))
                train_counter.append((batchidx * batch_size) + (epoch * len(cifar_train.dataset)))
                train_running_loss = 0.0
                #####################
        
        train_acc = train_total_correct / train_total_num
        train_acc_list.append(train_acc)
        train_epoch.append(epoch)
        #//print("Accuracy of the network on the 50000 train images:%.5f %%" % (100 * train_acc))    
        
        if epoch % 80 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
                
        test(epoch, test_log_interval)



def test(epoch, test_log_interval):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0

        #%https://zhuanlan.zhihu.com/p/431283706
        loss_fun = nn.CrossEntropyLoss(reduction='sum')
        
        test_running_loss = 0.0
        
        # 补充测试部分代码
        for batchidx, (inputs, labels) in enumerate(cifar_test):
            #### TODO ####
            inputs, labels =  inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            pred = outputs.argmax(dim=1)
            total_correct += torch.eq(pred, labels).float().sum().item()
            total_num += inputs.size(0)       
            
            test_running_loss += loss_fun(outputs, labels).item()
            #print(test_running_loss)
            
            #if (batchidx + 1) % test_log_interval == 0:
            #    print((test_running_loss / test_log_interval))
            #    test_loss_list.append((test_running_loss / test_log_interval))
            #    test_counter.append((batchidx * batch_size) + (epoch * len(cifar_test.dataset)))
            #    test_running_loss = 0.0
                
        test_running_loss /= len(cifar_test.dataset)
        test_loss_list.append(test_running_loss)

        # 观察测试集精度变化，下面3行代码不用改
        acc = total_correct / total_num
        test_acc_list.append(acc)
        test_epoch.append(epoch)
        print("Accuracy of the network on the 10000 test images:%.5f %%" % (100 * acc))
        print("===============================================")


if __name__ == '__main__':
       
    import numpy as np
    ## TODO ##
    epoch_num = 20
    train_log_interval = 100
    test_log_interval = 300
    
    test_counter = [i * len(cifar_train.dataset) for i in range(epoch_num)]
    
    start = time.time()
    train(epoch_num, train_log_interval, test_log_interval)
    end = time.time()
    
    fig = plt.figure()
    #plt.plot(test_counter, test_loss_list)
    plt.scatter(test_counter, test_loss_list, color='red')
    plt.plot(train_counter, train_loss_list)
    plt.legend(['testng loss', 'training loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('likelihood loss')
    plt.savefig("./figure_1.jpg")
    
    
    fig = plt.figure()
    plt.plot(test_epoch, test_acc_list)
    plt.plot(train_epoch, train_acc_list)
    plt.legend(['testng accuracy', 'training accuracy'], loc='upper right')
    plt.xlabel('number of epoch')
    plt.ylabel('Accuracy')
    plt.savefig("./figure_2.jpg")
    
    #print(test_counter)
    #print(test_loss_list)
    #print(train_acc_list)
    #np.savetxt('1.txt', test_counter)
    #np.savetxt('2.txt', test_loss_list)
    #np.savetxt('3.txt', train_counter)
    #np.savetxt('4.txt', train_loss_list)
    #np.savetxt('5.txt', test_epoch)
    #np.savetxt('6.txt', train_acc_list)
    #np.savetxt('7.txt', test_acc_list)
    
    examples = enumerate(cifar_test)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data, example_targets = example_data.to(device), example_targets.to(device)

    with torch.no_grad():
        output = model(example_data)

    
    fig = plt.figure()
    
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i].cpu().transpose(0, 2).transpose(0, 1), interpolation='none')
        plt.title("Prediction: {}\nGroundTruth: {}".format(
            output.data.max(1, keepdim=True)[1][i].item(), 
            example_targets[i].cpu()
            )
        )
        plt.xticks([])
        plt.yticks([])
        
    plt.savefig("./figure_3.jpg")
    
    #print(end - start)
    