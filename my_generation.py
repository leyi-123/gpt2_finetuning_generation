from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# gpt2预训练模型
model = GPT2LMHeadModel.from_pretrained("E:/gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
model.eval()
# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
# 损失函数
# top k选择
def select_top_k(predictions,k = 10):
    predicted_index = random.choice(predictions.sort(descending = True)[1][:k]).item()
    return predicted_index
path = 'E:/dialogue/nonpersona/ijcnlp_dailydialog/train/train/dialogues_train.txt'
f = open(path,'r',encoding='utf-8')
l = f.read()
l = l.split('\n')
L = list()
for i in l:
    if len(i)>=1:
        L.append(i)
l = L
i_sum = 0
sum = 0
length = 0
epoch = 3
a_loss = 0
# 对训练数据的处理
# 这里没有用torch里面的DataLoader, TensorDataset, 需改进
for num in range(epoch):
    i_sum = 0
    for data in l:
        traindata = data.split('__eou__')
        # "<|endoftext|>" 表示一句话的结束
        traindata[0] = traindata[0] + " <|endoftext|>"
        input_tokens = tokenizer.encode(traindata[0])
        input_tokens = torch.tensor([input_tokens]).to(device)
        tokens = tokenizer.encode(traindata[0])
        i_sum += 1
        for i in range(1,len(traindata)):
            traindata[i] = traindata[i] + " <|endoftext|>"
            # 只计算response的损失
            if (i%2 == 0) and (len(traindata[i]) >= 1):
                t = tokenizer.encode(traindata[i])
                tokens += t
                # 输入最大长度不超过100
                if len(tokens) > 100:
                    tokens = tokens[-99:]
                input_tokens = torch.tensor([tokens]).to(device)
            else:
                # 计算损失的过程
                STD = list()
                PRE = torch.from_numpy(np.zeros(50257)).to(device)
                PRE = PRE.view(1,-1)
                C = tokenizer.encode(traindata[i])
                length = 0
                for c in C:
                    sum += 1
                    length += 1
                    output = model(input_tokens)
                    predictions = output[0][0, -1, :]
                    predicted_index = select_top_k(predictions, k=10)
                    tokens += [predicted_index]
                    if len(tokens) > 100:
                        tokens = tokens[-99:]
                    input_tokens = torch.tensor([tokens]).to(device)
                    predictions = F.softmax(predictions, dim=0, dtype=torch.double).to(device)
                    predictions = predictions.view(1, -1)
                    STD.append(c)
                    PRE = (torch.cat((PRE, predictions), 0)).to(device)
                    if (predicted_index == 50256) or (length >= 20):
                        break
                    if len(STD) >= 4:
                        PRE = PRE[1:,:]
                        STD = torch.tensor(STD,dtype=torch.long)
                        loss = F.cross_entropy(PRE, STD)
                        a_loss += loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        STD = list()
                        PRE = torch.from_numpy(np.zeros(50257)).to(device)
                        PRE = PRE.view(1, -1)
                if (len(STD) != 0) and (int(torch.sum(PRE)) != 0):
                    PRE = PRE[1:, :]
                    STD = torch.tensor(STD, dtype=torch.long)
                    loss = F.cross_entropy(PRE,STD)
                    a_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        if i_sum%100 == 0:
            print("epoch:",num,",train loss:",a_loss/sum)
            sum = 0
            a_loss = 0
# 在测试集计算损失
# 具体过程与上类似
model.save_pretrained("/home1/zhenli/")
t_path = '/home1/zhenli/dialogues_test.txt'
t_f = open(path,'r',encoding='utf-8')
t_l = t_f.read()
t_l = t_l.split('\n')
t_L = list()
for i in t_l:
    if len(i)>=1:
        t_L.append(i)
t_l = t_L
sum = 0
test_result = 0
a_loss = 0
for data in t_l:
    testdata = data.split('__eou__')
    testdata[0] = testdata[0] + " <|endoftext|>"
    input_tokens = tokenizer.encode(testdata[0])
    input_tokens = torch.tensor([input_tokens]).to(device)
    tokens = tokenizer.encode(testdata[0])
    for i in range(1, len(testdata)):
        testdata[i] = testdata[i] + " <|endoftext|>"
        if (i % 2 == 0) and (len(testdata[i]) >= 1):
            t = tokenizer.encode(testdata[i])
            tokens += t
            if len(tokens) > 100:
                tokens = tokens[-99:]
            input_tokens = torch.tensor([tokens]).to(device)
        else:
            STD = list()
            PRE = torch.from_numpy(np.zeros(50257)).to(device)
            PRE = PRE.view(1, -1)
            C = tokenizer.encode(testdata[i])
            length = 0
            for c in C:
                sum += 1
                length = 0
                output = model(input_tokens)
                predictions = output[0][0, -1, :]
                predicted_index = select_top_k(predictions, k=10)
                tokens += [predicted_index]
                if len(tokens) > 100:
                    tokens = tokens[-99:]
                input_tokens = torch.tensor([tokens]).to(device)
                predictions = F.softmax(predictions, dim=0, dtype=torch.double).to(device)
                predictions = predictions.view(1, -1)
                STD.append(c)
                PRE = (torch.cat((PRE, predictions), 0)).to(device)
                if (predicted_index == 50256) or (length >= 20):
                    break
                if STD.shape[0] >= 4:
                    PRE = PRE[1:, :]
                    STD = torch.tensor(STD, dtype=torch.long)
                    loss = F.cross_entropy(PRE, STD)
                    a_loss += loss
                    STD = list()
                    PRE = torch.from_numpy(np.zeros(50257)).to(device)
                    PRE = PRE.view(1, -1)
            PRE = PRE[1:, :]
            STD = torch.tensor(STD, dtype=torch.long)
            loss = F.mse_loss(PRE, STD)
            a_loss += loss
print("test loss:", a_loss / sum)
