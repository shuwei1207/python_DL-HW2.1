# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:27:09 2020

@author: SeasonTaiInOTA
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(         
            input_size= 10,
            hidden_size=128,  
            num_layers=1,          
            batch_first=True,
        )
        self.out = nn.Linear(128,2)

    def forward(self, x):
        r_out,h_n = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        return out
    
    
if __name__ == '__main__':
    
    EPOCH = 100              
    BATCH_SIZE = 1
    TIME_STEP = 1         
    INPUT_SIZE = 10       
    LR = 0.001               
    
    L = 10
    label = []
    seq = []
    device = 'cpu'
    
    loss_batch = []
    accuracy = []
    plt_loss = []
    plt_acc = []
    a = 0
    
    print('-----------data processing------------')
    for i in range(df.shape[0]):
        for j in range(df.shape[1]-L-1):
            seq.append(list (df.iloc[i,j:j+L]))
            label.append(1 if df.iloc[i,j+L+1] >df.iloc[i,j+L] else 0)
            
    seq = torch.FloatTensor(seq)
    label = torch.FloatTensor(label)
    dataset = Data.TensorDataset(seq,label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('-----------start processed------------')
    
    lstm = LSTM()
    
    
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)   
    loss_func = nn.CrossEntropyLoss()
    
    print('-----------start training------------')
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):        
            b_x = b_x.view(-1, 1, 10)              
            output = lstm(b_x)                              
            #print(output)
            b_y = torch.tensor(b_y, dtype=torch.long, device=device)
            loss = loss_func(output, b_y)                  
            
            optimizer.zero_grad()
            loss.backward() # backpropagation
            optimizer.step() # apply gradients
            
            
            pred_y = torch.max(output, 1)[1].data.numpy()
            if(pred_y == b_y.data.numpy()):
                accuracy.append(1)
            else:
                accuracy.append(0)
            loss_batch.append(loss.data.numpy())
            
        average_loss = sum(loss_batch)/len(loss_batch)
        acc = sum(accuracy)/len(accuracy)
        plt_loss.append(average_loss)
        plt_acc.append(acc)
        print('Epoch: ', epoch, '| train loss: %.4f' % average_loss, '| train accuracy: %.4f' % acc)
    print('-----------end training------------')