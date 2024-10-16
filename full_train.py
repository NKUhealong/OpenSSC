import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import *
from utils import *
from dataset import *
from AdaptFormer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed(2023)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def main():
    save_path = './new/'  
    model_name = 'skin'
    lr = 0.0003
    img_size = 448
    epochs = 65
    batch_size = 24
    num_class = 7
    #model = resnet50(img_size, num_classes=num_class)
    model = ViT_AdptFormer_Model(img_size, num_classes=num_class)
    
    
    for name,p in model.named_parameters():
        if 'fc_head'  in name:   # adaptmlp
            p.requires_grad = True
        elif 'prompt'  in name or 'ssf'  in name or 'adaptmlp'  in name: 
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    
    train_dataset,val_dataset, test_dataset = generate_dataset('skin', img_size, ratio = 0.6) # 'pathology' skin
    estimator = Estimator(num_class)
    scaler = torch.cuda.amp.GradScaler()
    train(lr,model_name,epochs,batch_size,model,train_dataset,test_dataset,estimator,scaler)
   
    print('This is the performance of the best validation model')
    checkpoint = os.path.join(save_path,model_name+'_best_val_weight.pt')
    evaluate(lr,batch_size, model, checkpoint, test_dataset, estimator, test = True)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, model_name+'_final_weight.pt')
    evaluate(lr,batch_size, model, checkpoint, test_dataset, estimator, test = False)

def train(learning_rate,model_name,epochs,batch_size, model, train_dataset, val_dataset, estimator, scaler):
    save_path = './new/'    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()    
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=False,num_workers=4,
                              drop_last=True,pin_memory=True, sampler=sampler_train)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=4, pin_memory=True)
    model = model.to(device)
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa,avg_recall,avg_precision  = 0, 0, 0,0,0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        estimator.reset()
        progress = enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device).long()
            
            with torch.cuda.amp.autocast():
                y_pred= model(X)
                loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            #nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
    
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc,avg_kappa = estimator.get_accuracy(4),estimator.get_kappa(4)
            avg_recall,avg_precision = estimator.get_rec_pre(4)
            curr_lr = optimizer.param_groups[0]['lr']
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.5f}, acc: {:.4f}, kappa: {:.4f}, rec: {:.4f}, pre: {:.4f}, LR: {:.6f}'.format(current_time,\
                       epoch + 1, epochs, avg_loss, avg_acc, avg_kappa,avg_recall,avg_precision, curr_lr)
        print(message)

        eval(model, val_loader, estimator, device)
        acc, kappa = estimator.get_accuracy(4), estimator.get_kappa(4)
        recall,precision = estimator.get_rec_pre(4)
        print('valid accuracy: {}, kappa: {}, recall: {}, precision: {}'.format(acc,kappa,recall,precision))
        indicator = acc+recall+precision
        if indicator > max_indicator:
            torch.save(model.state_dict(),os.path.join(save_path, model_name+'_best_val_weight.pt'))
            max_indicator = indicator

        adjust_learning_rate(learning_rate, epochs, optimizer, epoch)
    torch.save(model.state_dict(), os.path.join(save_path, model_name+'_final_weight.pt'))

if __name__ == '__main__':
    main()
        