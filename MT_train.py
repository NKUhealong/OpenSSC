import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import cv2, torch, random, itertools, time,datetime

from model import *
from utils import *
from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed(2023)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train():
    base_lr = 0.0003
    num_classes = 5
    batch_size = 64
    model_name = 'Res50_MTlong_cancer'
    save_path = './new/'  
    image_size = (448,448)
    consistency = 0.1
    max_epoch = 75
        
    model = create_resnet_model(image_size[0], num_classes,False)   
    model.cuda()
    ema_model = create_resnet_model(image_size[0], num_classes,True)
    ema_model.cuda()
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    
    labeled_bs = int(batch_size/2)  # 'pathology' skin
    batch_sampler,train_dataset,ID_test_dataset,OOD_test_dataset=get_dataset_splits(image_size,batch_size,'skin',[0,1,2,3,4],0.1, True)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,num_workers=4, pin_memory=True)
    ID_test_loader = DataLoader(ID_test_dataset,batch_size=batch_size,num_workers=4, pin_memory=True)
    print('=> train len:', len(train_loader))
    
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr,weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    iter_num, max_indicator, max_iterations = 0, 0, max_epoch * len(train_loader)
    avg_loss,avg_acc, avg_kappa,avg_recall,avg_precision  = 0,0,0,0,0
    estimator = Estimator(num_classes)
    for epoch_num in range(max_epoch):
        estimator.reset()
        train_loss = 0
        start_time = time.time()
        model.train()
        
        for i_batch, (X,Y) in enumerate(train_loader):
            images, labels = X.cuda(), Y.cuda()
            unlabel_inputs = images
            with torch.cuda.amp.autocast():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    unlabel_output = ema_model(unlabel_inputs)
                    unlabel_output_soft = torch.softmax(unlabel_output, dim=1)

                sup_loss = ce_loss(outputs[:labeled_bs],labels[:labeled_bs].long())
                consist_loss = torch.mean((outputs_soft-unlabel_output_soft)**2)

                consist_weight = get_current_consistency_weight(consistency,iter_num,max_iterations)
                loss = sup_loss  + consist_weight*consist_loss

                estimator.update(outputs[:labeled_bs],labels[:labeled_bs]) 
                train_loss = train_loss + loss.detach().cpu().numpy()
                avg_loss = train_loss / (i_batch + 1)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update() 
            update_ema_variables(model, ema_model, 0.99, iter_num)
            adjust_learning_rate(base_lr, max_epoch, optimizer, epoch_num)
            curr_lr = optimizer.param_groups[0]['lr']
            iter_num = iter_num + 1
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_acc = estimator.get_accuracy(4)
        avg_kappa = estimator.get_kappa(4)
        avg_recall,avg_precision = estimator.get_rec_pre(4)
        message = 'train_loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, rec: {:.4f}, pre: {:.4f}, LR: {:.6f}'.format(avg_loss, avg_acc, avg_kappa,avg_recall,avg_precision, curr_lr)
        print('Epoch: {} / {} '.format(epoch_num, max_epoch), 'Training time {}'.format(total_time_str),'Initial LR {:4f}'.format(base_lr)) 
        print(message) 
        
        ##  evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval(model, ID_test_loader, estimator, device)
        acc, kappa = estimator.get_accuracy(4), estimator.get_kappa(4)
        recall,precision = estimator.get_rec_pre(4)
        print('Student valid acc: {}, kappa: {}, recall: {}, precision: {}'.format(acc,kappa,recall,precision))
        indicator = acc+recall+precision
        if indicator > max_indicator:
            torch.save(model.state_dict(),os.path.join(save_path, model_name+'_best_val_weight.pt'))
            max_indicator = indicator
            
        eval(ema_model, ID_test_loader, estimator, device)
        acc, kappa = estimator.get_accuracy(4), estimator.get_kappa(4)
        recall,precision = estimator.get_rec_pre(4)
        print('Teacher valid acc: {}, kappa: {}, recall: {}, precision: {}'.format(acc,kappa,recall,precision))
        
    torch.save(model.state_dict(), os.path.join(save_path, model_name+'_final_weight.pt'))
    
    print('This is the performance of the best validation model')
    checkpoint = os.path.join(save_path,model_name+'_best_val_weight.pt')
    evaluate(base_lr,batch_size, model, checkpoint, ID_test_dataset, estimator, test = True)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, model_name+'_final_weight.pt')
    evaluate(base_lr,batch_size, model, checkpoint, ID_test_dataset, estimator, test = False)

train()