import os
import cv2, torch, random, itertools, time,datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss, MSELoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import *
from utils import *
from dataset import *

set_random_seed(2024)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train():
    base_lr = 0.0003
    num_classes = 5
    batch_size = 46
    model_name = 'Vit_openOODtemp_skin'
    save_path = './new/'  
    image_size = (320,320)
    consistency = 1.0
    max_epoch = 50
        
    model = create_open_resnet_model(image_size[0], num_classes,False)
    
    model.cuda()
    ema_model = create_open_resnet_model(image_size[0], num_classes,True)
    ema_model.cuda()
    for name,p in model.named_parameters():
        if 'fc_head'  in name or 'our_mlp_proj'  in name or 'mb_classifiers'  in name:   # adaptmlp fc_transform2
            p.requires_grad = True
        elif 'prompt'  in name or 'fc_transform'  in name or 'adaptmlp'  in name or 'fc_transform2'  in name: 
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    
    labeled_bs = int(batch_size/2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'pathology' skin
    batch_sampler,train_dataset,ID_test_dataset,OOD_test_dataset=get_dataset_splits(image_size,batch_size,'DDR',[0,1,2,3,4],0.3, True)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,num_workers=4, pin_memory=True)
    ID_test_loader = DataLoader(ID_test_dataset,batch_size=batch_size,num_workers=4, pin_memory=True)
    print('=> train len:', len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr,weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    estimator = Estimator(num_classes)
    avg_loss, avg_acc, avg_kappa, avg_recall, avg_precision = 0,0,0,0,0,
    max_iterations = max_epoch * len(train_loader)
    unsup_mb_loss, init_thresh, unsup_loss, iter_num, max_indicator = 0, 0.9, 0, 0, 0
    
    for epoch_num in range(max_epoch):
        estimator.reset()
        train_loss, start_time = 0, time.time()
        model.train()
        for i_batch, (X,Y) in enumerate(train_loader):
            images, all_labels = X.cuda(), Y.cuda()
            labeled_image,labels = images[0:labeled_bs],all_labels[0:labeled_bs]
            unlabel_image = images[labeled_bs:]
            
            with torch.cuda.amp.autocast(): # Student Model
                label_out,prompt_out,label_features,_,logits_mb_label = model(labeled_image,labels,True)
                label_out_soft = torch.softmax(label_out, dim=1)
                
                unlabel_out,unprompt_out,unlabel_features,ID_proto,logits_mb_unlabel = model(unlabel_image)
                unlabel_out_soft = torch.softmax(unlabel_out, dim=1)

                with torch.no_grad(): # Teacher Model
                    ema_label_out,emaprompt_out,ema_label_feature,ema_ID_proto,ema_logits_mb_label = ema_model(labeled_image,labels,True)
                    ema_label_out_soft = torch.softmax(ema_label_out, dim=1)
                    
                    ema_unlabel_out,ema_unprompt_out,ema_unlabel_feature,ema_unID_proto,ema_logits_mb_unlabel = ema_model(unlabel_image)
                    ema_unlabel_out_soft = torch.softmax(ema_unlabel_out, dim=1)

                cut_data,cut_labels = cutmix(labeled_image, unlabel_image, labels, alpha=0.6)
                cut_out,_,cut_features,cut_queue,_ = model(cut_data)
                cut_loss = ce_loss(cut_out,labels.long())
                sup_mb_loss =  mb_sup_loss(logits_mb_label, labels)
                sup_loss = ce_loss(label_out,labels.long()) + ce_loss(prompt_out,labels.long())+cut_loss +sup_mb_loss
                #probs = torch.clamp(unlabel_out_soft, 1e-10, 1.0)  
                #entropy = -torch.mean(torch.sum(probs * torch.log2(probs), dim=1)) 
                KL_div = torch.mean(torch.sum(ema_unlabel_out_soft*torch.log2(ema_unlabel_out_soft/torch.clamp(unlabel_out_soft, 1e-10, 1.0)),dim=-1))
                
                unlabel_consist_loss = F.mse_loss(unlabel_out_soft,ema_unlabel_out_soft)+F.mse_loss(label_out_soft,ema_label_out_soft)+\
                                       F.mse_loss(label_features,ema_label_feature)+F.mse_loss(unlabel_features,ema_unlabel_feature)+\
                                       F.mse_loss(logits_mb_label,ema_logits_mb_label)+F.mse_loss(logits_mb_unlabel,ema_logits_mb_unlabel)+KL_div
                
                consist_weight = get_current_consistency_weight(consistency,iter_num,max_iterations)
                thresh = get_current_consistency_weight(init_thresh,iter_num,max_iterations)
                ################ open set ###############
                if epoch_num>=10:
                    similarity = (unlabel_features @ ID_proto.transpose(-2, -1)* 256 ** -0.5)
                    similarity = nn.Softmax(dim=-1)(similarity)
                    
                    '''
                    mb_out = torch.softmax(logits_mb_unlabel.view(logits_mb_unlabel.size(0), 2, -1), 1)
                    mb_out = mb_out[:,1,:]
                    similarity = similarity * mb_out
                    '''
                    max_values, max_indices = torch.max(similarity, dim=-1)
                    index = max_values>thresh
                    ID_unlabel_pseudo_label = torch.argmax(ema_unlabel_out_soft[index], dim=1)
                    unsup_loss = ce_loss(unlabel_out[index],ID_unlabel_pseudo_label.long())
                    pseudo_labels = torch.argmax(ema_unlabel_out_soft, dim=-1)
                    unsup_mb_loss =  mb_sup_loss(logits_mb_unlabel, pseudo_labels)
                ################ open set ###############
                loss = sup_loss+ consist_weight*(unlabel_consist_loss+unsup_loss+unsup_mb_loss)

                estimator.update(label_out,labels) 
                train_loss = train_loss + loss.detach().cpu().numpy()
                avg_loss = train_loss / (i_batch + 1)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update() 
                                                  
            update_ema_variables(model, ema_model, alpha=0.99, global_step=iter_num)
            adjust_learning_rate(base_lr, max_epoch, optimizer, epoch_num)
            curr_lr = optimizer.param_groups[0]['lr']
            iter_num = iter_num + 1
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_acc, avg_kappa = estimator.get_accuracy(4), estimator.get_kappa(4)
        avg_recall,avg_precision = estimator.get_rec_pre(4)
        message = 'train_loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, rec: {:.4f}, pre: {:.4f}, LR: {:.6f}'.format(avg_loss, avg_acc, avg_kappa,avg_recall,avg_precision, curr_lr)
        print('Epoch: {} / {} '.format(epoch_num, max_epoch), 'Training time {}'.format(total_time_str),'Initial LR {:4f}'.format(base_lr)) 
        print(message) 
        
        ##  evaluation
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
    
    ## ID Test
    estimator = Estimator(num_classes)
    print('This is the performance of the best validation model')
    checkpoint = os.path.join(save_path,model_name+'_best_val_weight.pt')
    evaluate(base_lr,batch_size, model, checkpoint, ID_test_dataset, estimator, test = True)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, model_name+'_final_weight.pt')
    evaluate(base_lr,batch_size, model, checkpoint, ID_test_dataset, estimator, test = False)
    
    ## OOD Test
    OOD_test_loader = DataLoader(OOD_test_dataset,batch_size=batch_size,num_workers=4, pin_memory=True)
    estimator = Estimator(num_classes+1)
    print('This is the OOD performance of the best validation model')
    checkpoint = os.path.join(save_path,model_name+'_best_val_weight.pt')
    OODevaluate(model, checkpoint, OOD_test_loader, estimator)
    print('This is the OOD performance of the final model:')
    checkpoint = os.path.join(save_path, model_name+'_final_weight.pt')
    OODevaluate(model, checkpoint, OOD_test_loader, estimator)

train() 