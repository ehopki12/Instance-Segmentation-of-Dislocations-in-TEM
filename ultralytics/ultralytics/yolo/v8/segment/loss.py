import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from ultralytics.DislocationMaskProcessing.get_length_box import get_endpoints as get_disdata

def get_bce_loss(pred,target):
    """
    get bce loss to segment each dislocation 
    pred has shape [N,C,H,W] 
    and output is [N]
    """
    bce_loss = torch.nn.BCELoss(reduction="none")
    l = bce_loss(pred,target)
    lmean = l.mean([2,3])
    return lmean
def get_dice_loss(pred, target):
    smooth = 1.
    intersection = (pred * target).float().sum(axis=[2,3])
    cardinality = pred.sum(axis=[2,3]) + target.sum(axis=[2,3])
    return 1.0 - (2.*intersection+smooth) / (cardinality + smooth)

def get_IOU_loss(pred, target):
    smooth = 1.
    intersection = (pred * target).float().sum(axis=[2,3])
    cardinality = pred.sum(axis=[2,3]) + target.sum(axis=[2,3])
    union = cardinality - intersection
    return 1.0 - (intersection+smooth) / (union + smooth)

class DiceBCELoss(nn.Module):
    # source : https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    def __init__(self, bce_weight=0.5, dice_weight=0.5,from_logits=True):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.from_logits = from_logits
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if(self.from_logits):
            inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.bce_weight*BCE + self.dice_weight*dice_loss
        return Dice_BCE


def get_len(match_mask,threshold, area_limits=(20,20000)):
    with torch.no_grad():
        match_mask[match_mask < threshold] = 0.0
        match_mask[match_mask > threshold] = 1.0
        match_mask = match_mask.clone().detach().cpu().numpy()
        _ , _ , l_pred, _ ,_,c_pred = get_disdata(match_mask, area_limits=area_limits,show_bboxes=False,) 
    return l_pred, c_pred   

def physical_loss(ytrue,ypred,num_dislocation,base_loss="dice",threshold=0.4,test_mode=False,debug = False):
    """
    Example ytrue = [1,4,512,512]
            ypred = [1,4,512,512]
            num_dislocation = [2]
            There are only two dislocations in the image and can have a maximum of 4 dislocations 
            
            For first dislocation y_true_0 we need to find the corresponding predicte dislocation. So 
            we calculate loss between pairs {(y_true_0,y_pred_0),
                                            (y_true_0,y_pred_1),
                                            (y_true_0,y_pred_2),
                                            (y_true_0,y_pred_3)}
            the minimum loss would give us the correspoding predicted dislocation, lets say it is y_pred_2, 
            Now we remove this from the ypred so that ypred now has shape [1,3,512,512], We will do the same 
            thing for the y_true_1 which might have y_pred_3. 
            Loss of the image is now ( Loss(y_true_0,y_pred_2) + Loss(y_true_1,y_pred_3) )/num_dislocation 
            
    
    """
    if(base_loss=="dice"):
        criterion = get_dice_loss
    elif(base_loss=="iou"):
        criterion = get_IOU_loss
    else:
        criterion = get_bce_loss
        
    mean_loss = 0.0   
    mean_empty_loss = 0.0 
    acc = np.array([0.0,0.0,0.0])
    img_size = ytrue.shape[2] # image size 
    if (debug): print(ypred.shape, ytrue.shape,num_dislocation)
    if (debug): plt.figure(figsize=(60,60))
    for k in range(num_dislocation.shape[0]): # each image in batch
        n_dis = ytrue.shape[1] # Number of classes         
        y_pred = ypred[0,0,:,:].float().view(1,1,img_size,img_size)
        y_true = ytrue[0,0,:,:].float().view(1,1,img_size,img_size)
        ml = criterion(y_pred,y_true).mean() # dummy loss 
        y_pred = ypred[k,:,:,:].float().view(1,1,img_size,img_size)
        temp = np.array([0.0,0.0,0.0]) 
        selected_dis = list(np.arange(0,n_dis))
        for i in range(1): # matching masks for minimum loss
            if (debug): print(f"calculation for {k} image and dislocation {i}".center(100,"-"))
            y_true = ytrue[k,i,:,:].float().view(1,1,img_size,img_size)
            y_true = torch.broadcast_to(y_true, (1,n_dis,img_size,img_size))
            loss = criterion(y_pred,y_true)[0]
            min_loss, min_indices = torch.min(loss,axis=0)
            if (debug): print((i+1,min_indices+1, loss, min_loss))
            if(test_mode):
                try:
                    match_mask = y_pred[0,min_indices,:,:].view(img_size,img_size).float() # getting the matched mask 
                    l_pred, c_pred = get_len(match_mask.clone().detach(),threshold) # getting length and center of dislocation
                    if(debug): 
                        plt.subplot(6,6,i+1)
                        plt.imshow(match_mask,cmap="gray")                
                    match_mask = ytrue[k,i,:,:].view(img_size,img_size).float()
                    l_true, c_true = get_len(match_mask.clone().detach(),threshold)

                    if (debug ): print("Predicted", l_pred, c_pred, "True", l_true, c_true)
                    if (len(l_pred) == 1): # Predicted mask has exactly one dislocations 
                        temp[1]= temp[1] + 1.0
                        if(debug): print(abs(l_true[0]-l_pred[0])/l_true[0])
                        if(l_true[0]>0.0):
                            error = abs(l_true[0]-l_pred[0])/l_true[0]
                            if(error < 2):
                                temp[2]= temp[2] + abs(1.0 - error) # 1 - Error 
                    # # print(f"new modified metric if more than one instance")
                    # max_l = np.array(l_pred).max()
                    # temp[2]= temp[2] + abs(1.0 - abs(l_true[0]-max_l)/l_true[0])

                    temp[0]= temp[0] + 1.0

                except:
                    if (debug): print(100*"-")
                    if (debug): print("Error in post calculation".center(100,"-"))
                    if (debug): print(100*"-")

            if(debug): 
                plt.title(f"loss: {min_loss:1.4f} {temp}",size=32)
                
            n_dis = n_dis-1
            a = torch.cat([y_pred[0,:min_indices,:,:], y_pred[0,min_indices+1:,:,:]])
            selected_dis.pop(min_indices)
            if (i==0):
                ml = min_loss.mean()
            else:
                ml += min_loss.mean()
                
        mean_loss+= ml/num_dislocation[k]
        acc = acc + temp/num_dislocation[k].cpu().numpy()

        if(debug):
            match_mask,_ = torch.max(ypred[k,:,:,:],axis=0)
            plt.subplot(6,6,i+2)
            plt.imshow(match_mask,cmap="gray")
            plt.title("all dislocations",size=32)

            match_mask,_ = torch.max(ytrue[k,:,:,:],axis=0)
            plt.subplot(6,6,i+3)
            plt.imshow(match_mask,cmap="gray")
            plt.title("True ",size=32)
            plt.show()            
    return mean_loss/num_dislocation.shape[0],acc/num_dislocation.shape[0] 