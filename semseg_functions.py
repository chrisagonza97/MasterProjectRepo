import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import tqdm
import pandas as pd
# import pytorch_lightning as pl
import os
import copy
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from transunet import TransUNet

def load_imgs_labels(train_dir="./train",val_dir="./val"):
    train_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(train_dir,"imgs","*.png"))))))
    X_train=torch.FloatTensor(train_imgs).permute((0,3,1,2))/255
    
    val_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(val_dir,"imgs","*.png"))))))
    X_val=torch.FloatTensor(val_imgs).permute((0,3,1,2))/255
    
    train_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(train_dir,"labels","*.png"))))))
    Y_train=torch.LongTensor(train_lbls)
    
    val_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(val_dir,"labels","*.png"))))))
    Y_val=torch.LongTensor(val_lbls)
    
    return X_train,Y_train,X_val,Y_val


def train_model(X_train, Y_train, X_val, Y_val, save=True, n_epochs=200, model_key="unet", encoder_name="resnet18", path_dir="./seg_models", device=None, fold_num=None):
    if device is None:  # Auto-detect GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}", flush=True) 

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    train_data = TensorDataset(X_train, Y_train)  
    val_data = TensorDataset(X_val, Y_val)

    train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True)
    train_dataloader_ordered=DataLoader(train_data,batch_size=8,shuffle=False)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)

    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    model=dict(unet=smp.Unet,
                fpn=smp.FPN,
                ).get(model_key, smp.Unet)
    #defualt learning rate is 0.001
    model=model(classes=3,in_channels=3, encoder_name=encoder_name, encoder_weights=None).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0002)
    #class_weight=compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.numpy().flatten()), y=Y_train.numpy().flatten())
    #class_weight=torch.FloatTensor(class_weight).to(device)
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.cpu().numpy().flatten()), y=Y_train.cpu().numpy().flatten())
    class_weight = torch.FloatTensor(class_weight).to(device)

    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True)
    min_loss=np.inf
    best_epoch = 0
    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)
        for i,(x,y_true) in enumerate(train_dataloader):
            
            x=x.to(device)
            y_true=y_true.to(device)
            optimizer.zero_grad()

            y_pred=model(x)
            loss=loss_fn(y_pred,y_true)

            loss.backward()
            optimizer.step()

            with open("training_log1.txt", "a") as log_file:
                log_file.write(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(), 3)}\n")


        # validation set
        model.train(False)
        with torch.no_grad():
            val_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                
                x=x.to(device)
                y_true=y_true.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y_true)
                val_loss.append(loss.item())

            val_loss=np.mean(val_loss)
            print(f"Val: Epoch {epoch}, Loss: {round(val_loss,3)}")
            if val_loss < min_loss:
                best_epoch = epoch
                min_loss=val_loss
                best_model=copy.deepcopy(model.state_dict())
                if save:
                    #with open(path_dir + f'/{epoch}.{i}_model.pkl', "w") as f:
                        #torch.save(model.state_dict(), path_dir + f'/{epoch}.{i}_model.pkl')
                    if fold_num is None:
                        with open(path_dir + f'/bestunet_model.pkl', "w") as f:
                            torch.save(model.state_dict(), path_dir + f'/bestunet_model.pkl')
                    else:
                        with open(path_dir + f'/bestunet_model_fold{fold_num}.pkl', "w") as f:
                            torch.save(model.state_dict(), path_dir + f'/bestunet_model_fold{fold_num}.pkl')
    #print whcih epoch had the best model
    print(f"Best model found at epoch {best_epoch}")

    model.load_state_dict(best_model)
    return model, min_loss


def train_model_transunet(X_train, Y_train, X_val, Y_val, save=True, n_epochs=200, model_key="unet", encoder_name="resnet18", path_dir="./seg_models", device=None, lr=0.01, momentum=0.9, weight_decay=0.0001, fold_num=None):    
    if device is None:  # Auto-detect GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}", flush=True) 


    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    train_data = TensorDataset(X_train, Y_train)  
    val_data = TensorDataset(X_val, Y_val)

    train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True)
    train_dataloader_ordered=DataLoader(train_data,batch_size=8,shuffle=False)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)

    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name

    model = TransUNet(in_channels=3, classes=3).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr = lr,momentum=0.9, weight_decay=0.0001)
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.cpu().numpy().flatten()), y=Y_train.cpu().numpy().flatten())
    class_weight = torch.FloatTensor(class_weight).to(device)

    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True)

    min_loss=np.inf
    best_model = None
    best_epoch = 0
    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)
        for i,(x,y_true) in enumerate(train_dataloader):
            
            x=x.to(device)
            y_true=y_true.to(device)
            optimizer.zero_grad()

            y_pred=model(x)
            loss=loss_fn(y_pred,y_true)

            loss.backward()
            optimizer.step()

            #print(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(),3)}")
            with open("training_log.txt", "a") as log_file:
                log_file.write(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(), 3)}\n")


        # validation set
        model.train(False)
        with torch.no_grad():
            val_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                
                x=x.to(device)
                y_true=y_true.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y_true)
                val_loss.append(loss.item())

            val_loss=np.mean(val_loss)
            #print(f"Val: Epoch {epoch}, Loss: {round(val_loss,3)}")
            with open("training_log.txt", "a") as log_file:
                log_file.write(f"Val: Epoch {epoch}, Loss: {round(val_loss, 3)}\n")
            if val_loss < min_loss:
                best_epoch = epoch
                min_loss=val_loss
                best_model=copy.deepcopy(model.state_dict())
                if save:
                        if fold_num is None:
                            with open(path_dir + f'/best_transunet_model.pkl', "w") as f:
                                torch.save(model.state_dict(), path_dir + f'/best_transunet_model.pkl')
                        else:
                            with open(path_dir + f'/best_transunet_model_fold{fold_num}.pkl', "w") as f:
                                torch.save(model.state_dict(), path_dir + f'/best_transunet_model_fold{fold_num}.pkl')
    #print which epoch had the best model
    #print(f"Best model found at epoch {best_epoch}")
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"Best model found at epoch {best_epoch}\n")

    model.load_state_dict(best_model)
    return model, min_loss
    
"""def train_model_transunet(X_train, Y_train, X_val, Y_val,
                          lr=0.001, momentum=0.9, weight_decay=0.0005,
                          num_epochs=25, fold_num=None):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    model = TransUNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=8)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for i, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb.squeeze(1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

           
            print(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(), 3)}")

        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb.squeeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if fold_num is not None:
                torch.save(model.state_dict(), f"model_fold{fold_num}.pt")
            else:
                torch.save(model.state_dict(), "best_transunet_model.pt")

        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return model, train_losses, val_losses, best_val_loss"""

def train_model_transunet_sweep(X_train, Y_train, X_val, Y_val, save=True, n_epochs=200, model_key="unet", encoder_name="resnet18", path_dir="./seg_models", device=None, lr=0.01, momentum=0.9, weight_decay=0.0001):    
    if device is None:  # Auto-detect GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}", flush=True) 

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    train_data = TensorDataset(X_train, Y_train)  
    val_data = TensorDataset(X_val, Y_val)

    train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True)
    train_dataloader_ordered=DataLoader(train_data,batch_size=8,shuffle=False)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)

    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name

    optimizer=torch.optim.SGD(model.parameters(),lr = lr,momentum=0.9, weight_decay=0.0001)
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.cpu().numpy().flatten()), y=Y_train.cpu().numpy().flatten())
    class_weight = torch.FloatTensor(class_weight).to(device)

    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True)

    min_loss=np.inf
    best_model = None
    best_epoch = 0
    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)
        for i,(x,y_true) in enumerate(train_dataloader):
            
            x=x.to(device)
            y_true=y_true.to(device)
            optimizer.zero_grad()

            y_pred=model(x)
            loss=loss_fn(y_pred,y_true)

            loss.backward()
            optimizer.step()

            print(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(),3)}")

        # validation set
        model.train(False)
        with torch.no_grad():
            val_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                
                x=x.to(device)
                y_true=y_true.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y_true)
                val_loss.append(loss.item())

            val_loss=np.mean(val_loss)
            print(f"Val: Epoch {epoch}, Loss: {round(val_loss,3)}")
            if val_loss < min_loss:
                best_epoch = epoch
                min_loss=val_loss
                best_model=copy.deepcopy(model.state_dict())
                if save:
                    model_name = f'best_transunet_lr{lr}_mom{momentum}_wd{weight_decay}.pkl'
                    model_path = os.path.join(path_dir, model_name)

                    #with open(path_dir + f'/{epoch}.{i}_transunet_model.pkl', "w") as f:
                        #torch.save(model.state_dict(), path_dir + f'/{epoch}.{i}_transunet_model.pkl')
                    with open(model_path, "wb") as f:  
                        torch.save(model.state_dict(), f)
    #print whcih epoch had the best model
    print(f"Best model found at epoch {best_epoch}")

    model.load_state_dict(best_model)
    return model, min_loss, best_epoch

def make_predictions(X_val,model=None,save=True,path_dir = "./seg_models", model_key="unet", encoder_name="resnet18", device="cpu"):
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    predictions=[]
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    # load most recent saved model
    if model is None and save:
        model=dict(unet=smp.Unet,
                fpn=smp.FPN).get(model_key, smp.Unet)
        model=model(classes=3,in_channels=3, encoder_name=encoder_name, encoder_weights=None)
        model_list=sorted(glob.glob(path_dir + '/bestunet_model.pkl'), key=os.path.getmtime)
        model.load_state_dict(torch.load(model_list[-1],map_location="cpu"))
    print(f"Loading model from: {model_list[-1]}")
    model=model.to(device)
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions

def make_predictions_transunet(X_val,model=None,save=True,path_dir = "./seg_models", model_key="transunet", encoder_name="resnet18", device="cuda"):
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    predictions=[]
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    # load most recent saved model
    if model is None and save:
        model = TransUNet(in_channels=3, classes=3)
        model_list=sorted(glob.glob(path_dir + '/*_transunet_model.pkl'), key=os.path.getmtime)
        model.load_state_dict(torch.load(model_list[-1],map_location="cpu"))
        print(f"Loading model from: {model_list[-1]}")
    model=model.to(device)
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions

def make_predictions_transunet_sweep(X_val,model=None,save=True,path_dir = "./seg_models", model_key="transunet", encoder_name="resnet18", device="cpu", lr=None,momentum=None,weight_decay=None):
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    predictions=[]
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    # load most recent saved model
    if model is None and save:
        filename = f'best_transunet_lr{lr}_mom{momentum}_wd{weight_decay}.pkl'
        model_path = os.path.join(path_dir, filename)
        model = TransUNet(in_channels=3, classes=3)
        model.load_state_dict(torch.load(model_path,map_location="cpu"))
    print(f"Loading model from: {model_path}")
    model=model.to(device)
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions