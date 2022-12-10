import PIL
import json
import pandas as pd
import os
import ast
import numpy as np
import cv2
from tqdm import tqdm
import random
import time

from sklearn import metrics

from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2





def seed_everything(seed_value=4995):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class HandwrittenDataset(Dataset):
    
    def __init__(self, df, visible_char_mapping, transform = None, image_resize = (1000,500), bad_classes = None):
        
        self.data = list(df.itertuples(index=False))
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        
        self.visible_char_mapping = visible_char_mapping
        
        self.labels=np.zeros((len(self.data),len(visible_char_mapping)))
        
        for tup_idx, tup in enumerate(self.data):
            visible_latex_chars = tup.visible_latex_chars
            labels = [*map(self.visible_char_mapping.get, visible_latex_chars)]
            
            for char in labels:
                self.labels[tup_idx, char - 1] = 1
        
        if bad_classes:
            
            new_data = []
            for tup_idx, tup in enumerate(self.data):
                visible_latex_chars = tup.visible_latex_chars
                for label in visible_latex_chars:
                    if label in bad_classes:
                        new_data.append(tup)
                        break
            
            self.data = new_data
            self.labels=np.zeros((len(self.data),len(visible_char_mapping)))
            for tup_idx, tup in enumerate(self.data):
                visible_latex_chars = tup.visible_latex_chars
                labels = [*map(self.visible_char_mapping.get, visible_latex_chars)]

                for char in labels:
                    self.labels[tup_idx, char - 1] = 1
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        sample = self.data[index]
        label = self.labels[index,:].astype(np.float32)
        labels = torch.tensor(label)

        
        f_name = sample.filename
        image = PIL.Image.open(f_name).convert("RGB")
        
        
        if self.transform:
            image = self.transform(image)

        return image, labels, f_name
    
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    images = [img for img,_,_ in batch]
    labels = torch.stack([lab for _, lab,_ in batch])
    f_names = [f_n for _,_, f_n in batch]
    
    return images, labels, f_names

def create_data_frame(raw_data, image_path):

    data = {}
    data['latex'] = []
    data['seq_len'] = []
    data['latex_string'] = []
    data['visible_latex_chars'] = []
    data['filename'] = []
    data['width'] = []
    data['height'] = []
    data['xmins_raw'] = []
    data['xmaxs_raw'] = []
    data['ymins_raw'] = []
    data['ymaxs_raw'] = []
    data['xmins'] = []
    data['xmaxs'] = []
    data['ymins'] = []
    data['ymaxs'] = []
    
    for image in raw_data:
        data['latex_string'].append(image['latex'])
        data['latex'].append(image['image_data']['full_latex_chars'])
        data['seq_len'].append(len(image['image_data']['full_latex_chars']))
        data['visible_latex_chars'].append(image['image_data']['visible_latex_chars'])
        data['filename'].append(os.path.join(image_path, image['filename']))
        data['xmins_raw'].append(image['image_data']['xmins_raw'])
        data['xmaxs_raw'].append(image['image_data']['xmaxs_raw'])
        data['ymins_raw'].append(image['image_data']['ymins_raw'])
        data['ymaxs_raw'].append(image['image_data']['ymaxs_raw'])
        data['xmins'].append(image['image_data']['xmins'])
        data['xmaxs'].append(image['image_data']['xmaxs'])
        data['ymins'].append(image['image_data']['ymins'])
        data['ymaxs'].append(image['image_data']['ymaxs'])
        
        data['width'].append(image['image_data']['width'])
        data['height'].append(image['image_data']['height'])


    df = pd.DataFrame.from_dict(data)
    return df

def load_data(path = 'data/all_data.csv'):
    if not os.path.isfile(path):
        df = pd.DataFrame()
        for i in range(1,11):
            print(f'data/batch_{i}/JSON/kaggle_data_{i}.json')
            with open(file=f'data/batch_{i}/JSON/kaggle_data_{i}.json') as f:
                raw_data = json.load(f)
            sub_df = create_data_frame(raw_data, f'data/batch_{i}/background_images')
            df = df.append(sub_df)
        df.to_csv(path)
        df = pd.read_csv(path).drop(columns = 'Unnamed: 0')
    else:
        df = pd.read_csv(path).drop(columns = 'Unnamed: 0')

    list_cols = ['xmins_raw', 'xmaxs_raw', 'ymins_raw', 'ymaxs_raw', 'xmins', 'xmaxs', 'ymins', 'ymaxs']
    for c in list_cols:
        df[c] = df[c].apply(json.loads)

    df['latex'] = df['latex'].replace("'\\\\", "'\\")
    df['latex'] = df['latex'].apply(ast.literal_eval)
    
    #vocab = df['latex'].explode().unique().tolist()[0]
    df['visible_latex_chars'] = df['visible_latex_chars'].replace("'\\\\", "'\\")
    df['visible_latex_chars'] = df['visible_latex_chars'].apply(ast.literal_eval)
    
    with open(file=f'data/extras/visible_char_map.json') as f:
        visible_char_map = json.load(f)
    
    return df, visible_char_map

def split_dataframe(df):
    X_train, X_test = train_test_split(df, test_size=0.20, random_state=4995)
    
    return X_train, X_test

def prepare_data(batch_size = 32):
    
    df, visible_char_map = load_data()
    
    # num_classes = len(visible_char_map)
    
    l = []
    for i in df['visible_latex_chars'].tolist():
        for j in i:
            l.append(j)
    
    classes = sorted(list(set(l)))
    num_classes = len(set(l))
    
    visible_char_map = {}
    for idx, symbol in enumerate(classes):
        visible_char_map[symbol] = idx + 1 
    
    return df, visible_char_map, num_classes, classes

def build_dataloaders(df, visible_char_map, df2 = None,  batch_size = 32, bad_classes = None):

    data_transforms = {
      'train': transforms.Compose([
        #  transforms.Resize((896,896)),
        #  transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
        #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
       #   transforms.Resize((896,896)),
          #transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
       #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }
    
    if df2 is None:
        train_df, val_df = split_dataframe(df)
    else:
        train_df, val_df = df, df2
    
    train_dataset = HandwrittenDataset(train_df, visible_char_map, transform = data_transforms['train'], bad_classes = bad_classes)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=1, collate_fn = collate_fn)
    
    val_dataset = HandwrittenDataset(val_df, visible_char_map, transform = data_transforms['val'], bad_classes = bad_classes)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=1, collate_fn = collate_fn)
    
    return train_loader, val_loader

class FasterRCNNBackboneModel(nn.Module):
    
    def __init__(self, faster_rcnn, num_classes = 54):
        super(FasterRCNNBackboneModel, self).__init__()
        
        self.trans = faster_rcnn.transform
        self.backbone = faster_rcnn.backbone
        
        self.adapt = nn.AdaptiveAvgPool2d((1,1))
        
        input_feat = 256
        
        self.classifier = nn.Linear(input_feat, num_classes)
        
    def forward(self, x):
        out = self.trans(x)
        out = self.backbone(out[0].tensors)['0']
        
        out = self.adapt(out).squeeze().squeeze()
        
        out = self.classifier(out)
        
        return out
        
def train_loop(model, train_loader, loss_fn, optimizer, scheduler, num_classes = 54, fine_tune = False):
    
    model.train()
    
    train_loss_list = []
    print("Train loop")
    
    concat_pred = np.zeros((1, num_classes))
    concat_labels = np.zeros((1, num_classes))
    avgprecs = np.zeros(num_classes)
    
    for i, data in enumerate((train_loader)):
        optimizer.zero_grad()
        images, targets, _ = data
        #images = images.to(DEVICE)
        
        images = list(image.to(DEVICE) for image in images)
        targets = targets.to(DEVICE)
        
        output = model(images)
        
        cpuout= output.detach().to('cpu')
        pred_scores = cpuout.numpy() 
        concat_pred = np.append(concat_pred, pred_scores, axis = 0)
        concat_labels = np.append(concat_labels, targets.cpu().numpy(), axis = 0)
        
        loss = loss_fn(output, targets)
        
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        
        loss.backward()
        optimizer.step()
        if fine_tune:
            scheduler.step()
        
        if i % 100 == 0:
            print(f'Batch: {i} of {len(train_loader)}. Loss: {loss_value}. Mean so far: {np.mean(train_loss_list)}. Mean of 100: {np.mean(train_loss_list[-100:])}')
            
    concat_pred = concat_pred[1:,:]
    concat_labels = concat_labels[1:,:]

    for c in range(num_classes):   
        avgprecs[c] =  metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])
        
    measure = np.mean(avgprecs)

    return np.mean(train_loss_list), measure, train_loss_list

def val_loop(model, val_loader, loss_fn, num_classes = 54):
    
    model.eval()
    
    val_loss_list = []

    concat_pred = np.zeros((1, num_classes))
    concat_labels = np.zeros((1, num_classes))
    avgprecs = np.zeros(num_classes)
    
    print("Validation loop")
    
    with torch.no_grad():
        for i, data in enumerate((val_loader)):
            images, targets, _ = data

            images = list(image.to(DEVICE) for image in images)
            #images = images.to(DEVICE)

            targets = targets.to(DEVICE)

            output = model(images)
            
            cpuout= output.detach().to('cpu')
            pred_scores = cpuout.numpy() 
            concat_pred = np.append(concat_pred, pred_scores, axis = 0)
            concat_labels = np.append(concat_labels, targets.cpu().numpy(), axis = 0)
        
            loss = loss_fn(output, targets)
            
            loss_value = loss.item()
            val_loss_list.append(loss_value)
            if i % 100 == 0:
                print(f'Batch: {i} of {len(val_loader)}. Loss: {loss_value}. Mean so far: {np.mean(val_loss_list)}. Mean of 100: {np.mean(val_loss_list[-100:])}')
            
    loss_mean = np.mean(val_loss_list)
    print("Eval loss:",loss_mean)

    concat_pred = concat_pred[1:,:]
    concat_labels = concat_labels[1:,:]

    for c in range(num_classes):   
        avgprecs[c]=  metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])
        
    measure = np.mean(avgprecs)
        
    return loss_mean, measure, val_loss_list

def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs = 5,  model_name = 'resnet_re_train', save_path = 'models', fine_tune = False):
    
    train_losses = []
    all_train_losses = []
    train_avg_prec_list = []
    val_losses = []
    val_avg_prec_list = []
    all_val_losses = []
    
    best_val_loss = 0
    best_val_avg_prec = 0
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        tic = time.time()
        train_loss, train_avg_prec, train_loss_un_aggregated = train_loop(model, train_loader, loss_fn, optimizer, scheduler, fine_tune = fine_tune)
        train_losses.append(train_loss)
        train_avg_prec_list.append(train_avg_prec)
        all_train_losses.extend(train_loss_un_aggregated)
        
        print("Train loss:", train_loss, train_avg_prec)
        print(f"Train loop took {time.time()-tic}")
        tic = time.time()
        val_loss, val_avg_prec, eval_loss_un_aggregated = val_loop(model, val_loader, loss_fn)
        print("Validation loss:", val_loss, val_avg_prec)
        print(f"Validation loop took {time.time()-tic}")
        val_losses.append(val_loss)
        val_avg_prec_list.append(val_avg_prec)
        all_val_losses.extend(eval_loss_un_aggregated)
        if not fine_tune:
            scheduler.step()
        
        with open(f'faster_rcnn_train_losses_no_aggregate_{fine_tune}.txt', 'w') as f:
            for line in all_train_losses:
                f.write(f"{line}\n")
        with open('faster_rcnn_train_losses_epoch.txt', 'w') as f:
            for line in train_losses:
                f.write(f"{line}\n")
        
        with open(f'faster_rcnn_eval_losses_no_aggregate_{fine_tune}.txt', 'w') as f:
            for line in all_val_losses:
                f.write(f"{line}\n")
        with open(f'faster_rcnn_eval_losses_epoch_{fine_tune}.txt', 'w') as f:
            for line in val_losses:
                f.write(f"{line}\n")
                
        with open(f'faster_rcnn_train_avg_prec_epoch_{fine_tune}.txt', 'w') as f:
            for line in train_avg_prec_list:
                f.write(f"{line}\n")
                
        with open(f'faster_rcnn_eval_avg_prec_epoch_{fine_tune}.txt', 'w') as f:
            for line in val_avg_prec_list:
                f.write(f"{line}\n")
                
        
        
        if not best_val_avg_prec:
            best_val_avg_prec = val_avg_prec
            
            model_name_pt = model_name+'.pt'
            PATH = os.path.join(save_path, model_name_pt)
            model.to('cpu')
            #torch.save(model.state_dict(), PATH)
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, PATH)
            model.to(DEVICE)
            
        else:
            if val_avg_prec > best_val_avg_prec:
                best_val_loss = val_avg_prec
                
                model_name_pt = model_name+'.pt'
                PATH = os.path.join(save_path, model_name_pt)
                model.to('cpu')
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, PATH)
                model.to(DEVICE)
    
    return train_losses, val_losses, train_avg_prec_list, val_avg_prec_list

if __name__ == "__main__":
    
    params = {
        'train_batch':4,
        'eval_batch':4,
        'lr':0.001,
        'model_name':'faster_rcnn_backbone_re_train'
    }

    model_name = params['model_name']

    seed_everything()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, visible_char_map, num_classes, classes = prepare_data()

    from sklearn.utils import shuffle
    df_shuf = shuffle(df, random_state = 1)

    red_df = df_shuf[:40000]
    val_df = df_shuf[40000:50000]

    train_loader, val_loader = build_dataloaders(red_df, visible_char_map, df2 = val_df, batch_size = params['train_batch'])


    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, pretrained_backbone = True)
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 55)
    st_dc = torch.load('models/fastrcnn.pt')
    faster_rcnn.load_state_dict(st_dc['model_state_dict'])
    model = FasterRCNNBackboneModel(faster_rcnn, num_classes)
    model.to(DEVICE)
    
    for n, param in model.named_parameters():
        if 'classifier' not in n:
            param.requires_grad = False

    train_params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    optimizer = torch.optim.Adam(train_params, lr = params['lr'], weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader) * 2)

    train_losses, val_losses, train_avg_prec, val_avg_prec = train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs = 2, model_name = params['model_name'])
    
    
    st_dc = torch.load('models/faster_rcnn_backbone_re_train.pt')
    model.load_state_dict(st_dc['model_state_dict'])
    for n, param in model.named_parameters():
        param.requires_grad = True
    
    train_params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader) * 5)


    train_losses, val_losses, train_avg_prec, val_avg_prec = train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs = 5, model_name = 'faster_rcnn_backbone_re_train_fine_tuned')