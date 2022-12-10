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
import matplotlib.pyplot as plt
import logging

from sklearn import metrics

from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn


from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import torchmetrics
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import CharErrorRate





def seed_everything(seed_value=4995):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def create_data_frame(raw_data, image_path):
    """
    Create a Pandas DataFrame and a list for all the latex expressions

    Parameters
    ----------
    raw_data : list
        A list that contains all the image information

    Returns
    ----------
    df: DataFrame
        A Pandas DataFrame for running the analysis
    all_latex_lst: list
        A list for all the tokens, used for creating the token distribution
    """
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

def prepare_data(batch_size = 32, caption_task = False):
    df, visible_char_map = load_data()

    if caption_task:
        l = []
        for i in df['latex'].tolist():
            for j in i:
                l.append(j)

        classes = sorted(list(set(l)))
        num_classes = len(set(l))

        visible_char_map = {}
        for idx, symbol in enumerate(classes):
            visible_char_map[symbol] = idx + 1 

        return df, visible_char_map, num_classes, classes
        
    else:
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

def build_dataloaders(df, visible_char_map, test_set = False, df2 = None,  batch_size = 32, bad_classes = None, caption = False, pad_index = 0):
    """
    train_transforms = A.Compose([
      # A.Flip(0.5),
       # A.Resize(896, 896), 
        A.ShiftScaleRotate(rotate_limit = 10),
        #A.Normalize(),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        #ToTensorV2(p=1),
        ],
        bbox_params={
                'format': 'albumentations',
                'label_fields': ['labels']
    })
    
    val_transforms = A.Compose([
        #A.Resize(896, 896), 
        #A.ShiftScaleRotate(rotate_limit = 10)

        #A.Normalize(),
    ], bbox_params={
        'format': 'albumentations', 
        'label_fields': ['labels']
    })
    """
    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize((896,896)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize((896,896)),
          #transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }
    
    if caption and test_set:
        test_dataset = HandwrittenCaptionDataset(df, visible_char_map, transform = data_transforms['train'], return_file_name = True, train_mode = False)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=1)
        return test_loader
    
    if df2 is None:
        train_df, val_df = split_dataframe(df)
    else:
        train_df, val_df = df, df2
    
    if caption:
        train_dataset = HandwrittenCaptionDataset(train_df, visible_char_map, transform = data_transforms['train'])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=0)

        val_dataset = HandwrittenCaptionDataset(val_df, visible_char_map, transform = data_transforms['val'], return_file_name = True)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
        
    else:
        train_dataset = HandwrittenDataset(train_df, visible_char_map, transform = data_transforms['train'], bad_classes = bad_classes)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=1)

        val_dataset = HandwrittenDataset(val_df, visible_char_map, transform = data_transforms['val'], bad_classes = bad_classes)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=1)
    

    return train_loader, val_loader





class HandwrittenCaptionDataset(Dataset):
    
    def __init__(self, df, tokenizer, transform = None, return_file_name = False, max_target_length=145, train_mode = False):
        
        self.data = list(df.itertuples(index=False))
        self.train_mode = train_mode
        
        self.processor = tokenizer
        self.return_file_name = return_file_name
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        sample = self.data[index]
        caption = sample.latex_string
                
        f_name = sample.filename
        image = PIL.Image.open(f_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        labels = self.processor.tokenizer(caption, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        if self.return_file_name:
            return encoding, caption, f_name
        else:
            return encoding
    


def train_caption_loop(model, train_loader, loss_fn, optimizer, scheduler, num_classes = 64, classes =  None):
    
    if not classes:
        classes = np.arange(num_classes)
    
    model.train()
    
    train_loss_list = []
    print("Train loop")
    
    print_epochs = round(len(train_loader) / 100)
    #confusion_matrix = np.zeros(shape=(num_classes, num_classes))
        
    for i, data in enumerate((train_loader)):
        optimizer.zero_grad()
        
        #images = processor(data[0],return_tensors="pt").pixel_values.to(DEVICE)
        #target = processor.tokenizer.batch_encode_plus(data[1], return_tensors = 'pt', padding = True).input_ids.to(DEVICE)
        batch = data
        for k,v in batch.items():
            batch[k] = v.to(DEVICE)

        #output = model(images, labels=target)
        output = model(**batch)
        
        loss = output.loss
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        #print(loss)
        
        if i % print_epochs == 0:
            print(f'Batch: {i} of {len(train_loader)}. Loss: {loss_value}. Mean so far: {np.mean(train_loss_list)}. Mean of 100: {np.mean(train_loss_list[-100:])}')
            
            
    #precision, recall, f1_score, macro_f1, macro_precision, macro_recall = self.calculate_metrics(current_epoch, mode)
    #print(f"macro_f1: {macro_f1}, macro_precision: {macro_precision}, macro_recall: {macro_recall}")

    return np.mean(train_loss_list), train_loss_list


def val_caption_loop(model, val_loader, loss_fn, num_classes = 64, classes = None, attention_model= False):
    
    if not classes:
        classes = np.arange(num_classes)
    
    model.eval()
    
    val_loss_list = []

    correct_pred = 0
    num_samples = len(val_loader.dataset)
    print_epochs = round(len(val_loader) / 100)
    #confusion_matrix = np.zeros(shape=(num_classes, num_classes))

    bleu_score = torchmetrics.SacreBLEUScore()
    #rogue_score = ROUGEScore()
    cer_score = CharErrorRate()
        
    print("Validation loop")
    
    with torch.no_grad():
        for i, data in enumerate((val_loader)):
            #tokenized_targets = processor.tokenizer.batch_encode_plus(b[1], return_tensors = 'pt', padding = True)
            #target = tokenized_targets.input_ids
            #attention_mask = tokenized_targets.attention_mask
        
            #images = processor(data[0],return_tensors="pt").pixel_values.to(DEVICE)
                        
            #generated_ids = model.generate(images)
            #generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            model_stuff  = data[0]
            outputs = model.generate(model_stuff["pixel_values"].to(DEVICE))
            
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
            target_strings = data[1]
            img_files = data[2]
            """
            for gen_str_idx, gen_str in enumerate(generated_text):
                print("Pred:",gen_str)
                print("GT:  ",target_strings[gen_str_idx])
                img = PIL.Image.open(img_files[gen_str_idx]).convert("RGB")
                fig = plt.figure(figsize = (14,7))
                plt.imshow(img)
                plt.show()
            """
            bleu_score.update(generated_text,  [[tar for tar in target_strings]])
            #rogue_score.update(generated_text,  [[tar for tar in target_strings]])
            cer_score.update(generated_text,  list(target_strings))
            
            if i % print_epochs == 0:
                print(f'Batch: {i} of {len(val_loader)}, BLEU: {bleu_score.compute()}, {cer_score.compute()}')
                      
    #loss_mean = np.mean(val_loss_list)
    eval_cer = cer_score.compute()
    print("Eval CER:",eval_cer.item())

    #accuracy = correct_pred / num_samples
        
    #measure = np.mean(accuracy)
    
    #precision, recall, f1_score, macro_f1, macro_precision, macro_recall = self.calculate_metrics(current_epoch, mode)
    #print(f"macro_f1: {macro_f1}, macro_precision: {macro_precision}, macro_recall: {macro_recall}")
        
    return eval_cer.item()


def train_caption_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_classes = 64, classes = None, epochs = 5,  model_name = 'full_transformer', save_path = 'models', attention_model  = False):
    
    train_losses = []
    train_losses_no_aggregate = []
    val_eval_cer = []
    best_eval_cer = 0
        
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.to(DEVICE)
        tic = time.time()
        train_loss, train_loss_list_ret = train_caption_loop(model, train_loader, loss_fn, optimizer, scheduler, num_classes = num_classes)
        train_losses.append(train_loss)
        train_losses_no_aggregate.extend(train_loss_list_ret)
        with open('train_losses_no_aggregate_epoch_1.txt', 'w') as f:
            for line in train_losses_no_aggregate:
                f.write(f"{line}\n")
        

        print("Train loss:", train_loss)
        print(f"Train loop took {time.time()-tic}")
        tic = time.time()
        eval_cer = val_caption_loop(model, val_loader, loss_fn, num_classes = num_classes, attention_model = attention_model)
        print("Validation CER:", eval_cer)
        print(f"Validation loop took {time.time()-tic}")
        val_eval_cer.append(eval_cer)

        with open('val_eval_cer.txt', 'w') as f:
            for line in val_eval_cer:
                f.write(f"{line}\n")
        
        break


        #scheduler.step()

        if not best_eval_cer:
            best_eval_cer = eval_cer

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
            if eval_cer < best_eval_cer:
                best_eval_cer = eval_cer

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
    
    return train_losses, train_losses_no_aggregate, val_eval_cer


if __name__ == "__main__":
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 

    transformer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    #st_dc = torch.load( 'models/full_transformer.pt')
    #transformer_model.load_state_dict(st_dc['model_state_dict'])


    transformer_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    transformer_model.config.pad_token_id = processor.tokenizer.pad_token_id
    transformer_model.config.vocab_size = transformer_model.config.decoder.vocab_size

    transformer_model.config.eos_token_id = processor.tokenizer.sep_token_id
    transformer_model.config.max_length = 145
    transformer_model.config.early_stopping = True
    transformer_model.config.no_repeat_ngram_size = 2
    transformer_model.config.length_penalty = 2.0
    transformer_model.config.num_beams = 4

    params = {
        'train_batch':4,
        'eval_batch':4,
        'lr':5e-5,
        'model_name':'full_transformer',
        'epochs':5,
    }
    model_name = params['model_name']


    seed_everything()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transformer_model.to(DEVICE)

    df, visible_char_map, num_classes, classes = prepare_data(caption_task = True)
    print("Built dataset")

    from sklearn.utils import shuffle
    df_shuf = shuffle(df, random_state = 1)
    red_df = df_shuf[:40000]
    val_df = df_shuf[40000:42500]
    #val_df = df_shuf[40000:50000]
    test_df = df_shuf[50000:50100]


    pad_index = processor.tokenizer.pad_token_id


    train_loader, val_loader = build_dataloaders(red_df, processor, df2 = val_df, batch_size = params['train_batch'], caption = True, pad_index = pad_index)
    test_loader = build_dataloaders(test_df, processor, test_set=True, batch_size = params['train_batch'], caption = True, pad_index = pad_index)
    print("Built dataloader")



    optimizer = torch.optim.AdamW(transformer_model.parameters(), lr = 5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader) * 5)
    #optimizer.load_state_dict(st_dc['optimizer_state_dict'])
    #scheduler.load_state_dict(st_dc['scheduler_state_dict'])
    print(optimizer)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2,4], gamma = 0.5)

    train_losses, train_losses_no_aggregate, val_eval_cer = train_caption_model(transformer_model, train_loader, val_loader, None, optimizer, scheduler, num_classes = num_classes, classes = classes, epochs = 5,  model_name = 'full_transformer', save_path = 'models')

