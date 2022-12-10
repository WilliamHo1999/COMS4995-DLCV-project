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

import albumentations as A
from albumentations.pytorch import ToTensorV2

import gensim
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing

import torchmetrics
from torchmetrics import CharErrorRate



def seed_everything(seed_value=4995):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class HandwrittenCaptionDataset(Dataset):
    
    def __init__(self, df, visible_char_mapping, transform = None, image_resize = (1000,500), train_mode = True, return_file_name = False):
        
        self.data = list(df.itertuples(index=False))
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.train_mode = train_mode
        
        self.visible_char_mapping = visible_char_mapping
        self.return_file_name = return_file_name
                
        """
        for tup_idx, tup in enumerate(self.data):
            visible_latex_chars = tup.visible_latex_chars
            caption = visible_latex_chars.copy()
            caption.append('eos')
            caption.insert(0, 'sos')
            
        """
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        sample = self.data[index]
        caption = sample.latex
        if self.train_mode:
            caption.append('<eos>')
            caption.insert(0, '<sos>')
            caption = [*map(self.visible_char_mapping.get, caption)]
            caption = torch.tensor(caption)
        
        f_name = sample.filename
        image = PIL.Image.open(f_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.return_file_name:
            return image, caption, f_name
        else:
            return image, caption
    
def pad_batch_rnn(batch, pad_index):
    """
    pad batches
    """
    longest_sentence = max([y.size(0) for X, y in batch])
    new_caption = torch.stack([F.pad(y, (0, longest_sentence - y.size(0)), value=pad_index) for X, y in batch])
    new_images = torch.stack([img for img, _ in batch])
    lengths = torch.LongTensor([y.size(0)-1 for X, y in batch])
    return new_images, (new_caption, lengths)

def pad_batch_rnn_validation(batch, pad_index):
    """
    pad batches
    """
    longest_sentence = max([len(y) for X, y in batch])
    new_caption = [' '.join(y) for X, y in batch]
    new_images = torch.stack([img for img, _ in batch])
    lengths = torch.LongTensor([len(y)-1 for X, y in batch])
    return new_images, (new_caption, lengths)

def pad_batch_rnn_validation_file_name(batch, pad_index):
    """
    pad batches
    """
    longest_sentence = max([len(y) for X, y, _ in batch])
    new_caption = [' '.join(y) for X, y,_ in batch]
    new_images = torch.stack([img for img, _,_ in batch])
    lengths = torch.LongTensor([len(y)-1 for X, y,_ in batch])
    file_names = [f_name for _,_, f_name in batch]
    return new_images, (new_caption, lengths), file_names

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
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=1, collate_fn=lambda x: pad_batch_rnn_validation_file_name(x, pad_index=pad_index))
        return test_loader
    
    if df2 is None:
        train_df, val_df = split_dataframe(df)
    else:
        train_df, val_df = df, df2
    
    if caption:
        train_dataset = HandwrittenCaptionDataset(train_df, visible_char_map, transform = data_transforms['train'])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=1, collate_fn=lambda x: pad_batch_rnn(x, pad_index=pad_index))

        val_dataset = HandwrittenCaptionDataset(val_df, visible_char_map, transform = data_transforms['val'], train_mode = False)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=1, collate_fn=lambda x: pad_batch_rnn_validation(x, pad_index=pad_index))
        
    else:
        train_dataset = HandwrittenDataset(train_df, visible_char_map, transform = data_transforms['train'], bad_classes = bad_classes)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=1)

        val_dataset = HandwrittenDataset(val_df, visible_char_map, transform = data_transforms['val'], bad_classes = bad_classes)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=1)
    

    return train_loader, val_loader

class CNNEncoder(nn.Module):
    
    def __init__(self, model, encoded_image_size = 28):
        super(CNNEncoder, self).__init__()
        
        modules = list(model.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.fine_tune(False)
        
    def forward(self, x):
        out = self.cnn(x) 
        out = self.adaptive_pool(out) 
        out = out.permute(0, 2, 3, 1) 
        return out
        
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.cnn.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.cnn.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

def build_resnet(num_classes, model_path = 'models/resnet.pt', to_cuda = True):
    if not model_path:
        model = torchvision.models.resnet18(pretrained = True)
        input_feat = model.fc.in_features
        
        model.fc = nn.Linear(input_feat, num_classes)
        loaded_state_dict = False

    else:
        print("Loaded", model_path)
        model = torchvision.models.resnet18()
        input_feat = model.fc.in_features
        model.fc = nn.Linear(input_feat, num_classes)
        loaded_model = torch.load(model_path)
        if len(loaded_model) > 1:
            loaded_model = loaded_model['model_state_dict']
        model.load_state_dict(loaded_model)
        loaded_state_dict = True
        
    if to_cuda:
        model = model.to(DEVICE)
        
    return model, loaded_state_dict

class ImageCaptioningModel(nn.Module):
    
    def __init__(self, image_feature_extractor, embedding_vectors, hidden_size = 150, num_layers = 1, dropout = 0, rnn_type = 'LSTM'):
        super(ImageCaptioningModel, self).__init__()
        
        self.cnn = image_feature_extractor
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors.vectors))
        
        self.input_size = self.embedding.embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        
        self.num_classes = len(embedding_vectors.key_to_index)

        self.classifier = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.classifier.weight = self.embedding.weight
        
        self.vocab_dict = embedding_vectors.key_to_index
        self.pad_id = self.vocab_dict['<pad>']
        
        cnn_output = self.cnn.fc.in_features
        
        input_layer = nn.Sequential(
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(cnn_output, self.input_size),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.cnn.fc = input_layer
        
        
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = 2 * self.input_size,
                                    hidden_size = self.hidden_size,
                                    num_layers = self.num_layers,
                                    bidirectional = False,
                                    dropout = self.dropout,
                                    batch_first=True
                              )
        else:
            self.rnn = nn.GRU(input_size = 2 * self.input_size,
                                    hidden_size = self.hidden_size,
                                    num_layers = self.num_layers,
                                    bidirectional = False,
                                    dropout = self.dropout,
                                    batch_first=True
                             )
        
        self.freeze_cnn()
    
    def forward(self, x):
        
        image, caption, length = x
        
        img_features = self.cnn(image)
        
        embed = self.embedding(caption)
        b_size = embed.shape[0]
        seq_len = embed.shape[1]
        cat_input = torch.cat([embed, img_features.unsqueeze(1).expand(b_size, seq_len, -1)],axis=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(cat_input, length, batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'LSTM':
            init_hid = torch.zeros(self.num_layers, img_features.shape[0], self.hidden_size).to(DEVICE)
            init_control = torch.zeros(self.num_layers, img_features.shape[0], self.hidden_size).to(DEVICE)
            initial_hidden_state = (init_hid, init_control)
        else:
            initial_hidden_state = torch.zeros(self.num_layers, img_features.shape[0], self.hidden_size).to(DEVICE)
            
        states, _ = self.rnn(packed_input, initial_hidden_state)
        output,_ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)

        logits = self.classifier(output).permute(0,2,1)
        
        return logits
    
    def get_image_features(self, image):
        img_features = self.cnn(image)
        
        return img_features
    
    def predict_next(self, image_feature, source_prefix):
        source = source_prefix
        source_length = (source != self.pad_id).sum(axis = 1).to('cpu')
        embed = self.embedding(source)
        
        b_size = embed.shape[0]
        seq_len = embed.shape[1]
        cat_input = torch.cat([embed, image_feature.unsqueeze(1).expand(b_size, seq_len, -1)],axis=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(cat_input, source_length, batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'LSTM':
            init_hid = torch.zeros(self.num_layers, image_feature.shape[0], self.hidden_size).to(DEVICE)
            init_control = torch.zeros(self.num_layers, image_feature.shape[0], self.hidden_size).to(DEVICE)
            initial_hidden_state = (init_hid, init_control)
        else:
            initial_hidden_state = torch.zeros(self.num_layers, image_feature.shape[0], self.hidden_size).to(DEVICE)
            
        states, _ = self.rnn(packed_input, initial_hidden_state)
        output,_ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)
        
        logits = self.classifier(output)[:,-1,:].squeeze()
        
        return logits

    def freeze_cnn(self):
        for n, param in self.cnn.named_parameters():
            if "fc" not in n:
                param.requires_grad = False
    
    def unfreeze_cnn(self):
        for n, param in self.cnn.named_parameters():
            param.requires_grad = True

def train_caption_loop(model, train_loader, loss_fn, optimizer, scheduler, num_classes = 64, classes =  None):
    
    if not classes:
        classes = np.arange(num_classes)
    
    model.train()
    
    train_loss_list = []
    print("Train loop")
    
    correct_pred = 0
    num_samples = len(train_loader.dataset)
    print_epochs = round(len(train_loader) / 100)
        
    for i, data in enumerate((train_loader)):
        optimizer.zero_grad()
        
        inp = []
        for dat in data:
            if isinstance(dat, tuple):
                inp.append(dat[0][:,0:-1].to(DEVICE))
                inp.append(dat[1])
            else:
                inp.append(dat.to(DEVICE))
        
        target = dat[0][:,1:].to(DEVICE)
        
        output = model(inp)
        loss = loss_fn(output, target)
        
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        
        pred = F.softmax(output).argmax(1)
        correct = (pred == target).sum().item()
        correct_pred += correct
        
        loss.backward()
        optimizer.step()
        
        if i % print_epochs == 0:
            print(f'Batch: {i} of {len(train_loader)}. Loss: {loss_value}. Mean so far: {np.mean(train_loss_list)}. Mean of 100: {np.mean(train_loss_list[-100:])}')
            
    accuracy = correct_pred / num_samples
        
    measure = np.mean(accuracy)

    return np.mean(train_loss_list), measure, train_loss_list


class GreedySearchRNN:
    def __init__(self, model, tokenizer, attention_model = False, max_length=120):
        self.model = model
        self.tokenizer = tokenizer
        self.reverse_tokenizer =  {v: k for k, v in tokenizer.items()}
        self.max_length = max_length
        self.attention_model = attention_model

        self.sos_id = tokenizer['<sos>']
        self.eos_id = tokenizer['<eos>']
        self.pad_id = tokenizer['<pad>']

    @torch.no_grad()
    def __call__(self, image):
        if self.attention_model:
            image_features, hid, cont = self.model.get_image_features_and_init_states(image)
        else:
            image_features = self.model.get_image_features(image)
        
        source = torch.full([image_features.size(0), 1], fill_value=self.sos_id, device=image_features.device)
        stop = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for timestep in range(self.max_length):
            if self.attention_model:
                prediction, hid, cont = self.model.predict_next(image_features, source, hid, cont, timestep)
            else:
                prediction = self.model.predict_next(image_features, source)
            prediction = torch.where(stop, self.pad_id, prediction.argmax(-1))
            stop |= prediction == self.eos_id

            source = torch.cat([source, prediction.unsqueeze(1)], dim=1)

            if stop.all():
                break
            
        sentences = []
        for sent in source.tolist():
            translated_sentence = []
            for tok in sent:
                if tok not in [self.sos_id, self.eos_id, self.pad_id]:
                    translated_sentence.append(self.reverse_tokenizer[tok])
            sentences.append(' '.join(translated_sentence))
            
        return sentences


def val_caption_loop(model, val_loader, loss_fn, num_classes = 64, classes = None, attention_model= False):
    
    if not classes:
        classes = np.arange(num_classes)
    
    model.eval()
    
    val_loss_list = []

    correct_pred = 0
    num_samples = len(val_loader.dataset)
    print_epochs = round(len(val_loader) / 100)

    bleu_score = torchmetrics.SacreBLEUScore()
    cer_score = CharErrorRate()

    
    greedy_search = GreedySearchRNN(model, val_loader.dataset.visible_char_mapping, attention_model=attention_model)
    
    print("Validation loop")
    
    with torch.no_grad():
        for i, data in enumerate((val_loader)):

            images = data[0].to(DEVICE)
            
            target_strings = data[1][0]
            
            predictions_str = greedy_search(images)
            bleu_score.update(predictions_str, target_strings)
            cer_score.update(predictions_str,  list(target_strings))


            if i % print_epochs == 0:
                print(f'Batch: {i} of {len(val_loader)}, BLEU: {bleu_score.compute()}, {cer_score.compute()}')
            
    eval_bleu = bleu_score.compute()
    eval_char_error = cer_score.compute()
    print("Eval BLEU:",eval_bleu, eval_char_error)

    return eval_bleu, eval_char_error

def train_caption_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_classes = 64, classes = None, epochs = 5,  model_name = 'caption_model', save_path = 'models', attention_model  = False):
    
    train_losses = []
    train_accuracies = []
    val_bleu_scores = []
    val_char_error = []
    best_bleu_score = 0
    best_char_error_rate = 0
    
    all_train_losses = []
        
    for epoch in range(epochs):
        
        print(f"Epoch: {epoch}")
        tic = time.time()
        train_loss, train_acc, train_loss_return = train_caption_loop(model, train_loader, loss_fn, optimizer, scheduler, num_classes = num_classes)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print("Train loss:", train_loss, train_acc)
        print(f"Train loop took {time.time()-tic}")
        
        tic = time.time()
        val_bleu, eval_char_error = val_caption_loop(model, val_loader, loss_fn, num_classes = num_classes, attention_model = attention_model)
        print("Validation BLEU:", val_bleu)
        print(f"Validation loop took {time.time()-tic}")
        val_bleu_scores.append(val_bleu)
        val_char_error.append(eval_char_error)
        
        all_train_losses.extend(train_loss_return)
        
        with open(f'caption_cnn_losses_all.txt', 'w') as f:
            for line in all_train_losses:
                f.write(f"{line}\n")
        with open('caption_cnn_losses.txt', 'w') as f:
            for line in train_losses:
                f.write(f"{line}\n")
        with open(f'caption_cnn_val_bleu_scores.txt', 'w') as f:
            for line in val_bleu_scores:
                f.write(f"{line}\n")
        with open(f'caption_cnn_val_cer_scores.txt', 'w') as f:
            for line in val_char_error:
                f.write(f"{line}\n")
        

        scheduler.step()

        if not best_char_error_rate:
            best_char_error_rate = eval_char_error

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
            if eval_char_error < best_char_error_rate:
                best_char_error_rate = eval_char_error

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
    
    return train_losses, train_accuracies, val_bleu_scores


if __name__ == "__main__":

    params = {
        'train_batch':16,
        'eval_batch':16,
        'lr':0.001,
        'model_name':'caption_model_simple_rnn_re_train'
    }
    model_name = params['model_name']

    seed_everything()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, visible_char_map, num_classes, classes = prepare_data(caption_task = True)

    embedder = KeyedVectors.load("embedding_models/skipgram_True_window_4_iterations_20_vocabsize_60_.model")
    embedder.add_vector('<unk>', np.random.rand(embedder.vector_size))
    embedder.add_vector('<pad>', np.zeros(embedder.vector_size))
    embedder.add_vector('<sos>', np.random.rand(embedder.vector_size))
    embedder.add_vector('<eos>', np.random.rand(embedder.vector_size))
    pad_index = embedder.key_to_index['<pad>']


    from sklearn.utils import shuffle
    df_shuf = shuffle(df, random_state = 1)
    red_df = df_shuf[:40000]
    val_df = df_shuf[40000:50000]


    train_loader, val_loader = build_dataloaders(red_df, embedder.key_to_index, df2 = val_df, batch_size = params['train_batch'], caption = True, pad_index = pad_index)


    cnn_model, loaded_state_dict = build_resnet(54, 'models/resnet_re_train_fine_tuned.pt')
    caption_model = ImageCaptioningModel(cnn_model, embedder)

    caption_model.to(DEVICE)


    loss_fn = nn.CrossEntropyLoss(ignore_index = pad_index)

    optimizer = torch.optim.Adam(caption_model.parameters(), lr = 0.001, weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3,4], gamma = 0.1)

    train_caption_model(caption_model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_classes = num_classes, classes = classes, epochs = 5,  model_name = 'caption_model_simple_rnn_re_train', save_path = 'models')
