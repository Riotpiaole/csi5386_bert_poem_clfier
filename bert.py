import time
import torch
import numpy as np
import pandas as pd
from os import listdir
from tqdm import tqdm 
from pdb import set_trace

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from preprocessing.preprocessing import *


train_dir = "./claff-happydb/data/TRAIN/"
test_dir = "./claff-happydb/data/TEST/"

max_token_size = 512

def read_data(csv_fn):
    df = pd.read_csv(csv_fn)
    df['agency'].dropna(inplace=True)
    df['social'].dropna(inplace=True)
    df['moment'].dropna(inplace=True)
    return df

def combine_labeled_df():
    train_label_path = train_dir + "labeled_10k.csv"
    test_label_path = test_dir + "labeled_17k.csv"

    df_train = read_data(train_label_path)
    df_test = read_data(test_label_path)

    return df_train , df_test 

from transformers import BertTokenizer
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def maximum_sentences_length(train_df , test_df, labels ='social'):
    assert labels in ['social', 'agency']
    concated_df = pd.concat([train_df , test_df])
    return max_sentence_length(concated_df)

def max_sentence_length(df,label):
    max_len = 0
    for sen in df[label]:
        input_ids =  tokenizer.encode(
            sen, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    
    return max_len

df = pd.read_csv("./poem_clf_dataset/dataAll_with_label.csv")

# train_df , test_df = combine_labeled_df()
# print('Tokenized: ', tokenizer.tokenize(train_df['moment'][0]))
# print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_df['moment'][0])))

# max_token_size = max_sentence_length(df, "Poem")
print('Tokenized: ', tokenizer.tokenize(df['Poem'][0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df['Poem'][0])))



def torch_dataset(df ,label="agency"):
    # dropped_label = "agency" if label != "agency" else "social"
    labels = LabelEncoder().fit_transform(df[label])
    input_ids = [] 
    attention_masks = []
    for sent in tqdm(df['moment']):
        encoded_dict = tokenizer.encode_plus( 
            sent,
            max_length = 128,           # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.   
            )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels).long()
    return input_ids , attention_masks , labels

def obtain_dataset_by_label(label, train_df= None , test_df=None, batch_size=32, hmid=False):
    hmids = list(test_df['hmid'].values)
    from torch.utils.data import TensorDataset, random_split

    # Combine the training inputs into a TensorDataset.
    test_ids , test_attention_masks , test_labels = torch_dataset(test_df, label)
    test_dataset = TensorDataset(test_ids, test_attention_masks, test_labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    if train_df is not None:
        # Divide the dataset by randomly selecting samples.
        input_ids , attention_masks , labels = torch_dataset(train_df, label)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        
        train_size = int(0.98 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))
        print('{:>5,} testing samples'.format(len(test_ids)))    
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
    else:
        train_dataloader, validation_dataloader = None, None

    test_dataloader = DataLoader(           
                test_dataset,
                batch_size = batch_size,    
    )   
    return train_dataloader , validation_dataloader , test_dataloader, hmids

if __name__ == "__main__":
    # from sklearn.preprocessing import LabelEncoder
    # train_dataloader , validation_dataloader , test_dataloader, hmids = obtain_dataset_by_label('social',batch_size=1, train_df=train_df)
    data = pd.read_csv("./poem_clf_dataset/dataAll_with_label.csv")
