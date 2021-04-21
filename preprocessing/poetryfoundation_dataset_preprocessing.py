import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm 
from pdb import set_trace
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def formattingData(root_path, cate_list):
    '''
    Reading data from files and preprocessing data including removing extra blank space and extra categories.
    '''
    count = 0
    title_list = []
    author_list = []
    labels_list = []
    poem_list = []
    file_list = []
    for cate in cate_list:
        cate_path = root_path + cate
        print("Processing on %s" % cate_path)
        for maindir, subdir, file_name_list in os.walk(cate_path):
            for filename in file_name_list:
                if filename.split('.')[-1] == 'txt':
                    if filename not in file_list:
                        file_list.append(filename)
                        fpath = os.path.join(maindir, filename)
                        try:
                            with open(fpath, 'r') as f:
                                fcontent = f.read().split('\n')
                                title = fcontent[0]
                                author = fcontent[1]
                                labels = fcontent[5].split(',')
                                poem = fcontent[9:]
                                poem = " ".join("".join(poem).split())
                                poem_list.append(poem)
                                labels_list.append(labels)
                                title_list.append(title)
                                author_list.append(author)
                        except:
                            print("Error on %s" % fpath)
                            pass
    
    for labels in labels_list:
        # remove extra blank space
        for i in range(len(labels)):
            labels[i] = labels[i].strip()
        l_copy = labels.copy()
        # remove categories which doesn't belong to the major 9 categories
        for label in l_copy:
            if label not in cate_list:
                labels.remove(label)
    
    for i in range(len(labels_list)):
        labels_list[i] = ",".join(labels_list[i])

    # Transfer to DataFrame 
    data_dict = {
        "Poem":poem_list,
        "Tags":labels_list
    }

    data = pd.DataFrame(data_dict)
    return data

def tags_to_binary(tags):
    tag_list = tags.split(',')
    # The number of major category are 9 
    binary_category = [0] * 9
    flag = False
    for t in tag_list:
        if t in cate_list:
            binary_category[cate_list.index(t)] = 1
            flag = True
    if not flag:
        binary_category = binary_category
    return binary_category


def torch_dataset(df, labels , label='Label'):
    from torch.utils.data import TensorDataset
    input_ids = []
    attention_masks = []
    
    for poem in tqdm(df.Poem):
        encoded_dict = tokenizer.encode_plus(
            poem,
            max_length = 128,           # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.  
        )
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.to_numpy())

    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    with open(f'{os.path.abspath('')}/dataset/pfundation_dataset.pkl', 'wb') as f:
        pickle.dump(dataset,  f)
    return dataset

if __name__ = "main":
    """
    Totally includ 9 categories representing 9 folders.
    Change the data path before run it.
    
    root_path: head path of data
    """
    root_path = os.path.abspath('')+"/categoriespoems/"
    cate_list = ['Activities', 'Arts & Sciences', 'Living', 'Love', 'Mythology & Folklore', 'Nature', 'Relationships', 'Religion', 'Social Commentaries']
    
    data = formattingData(root_path, cate_list)

