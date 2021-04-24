import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import torch
from os.path import dirname , realpath
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
import pandas as pd
from tqdm import tqdm 
from pdb import set_trace
from transformers import BertTokenizer
from os.path import abspath
from poetryfoundation_dataset_preprocessing import formattingData





def get_current_project_directory():
    return dirname(realpath(__file__))

''' Load poem data'''
def load_data():
    data_folder_path = f"{get_current_project_directory()}/../poem_clf_dataset/"
    file_name = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
    file_list = []
    for i in file_name:
        f = pd.read_csv(data_folder_path + i)
        file_list.append(f)
        dataAll = pd.concat(file_list, ignore_index=True)
    dataAll.dropna(inplace=True)
    print("Loaded Data: ", file_name)
    print("Total poem amount: ", len(dataAll))
    return dataAll


''' Preprocess poem text'''
def preprocess_poem(Poem):
    Poem = Poem.str.replace('(<br/>)', '')
    Poem = Poem.str.replace('(<a).*(>).*(</a>)', '')
    Poem = Poem.str.replace('(&amp)', '')
    Poem = Poem.str.replace('(&gt)', '')
    Poem = Poem.str.replace('(&lt)', '')
    Poem = Poem.str.replace('(\xa0)', ' ')  
    Poem = Poem.str.replace('(\n)', '')
    Poem = Poem.str.replace(r'#', '')
    return Poem


''' Split every poem's tags and save tags with high frequency'''
def preprocess_tag(dataAll, freq = 100):
    dataAll_splited = dataAll.assign(Tags=dataAll.Tags.str.split(',')).explode('Tags')
    dataAll_splited.dropna(inplace=True)
    category = {}

    # Count unique tags
    for i in dataAll_splited.Tags:
        if i not in category:
            category[i] = 1
        else:
            category[i] += 1
    
    # Save tags with frequency higher than freq(default 100)
    filtered_category = {}
    for i, j in category.items():
        if j > freq:
            filtered_category[i] = j
        
    return filtered_category


''' According to the top tags, 
    change tags to a binary list 
    and drop poems without any top tag '''
def tags_to_binary(tags):
    tag_list = tags.split(',')
    binary_category = [0] * len(top_category)
    flag = False
    for t in tag_list:
        if t in top_category:
            binary_category[top_category.index(t)] = 1
            flag = True
    if not flag:
        binary_category = binary_category
    return binary_category

'''
Convert The data to torch dataset
'''

def torch_dataset(df, labels , label='Label'):
    from torch.utils.data import TensorDataset
    input_ids = []
    attention_masks = []
    
    for poem in tqdm(df.Poem):
        encoded_dict = tokenizer.encode_plus(
            poem,
            max_length = 512,           # Pad & truncate all sentences.
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
    with open(f'{get_current_project_directory()}/../poem_clf_dataset/dataset.pkl', 'wb') as f:
        pickle.dump(dataset,  f)
    return dataset

if __name__ == "__main__":
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    dataAll = load_data()
    dataAll.Poem = preprocess_poem(dataAll.Poem)
    filtered_category = preprocess_tag(dataAll)
    # Get top tags
    top_category = list(filtered_category.keys())
    top_category = ['Activities', 'Arts & Sciences', 'Living', 'Love', 'Nature', 'Relationships', 'Religion', 'Social Commentaries']

    print(top_category)
    dataAll.drop(columns=["Unnamed: 0", "Title", 'Poet'],inplace=True)
    dataAll.dropna(inplace=True)

    root_path = abspath('')+"/categoriespoems/"
    data = formattingData(root_path, top_category)
    overall_data = pd.concat([dataAll, data ])
    overall_data.dropna(inplace=True)

    labels = overall_data['Tags'].apply(tags_to_binary)
    
    labels = pd.DataFrame(
        data=np.array(labels.to_numpy().tolist()),
        columns=top_category)


    dataset =torch_dataset(overall_data, labels)
    print("total amount of dataset %d", len(dataset))
    from os.path import abspath
    with open(f"{abspath('')}/poem_clf_dataset/dataset.pkl",'wb') as f:
        pickle.dump(dataset,  f)
