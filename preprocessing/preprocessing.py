import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


''' Load poem data'''
def load_data():
    data_head_path = ""
    file_name = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '11.csv']
    file_list = []
    for i in file_name:
        f = pd.read_csv(data_head_path+i)
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
        binary_category = None
    return binary_category


if __name__ == "__main__":
    dataAll = load_data()
    dataAll.Poem = preprocess_poem(dataAll.Poem)
    filtered_category = preprocess_tag(dataAll)
    # Get top tags
    top_category = list(filtered_category.keys())
    print(top_category)
    
    dataAll['Label'] = dataAll['Tags'].apply(tags_to_binary)
    dataAll.to_csv('dataAll_with_label.csv')
    print(dataAll)



