import pandas as pd 
from os import listdir

from numpy import argmax
# define input string

def tag_preporcessing(data):
    tags_space = set([])
    for tags in data:
        for tag in tags:
            tags_space.add(tag)
    return list(tags_space)
    

def read_dataset(fp):
    df = pd.read_csv(fp)
    return df 


def preprocess(Poem):
    Poem = Poem.str.replace("(<br/>)", "")
    Poem = Poem.str.replace('(<a).*(>).*(</ a>)', '')
    Poem = Poem.str.replace('(&amp)', '')
    Poem = Poem.str.replace('(&gt)', '')
    Poem = Poem.str.replace('(&lt)', '')
    Poem = Poem.str.replace('(\xa0)', ' ')  
    Poem = Poem.str.replace("(\n)", "")
    return Poem
from pdb import set_trace
def bundle_entire_dataset(folder):
    entire_dataset = []
    for fp in listdir(folder):
        entire_dataset.append(
            read_dataset(f"{folder}/{fp}"))
    
    full_data = pd.concat( entire_dataset, ignore_index=True)
    full_data = full_data.drop("Unnamed: 0", 1)
    full_data = full_data.drop("Poet", 1)
    full_data.drop('Title', axis=1, inplace=True)
    full_data.Poem = preprocess(full_data.Poem)
    full_data.dropna(inplace=True)
    all_unique_tags = tag_preporcessing(full_data.Tags.str.split(','))
    
    return full_data

if __name__ == "__main__":
    full_df = bundle_entire_dataset("./poem_clf_dataset")