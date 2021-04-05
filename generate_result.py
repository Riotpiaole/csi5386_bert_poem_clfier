import torch
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pdb import set_trace
from copy import deepcopy
from train import model , evaluation
from bert import obtain_dataset_by_label
from collections import defaultdict


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def load_weights(path):
    new_model = deepcopy(model)
    new_model.load_state_dict(torch.load(path))
    return new_model

def predict_model_batch(model, batch, label, result_csv, device=device):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
        res = model(b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask,
                labels=b_labels)
        loss, logits = res['loss'], res['logits'].cpu().numpy()
        prediction = list(np.argmax(logits, axis=1).flatten())
        result_csv[label] += prediction

_, _, social_test_dataloader , hmids = obtain_dataset_by_label('social',hmid=True)

def prediction(agency_model, social_model,  dataloader=social_test_dataloader, label="social", epochs=15):

    result_csv = defaultdict(lambda : [])
    agency_model.cuda()
    social_model.cuda()
    
    result_csv['hmid']= list(hmids)
    for batch in tqdm(dataloader):
        # predict_model_batch(agency_model,batch, 'agency_label', result_csv)
        predict_model_batch(social_model,batch, 'social_label', result_csv)

    result_csv = pd.DataFrame(result_csv)
    result_csv.to_csv(f"./{label}_{epochs}_result.csv",index=False)

if __name__ == "__main__":
    res = defaultdict(lambda :[])
    epoch_social_15_model = load_weights("./trained_weights/15/social_epoch_15_bert_model.pth")
    epoch_social_15_model.to(device)
    for batch in social_test_dataloader:
        predict_model_batch(epoch_social_15_model, batch, 'social_label', res)
    
    # epoch_social_30_model = load_weights("./trained_weights/30/social_epoch_30_bert_model.pth")
    # epoch_agency_30_model = load_weights("./trained_weights/30/agency_epoch_30_bert_model.pth")
    # prediction(epoch_agency_30_model, epoch_social_30_model,  epochs=30)
    # import os 
    # os.system("shutdown /s /t 1")

    