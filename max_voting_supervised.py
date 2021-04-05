import torch
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pdb import set_trace
from train import model as OriginalBertModel
from collections import defaultdict
from copy import deepcopy
from generate_result import (
    device, 
    load_weights,
    hmids, 
    social_test_dataloader,
    prediction
)

def predict_stacking_batch(models, batch, label, result_csv):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    voting_candidate = []

    with torch.no_grad():
        for model in models:
            res = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels)
            loss, logits = res['loss'], res['logits'].cpu().numpy()
            prediction = np.argmax(logits, axis=1).flatten()
            voting_candidate.append(prediction)
        voting_candidate = list(
            np.apply_along_axis(
                lambda x,: np.bincount(x).argmax(), 0, 
                voting_candidate))

        result_csv[label] += voting_candidate
    
def stacking(label):
    result_csv = defaultdict(lambda : [])
    result_csv['hmid'] = hmids
    models=[
        load_weights(f"./trained_weights/15/{label}_epoch_15_bert_model.pth").cuda(),
        load_weights(f"./trained_weights/30/{label}_epoch_30_bert_model.pth").cuda(),
        deepcopy(OriginalBertModel).cuda()
    ]
    
    for batch in tqdm(social_test_dataloader):
        predict_stacking_batch(models ,batch, f'{label}_label', result_csv)
    
    result_csv = pd.DataFrame(result_csv)
    result_csv.to_csv(f"./{label}_stacking_e15_e30_origin_result.csv",index=False)

    del models



if __name__ == "__main__":
    # stacking("agency")
    stacking("social")
    
    # epoch_social_15_model = load_weights("./trained_weights/15/social_epoch_15_bert_model.pth")
    # epoch_agency_15_model = load_weights("./trained_weights/15/agency_epoch_15_bert_model.pth")
    # prediction(epoch_agency_15_model, epoch_social_15_model)

    # epoch_social_15_model.cuda.empty()
    # epoch_agency_15_model.cuda.empty()
    
    # del epoch_social_15_model
    # del epoch_agency_15_model

    # epoch_social_30_model = load_weights("./trained_weights/30/social_epoch_30_bert_model.pth")
    # epoch_agency_30_model = load_weights("./trained_weights/30/agency_epoch_30_bert_model.pth")
    # prediction(epoch_agency_30_model, epoch_social_30_model)

    test_df = pd.read_csv("./claff-happydb/data/TEST/labeled_17k.csv")

    agency_stacking = pd.read_csv('./stacking_result/agency_stacking_e15_e30_origin_result.csv')
    agency_stacking = pd.read_csv('./stacking_result/social_stacking_e15_e30_origin_result.csv')
    