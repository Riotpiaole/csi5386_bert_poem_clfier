import pandas as pd 
from collections import defaultdict
from sklearn.metrics import precision_score , recall_score , f1_score
from sklearn.preprocessing import LabelEncoder
from pdb import set_trace

def pack_res(prediction, labels, label='agency_label', res= defaultdict(lambda: []), tag=""):
    precision = 
    
    res[f'{label.replace("label", "")}=yes_precision'].append(precision[0])
    res[f'{label.replace("label", "")}=no_precision'].append(precision[1])

    res[f'{label.replace("label", "")}=yes_recall'].append(recall[0])
    res[f'{label.replace("label", "")}=no_recall'].append(recall[1])

    
    res[f'{label.replace("label", "")}=yes_f1_score'].append(f_score[0])
    res[f'{label.replace("label", "")}=no_f1_score'].append(f_score[1])

    set_trace()
    res_df = pd.DataFrame(res)
    
    res_df.to_csv(f"./{tag}.csv")
    res = defaultdict(lambda: [])

if __name__ == "__main__":
    all_pd_final = defaultdict(lambda : [])

    agency_stacking = pd.read_csv('./stacking_result/agency_stacking_e15_e30_origin_result.csv')
    social_stacking = pd.read_csv('./stacking_result/social_stacking_e15_e30_origin_result.csv')

    result = pd.read_csv("./result.csv")

    test_df = pd.read_csv("./claff-happydb/data/TEST/labeled_17k.csv")

    test_df['agency'] = LabelEncoder().fit_transform(test_df['agency'])
    test_df['social'] = LabelEncoder().fit_transform(test_df['social'])

    pack_res(agency_stacking, test_df['agency'], 'agency_label', all_pd_final, tag="stacking_agency")
    pack_res(social_stacking, test_df['social'], 'social_label', all_pd_final, tag="stacking_social")

    pack_res(result, test_df['agency'], 'agency_label', all_pd_final, tag="all_res_agency")
    pack_res(result, test_df['social'], 'social_label', all_pd_final, tag="all_res_social")
    
    set_trace()
    


    