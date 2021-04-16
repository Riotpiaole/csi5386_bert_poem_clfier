import torch
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import( 
    BertForPreTraining, 
    BertPreTrainedModel, 
    BertModel, 
    BertConfig, 
    BertForMaskedLM, 
    BertForSequenceClassification,
)
from transformers import MobileBertModel, MobileBertConfig
from pdb import set_trace

class MobileBertForMultiLabelSequenceClassification(Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, num_labels=17):
        self.num_labels = num_labels
        super(MobileBertForMultiLabelSequenceClassification, self).__init__()
        self.bert = MobileBertModel.from_pretrained('google/mobilebert-uncased',
            hidden_act= "gelu",
            num_labels=num_labels)
        
        self.dropout = torch.nn.Dropout( 0.1 )
        self.classifier = torch.nn.Linear( 512 , num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        res = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = res['pooler_output']
        pooled_output = self.dropout(pooled_output)
        set_trace()
        logits = self.classifier(pooled_output)
        zeros = torch.zeros_like(logits)
        ones = torch.ones_like(logits)

        labels = labels.to(torch.float)
        loss_fct = BCEWithLogitsLoss()
        
        loss = loss_fct(logits, labels)
        return loss , logits

        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True