import torch
from torch.nn import Module
from transformers import BertForSequenceClassification, AdamW, BertConfig
from pdb import set_trace

class BertForMultiLabelSequenceClassification(Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, num_labels=17):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels , # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.dropout = torch.nn.Dropout( 0.8 )
        self.classifier = torch.nn.Linear( 768, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        res = self.bert(input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, 
            labels=labels)
        set_trace()
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True