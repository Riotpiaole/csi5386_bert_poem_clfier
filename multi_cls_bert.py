import torch
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from pdb import set_trace

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=17):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout( config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear( config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
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