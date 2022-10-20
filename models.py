import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torchcrf import CRF

import streamlit as st



@st.cache(allow_output_mutation=True)
class Bert_entity(nn.Module): #0.8785
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.hidden_size = 768
        self.block_size = 64
        self.hidden_layer = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.4)
        self.num_labels = num_labels
        self.crf = CRF(num_tags = num_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                valid_ids =None,
                label_mask=None
                ):

        output = self.model(input_ids =input_ids, attention_mask = attention_mask)

        last_hidden_state = output.last_hidden_state

        valid_output = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], last_hidden_state.size()[2], dtype=torch.float32, device='cpu')
        valid_masks = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], dtype=torch.bool, device='cpu')
        for i in range(last_hidden_state.size()[0]):
            jj = -1
            for j in range(last_hidden_state.size()[1]):
                if valid_ids[i][j].item() == 1:
                    jj+=1
                    valid_masks[i][jj] = 1
                    valid_output[i][jj] = last_hidden_state[i][j]

        relations = self.dropout(valid_output)
    
        logits = self.classifier(relations)
        
        loss = None 
        
        if labels is not None:
            
            if attention_mask is not None:
                loss, logits = self.crf(logits, labels, mask = valid_masks, reduction = 'token_mean'), self.crf.decode(logits, mask=valid_masks)
                
            else: 
                loss, logits = self.crf(logits, labels, mask = valid_masks, reduction = 'token_mean'), self.crf.decode(logits, mask=valid_masks)
        else:
                logits = self.crf.decode(logits, mask=valid_masks)
        return None, logits, last_hidden_state, valid_output, output.attentions


@st.cache(allow_output_mutation=True)
class Bert_relation(nn.Module): #0.8785
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.hidden_size = 768
        self.block_size = 64
        
        self.head_extractor = nn.Linear(768, 768)
        self.tail_extractor = nn.Linear(768, 768)
        
        
        self.bilinear_classifier = nn.Linear(768 * 3, num_labels)
        
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels

        
    def get_hrts(self, valid_output, left_ids, right_ids):
        
        left_masks = left_ids.unsqueeze(-1).expand(valid_output.size()).float()
        right_masks = right_ids.unsqueeze(-1).expand(valid_output.size()).float()
        
        
        left_rep = torch.logsumexp(valid_output * left_masks, dim=1)
        right_rep = torch.logsumexp(valid_output * right_masks, dim=1)
        
        return left_rep, right_rep
        
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                valid_ids =None,
                label_mask=None, 
                left_ids= None,
                right_ids=None
                ):

        output = self.model(input_ids =input_ids, attention_mask = attention_mask)

        last_hidden_state = output.last_hidden_state
        pooled_output = output.pooler_output
        context = self.dropout(pooled_output)
        valid_output = last_hidden_state
    
        
        left_rep, right_rep = self.get_hrts(valid_output, left_ids, right_ids)
        
        left_rep = self.head_extractor(left_rep)
        right_rep = self.head_extractor(right_rep)
        
        bl = torch.cat([context, left_rep, right_rep], dim=-1)
        
    
        logits = self.bilinear_classifier(bl)
        
        loss = None 
        
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            
        output = logits
        
        return loss, output if loss is not None else output
    
    
@st.cache(allow_output_mutation=True)
class Bert_entity_occurred(nn.Module): 
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.hidden_size = 768
        self.block_size = 64
        self.hidden_layer = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.4)
        self.num_labels = num_labels
        self.crf = CRF(num_tags = num_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                valid_ids =None,
                label_mask=None
                ):

        output = self.model(input_ids =input_ids, attention_mask = attention_mask)

        last_hidden_state = output.last_hidden_state

        valid_output = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], last_hidden_state.size()[2], dtype=torch.float32, device='cpu')
        valid_masks = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], dtype=torch.bool, device='cpu')
        for i in range(last_hidden_state.size()[0]):
            jj = -1
            for j in range(last_hidden_state.size()[1]):
                if valid_ids[i][j].item() == 1:
                    jj+=1
                    valid_masks[i][jj] = 1
                    valid_output[i][jj] = last_hidden_state[i][j]

        relations = self.dropout(valid_output)
    
        logits = self.classifier(relations)
        
        loss = None 
        
        if labels is not None:
            
            if attention_mask is not None:
                loss, logits = self.crf(logits, labels, mask = valid_masks, reduction = 'token_mean'), self.crf.decode(logits, mask=valid_masks)
                
            else: 
                loss, logits = self.crf(logits, labels, mask = valid_masks, reduction = 'token_mean'), self.crf.decode(logits, mask=valid_masks)
        else:
            logits = self.crf.decode(logits, mask=valid_masks)
            
        return None, logits, last_hidden_state, valid_output, output.attentions