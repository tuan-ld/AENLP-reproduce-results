import codecs
import numpy as np
import csv
from transformers import BertTokenizer, BertModel
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook

from transformers import BertTokenizerFast, BertForMaskedLM, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from torch import nn
import torch
from transformers import RobertaForMaskedLM
from transformers import BertForMaskedLM
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import torch
import torch.utils.data as data_utils
import torch
import torch.nn as nn
from torchcrf import CRF
import re

from tokenization_kobert import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')


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

        valid_output = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], last_hidden_state.size()[2], dtype=torch.float32, device='cuda:1')
        valid_masks = torch.zeros(last_hidden_state.size()[0], last_hidden_state.size()[1], dtype=torch.bool, device='cuda:1')
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


