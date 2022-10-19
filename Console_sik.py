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
import json
from tokenization_kobert import KoBertTokenizer
import pickle

f = open("./model_configs/dict_yak.p", 'rb')
a_dic = pickle.load(f)
f.close()

f = open("./model_configs/dic_rela.p", 'rb')
b_dic = pickle.load(f)
f.close()

f = open("annotations-legend.json", 'rb')
q_dic = json.load(f)
f.close()
q_dic[0] = 'None'

b_inv_dic = {v:k for k, v in b_dic.items()}
a_inv_dic = {v: k for k, v in a_dic.items()}
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

class MEDTrainDataset(data_utils.Dataset):
    def __init__(self, max_len, sep_token, cls_token, pad_token, text_seq, label_seq, valid_seq, tokenizer):
        
        self.text_seq = text_seq
        self.label_seq = label_seq
        self.num_vocabs = tokenizer.vocab_size
        self.max_len = max_len
        self.cls_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cls_token))
        self.sep_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sep_token))
        self.tokenizer= tokenizer
        self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pad_token))
        self.valid_seq = valid_seq

    def __len__(self):

        return len(self.text_seq)

    def __getitem__(self, index):
        
        text= self.text_seq[index]
        labels = self.label_seq[index]
        valid_seq = self.valid_seq[index]
        text = self.tokenizer.convert_tokens_to_ids(text)

        text = self.cls_token + text[:self.max_len-2] + self.sep_token
        labels = [23] + labels[:self.max_len-2] + [23]
        valid_seq = [1] + valid_seq[:self.max_len-2] + [1]

        segment_ids_length = min(len(text), self.max_len)
        valid_length = min(len(text), self.max_len)   
        label_length = min(len(labels), self.max_len)

        mask_len = self.max_len - len(text)
        text =  text + self.pad_token * mask_len

        mask_len = self.max_len - len(labels)
        labels = labels + [23] * mask_len 

        mask_len = self.max_len - len(valid_seq)
        valid_seq = valid_seq + [0] * mask_len 

        attention_masks = [0.0]*len(text)

        for i in range(valid_length):
            attention_masks[i] = 1.0

        label_masks = [0.0] * len(labels)
        for i in range(label_length):
            label_masks[i] = 1.0
        
        segment_ids = [0.0]*len(text)
       
        return torch.LongTensor(text), torch.LongTensor(labels), torch.LongTensor(valid_seq),torch.LongTensor(attention_masks),torch.LongTensor(segment_ids), torch.LongTensor(label_masks)

class MEDinfDataset(data_utils.Dataset):
    def __init__(self, max_len, sep_token, cls_token, pad_token, text_seq, valid_seq, tokenizer):
        
        self.text_seq = text_seq
        self.num_vocabs = tokenizer.vocab_size
        self.max_len = max_len
        self.cls_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cls_token))
        self.sep_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sep_token))
        self.tokenizer= tokenizer
        self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pad_token))
        self.valid_seq = valid_seq

    def __len__(self):

        return len(self.text_seq)

    def __getitem__(self, index):
        
        text= self.text_seq[index]
        
        valid_seq = self.valid_seq[index]
        text = self.tokenizer.convert_tokens_to_ids(text)

        text = self.cls_token + text[:self.max_len-2] + self.sep_token
        
        valid_seq = [1] + valid_seq[:self.max_len-2] + [1]

        segment_ids_length = min(len(text), self.max_len)
        valid_length = min(len(text), self.max_len)   
        

        mask_len = self.max_len - len(text)
        text =  text + self.pad_token * mask_len

        mask_len = self.max_len - len(valid_seq)
        valid_seq = valid_seq + [0] * mask_len 

        attention_masks = [0.0]*len(text)

        for i in range(valid_length):
            attention_masks[i] = 1.0

 
        
        segment_ids = [0.0]*len(text)
       
        return torch.LongTensor(text),  torch.LongTensor(valid_seq),torch.LongTensor(attention_masks),torch.LongTensor(segment_ids)

class MEDinfDataset_occurred(data_utils.Dataset):
    def __init__(self, max_len, sep_token, cls_token, pad_token, text_seq, valid_seq, tokenizer):
        
        self.text_seq = text_seq
        self.num_vocabs = tokenizer.vocab_size
        self.max_len = max_len
        self.cls_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cls_token))
        self.sep_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sep_token))
        self.tokenizer= tokenizer
        self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pad_token))
        self.valid_seq = valid_seq

    def __len__(self):

        return len(self.text_seq)

    def __getitem__(self, index):
        
        text= self.text_seq[index]
        
        valid_seq = self.valid_seq[index]
        text = self.tokenizer.convert_tokens_to_ids(text)

        text = self.cls_token + text[:self.max_len-2] + self.sep_token
        
        valid_seq = [1] + valid_seq[:self.max_len-2] + [1]

        segment_ids_length = min(len(text), self.max_len)
        valid_length = min(len(text), self.max_len)   
        

        mask_len = self.max_len - len(text)
        text =  text + self.pad_token * mask_len

        mask_len = self.max_len - len(valid_seq)
        valid_seq = valid_seq + [0] * mask_len 

        attention_masks = [0.0]*len(text)

        for i in range(valid_length):
            attention_masks[i] = 1.0

 
        
        segment_ids = [0.0]*len(text)
       
        return torch.LongTensor(text),  torch.LongTensor(valid_seq),torch.LongTensor(attention_masks),torch.LongTensor(segment_ids)

class MEDInf_totDataset(data_utils.Dataset):
    def __init__(self, max_len, sep_token, cls_token, pad_token, text_seq, left_seq, right_seq, tokenizer):
        
        self.text_seq = text_seq
        self.num_vocabs = tokenizer.vocab_size
        self.max_len = max_len
        self.left_seq = left_seq
        self.right_seq = right_seq
        self.cls_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cls_token))
        self.sep_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sep_token))
        self.tokenizer= tokenizer
        self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pad_token))
      
    def __len__(self):

        return len(self.text_seq)

    def __getitem__(self, index):
        
        text= self.text_seq[index]
        left_seq = self.left_seq[index]
        right_seq = self.right_seq[index]
        
        text = self.tokenizer.convert_tokens_to_ids(text)

        text = self.cls_token + text[:self.max_len-2] + self.sep_token
        left_seq = [1] + left_seq[:self.max_len-2] + [1]
        right_seq = [1] + right_seq[:self.max_len-2] + [1]
        
        segment_ids_length = min(len(text), self.max_len)
        valid_length = min(len(text), self.max_len)   
      
        mask_len = self.max_len - len(text)
        text =  text + self.pad_token * mask_len

        mask_len = self.max_len - len(left_seq)
        left_seq = left_seq + [0] * mask_len 
        
        mask_len = self.max_len - len(right_seq)
        right_seq = right_seq + [0] * mask_len 
        
        attention_masks = [0.0]*len(text)

        for i in range(valid_length):
            attention_masks[i] = 1.0

        segment_ids = [0.0]*len(text)
       
        return torch.LongTensor(text),  torch.LongTensor(left_seq),  torch.LongTensor(right_seq), torch.LongTensor(attention_masks), torch.LongTensor(segment_ids)

def tokenizing_(token_list,  tokenizer):
    text_seq = []
    valid_seq = []
    ## target_seq ~ target_text_seq 똑같
   
    for tokens in token_list:
        new_target_text = []
        new_valid = []
        
        for i, t in enumerate(tokens):
            tokens_wordpiece = tokenizer.tokenize(t)
            new_v = [1] + [0]*(len(tokens_wordpiece)-1)
            new_target_text.extend(tokens_wordpiece)
            new_valid.extend(new_v)
            
        valid_seq.append(new_valid)
        text_seq.append(new_target_text)
       
    return text_seq, valid_seq

def tokenizing_for_rel(token_list, label_list,  tokenizer):
    text_seq = []
    valid_seq = []
    ## target_seq ~ target_text_seq 똑같
    label_seq = label_list
   
    for tokens, label in zip(token_list, label_list):
        new_target_text = []
        new_valid = []
        before_label = -1
        
        for i, (t, l) in enumerate(zip(tokens, label)):
            tokens_wordpiece = tokenizer.tokenize(t)
            
            if before_label != l and l != -1:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if l != -1 and i+1 != len(label) and l != label[i+1]:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            if l != -1 and i+1 == len(label):
                tokens_wordpiece = tokens_wordpiece + ["*"]
            before_label = l
            
            new_v = [l]*(len(tokens_wordpiece))
            new_target_text.extend(tokens_wordpiece)
            new_valid.extend(new_v)
            
        valid_seq.append(new_valid)
        text_seq.append(new_target_text)
       
    return text_seq, valid_seq, label_seq

def tokenizing_for_relv2(token_list, label_list, rel_list, tokenizer):
    text_seq = []
    valid_seq = []
    final_left_seq = []
    final_right_seq = []
    ## target_seq ~ target_text_seq 똑같
    label_seq = []
   
    for tokens, label, rel in zip(token_list, label_list, rel_list):
        
        lefts, rights = rel
        
        for le, ri in zip(lefts, rights):
            
            new_target_text = []
            new_valid = []
            new_left = []
            new_right = []
            before_label = -1
            
            for i, (t, l) in enumerate(zip(tokens, label)):
                tokens_wordpiece = tokenizer.tokenize(t)

                if before_label != l and l != -1:
                    if l == le:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    elif l == ri:
                        tokens_wordpiece = ["#"] + tokens_wordpiece
                        
                if l != -1 and i+1 != len(label) and l != label[i+1]:
                    if l == le:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                    elif l == ri:
                        tokens_wordpiece = tokens_wordpiece + ["#"]
                if l != -1 and i+1 == len(label):
                    if l == le:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                    elif l == ri:
                        tokens_wordpiece = tokens_wordpiece + ["#"]
                        
                before_label = l
                new_v = [l]*(len(tokens_wordpiece))
                new_target_text.extend(tokens_wordpiece)
                new_valid.extend(new_v)
                
            leftaa = [float(x == le) for x in new_valid]
            rightaa = [float(x == ri) for x in new_valid]
            final_left_seq.append(leftaa)
            final_right_seq.append(rightaa)
            text_seq.append(new_target_text)
           
   
    return text_seq, final_left_seq, final_right_seq


def sentence_to_tokens(sentence):
    
    text_seq = []
    q = sentence.split()
    text_seq.append(q)
    
    return text_seq

def sentence_to_input(sentence, tokenizer):
    
    tokens = sentence_to_tokens(sentence)
    
    text_seq, valid_seq = tokenizing_(tokens, tokenizer)
    
    return text_seq, valid_seq

def extracted_labels(predicted_results, tokens):
    result = predicted_results[0]
    result = result[1:-1]
    before = 0
    q = -1
    labels = []
    entities = []
    ordered_tokens = []
    for re in result:
        if re != 0:
            if re != before:
                q = q + 1
                labels.append(q)
            else:
                labels.append(q)
        else:
            labels.append(-1)
        
        before = re
    
    for i in range(q+1):
        tok = ""
        rea = 0
        for label, token, resul in zip(labels, tokens, result):
            if label == i:
                tok = tok + " " + token
                rea = resul
        ordered_tokens.append(tok)
        entities.append(rea)
    return q, labels, ordered_tokens, entities

def token_label_together(token_list, label_list):
    new_token_list = []

    for token, label in zip(token_list, label_list):
        new_token = []
       
        for i, (tok, lab) in enumerate(zip(token, label)):
            if lab != 0 and lab != -1:
                new_token.append(tok +"/"+str(lab))
            else:
                new_token.append(tok)
        new_token_list.append(new_token)
    return new_token_list

def make_pair_by_max(value, ordered_tokens, entities, q_pair):
    lefts = []
    rights = []
    pairs = []
    entity_pairs =[]
    for i in range(value+1):
        for j in range(value+1):
            if i != j and entitie_check(entities[j], entities[i], q_pair) == True:
                pairs.append((ordered_tokens[j], ordered_tokens[i]))
                lefts.append(j)
                rights.append(i)
                entity_pairs.append((entities[j], entities[i]))
                
    return lefts, rights, pairs, entity_pairs

def entitie_check(l, r, q_pair):
    booa = False
    if (l, r) in q_pair:
        booa = True
    return booa    

def make_left_rights_sents(lefts, rights, labels):
    left_ids = []
    right_ids = []
    for le, ri in zip(lefts, rights):
        left_id = []
        right_id = []
        
        for l in labels:
            if l == le:
                left_id.append(1)
                right_id.append(0)
            elif l == ri:
                right_id.append(1)
                left_id.append(0)
            else:
                left_id.append(0)
                right_id.append(0)
        left_ids.append(left_id)
        right_ids.append(right_id)
    return left_ids, right_ids

def inferred_sentence_to_input(sentence, predicted_results, tokenizer, q_pair):
    
    tokens = sentence_to_tokens(sentence)
    
    max_labs, labels, ordered_tokens, entities = extracted_labels(predicted_results, tokens[0])
    
    new_tokens = token_label_together(tokens, [labels])
    
    lefts, rights, pairs, entity_pairs = make_pair_by_max(max_labs, ordered_tokens, entities, q_pair)
 
    text_seq, left_ids, right_ids = tokenizing_for_relv2(new_tokens, [labels], [(lefts, rights)], tokenizer)
    
    
    
    return text_seq, left_ids, right_ids, pairs, entity_pairs
    
def inference(sentence, tokenizer, model):

    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    pad_token = tokenizer.pad_token
    maxlen = 200
    text_seq, valid_seq = sentence_to_input(sentence, tokenizer)
    model.to('cpu')
    inf_dataset = MEDinfDataset(maxlen, sep_token, cls_token, pad_token, text_seq, valid_seq, tokenizer)
    inf_dataset = inf_dataset[0]
    eval_batch = tuple(t.to('cpu') for t in inf_dataset)
    target_text,  valid_seq, attention_masks, segment_ids = eval_batch
    target_text = torch.unsqueeze(target_text, 0)
    valid_seq = torch.unsqueeze(valid_seq, 0)
    attention_masks = torch.unsqueeze(attention_masks, 0)
    outputs = model(input_ids=target_text,
                attention_mask=attention_masks,
                valid_ids =valid_seq)
        
    logits = outputs[1]
    return logits

def inference_total(sentence, tokenizer, modelA, modelB, modelC, q_pair, q_triplet):

    print("\n Original 문장: " + sentence + "\n")
    predicted_result = inference(sentence, tokenizer, modelA)
    predicted_result_occurred = inference(sentence, tokenizer, modelC)
    new_sent = post_process(predicted_result, sentence)
    print(" Entity Recognition 결과: " + new_sent + "\n")

    # new_sent_occurred = post_process_occurred(predicted_result_occurred, sentence)
    # print(" Occurred Recognition 결과: " + new_sent_occurred + "\n")
    # token_tokens, left_ids, right_ids, pairs, entity_pairs = inferred_sentence_to_input(sentence, predicted_result, tokenizer, q_pair)

    # sep_token = tokenizer.sep_token
    # cls_token = tokenizer.cls_token
    # pad_token = tokenizer.pad_token
    # maxlen = 200
    # modelB.to('cpu')
    # len_tot = len(left_ids)
    # inf_dataset = MEDInf_totDataset(maxlen, sep_token, cls_token, pad_token, token_tokens, left_ids, right_ids, tokenizer)
    # infs = []
    
    # for i in range(len_tot):
    #     inf_dataset1 = inf_dataset[i]
    #     eval_batch = tuple(t.to('cpu') for t in inf_dataset1)
    #     target_text, left_seq, right_seq, attention_masks, segment_ids = eval_batch
    #     target_text = torch.unsqueeze(target_text, 0)
    #     left_seq = torch.unsqueeze(left_seq, 0)
    #     right_seq = torch.unsqueeze(right_seq, 0)
    #     attention_masks = torch.unsqueeze(attention_masks, 0)
    #     outputs = modelB(input_ids =target_text, attention_mask=attention_masks, left_ids=left_seq, right_ids=right_seq)
    #     logits = outputs[1]
    #     logits = logits[0]
    #     logits = torch.argmax(F.log_softmax(logits, -1))
    #     logits = logits.detach().cpu().numpy()
    #     infs.append(logits)

    # # post_process_rel(pairs, infs, q_triplet, entity_pairs)

    # return pairs, infs, new_sent

def relation_pair(q_dic, a_dic, b_dic):
    keys = q_dic.keys()
    relation_items = []
    titles= []
    for key in keys:
        if key != 0 and key[0] == 'r':
            relation_items.append(q_dic[key])
            titles.append(key)
    pairs = []
    triplets = []
    for relation, title in zip(relation_items,titles):
        temps = relation.split("(")[1]
        tempss = temps.split("|")
        tempsss = (tempss[0], tempss[1][:-1])
        ta = (title, tempss[0], tempss[1][:-1])
        pairs.append(tempsss)
        triplets.append(ta)
    new_pairs = []
    new_triplets = []
    for pair in pairs:
        temps = (a_dic[pair[0]], a_dic[pair[1]])
        new_pairs.append(temps)
    for ts in triplets:
        new_triplets.append((b_dic[ts[0]], a_dic[ts[1]], a_dic[ts[2]]))
        
    return new_pairs, new_triplets

def post_process(predict_result, sentence):
    result = predict_result[0]
    sentence_split = sentence.split()
    
    result = result[1:-1]
    new_sent = ""
    for r, s in zip(result, sentence_split):
        if r != 0:
            new_sent = new_sent + " " + s + "/" + q_dic[a_inv_dic[r]]
        else:
            new_sent = new_sent + " " + s
    return new_sent

def post_process_occurred(predict_result, sentence):
    result = predict_result[0]
    sentence_split = sentence.split()
    
    result = result[1:-1]
    new_sent = ""
    for r, s in zip(result, sentence_split):
        if r == 1:
            new_sent = new_sent + " " + s + "/" +"Unstated"
        elif r==2:
            new_sent = new_sent + " " + s + "/" +"not_occurred"
        elif r==3:
            new_sent = new_sent + " " + s + "/" +"occurred"
        else:
            new_sent = new_sent + " " + s
    return new_sent

def triplet_check(pair, inf, re_pair):
    booa = False
    if (pair[0], pair[1]) in re_pair:
        booa = True
        
    return booa

def post_process_rel(pairs, infs, re_triplets, entity_pairs):
    for pair, inf, entity_pair in zip(pairs, infs, entity_pairs):
        if int(inf) != 0 and triplet_check(entity_pair, int(inf), re_triplets) == True:
            print(pair[0] + "와" + pair[1] + " 은(는) " + str(q_dic[b_inv_dic[re_triplet_dict[entity_pair]]]) + "의 관계에 있습니다\n")
    return 

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    bert_model = BertModel.from_pretrained('monologg/kobert')
    model = Bert_entity(bert_model, 24)
    model.load_state_dict(torch.load("./model_configs/sik_mybest", map_location=torch.device('cpu')))
    bert_model_2 = BertModel.from_pretrained('monologg/kobert')
    model_b = Bert_relation(bert_model_2, 2)
    model_b.load_state_dict(torch.load("./model_configs/sik_mybest_rel1", map_location=torch.device('cpu')))
    bert_model_3 = BertModel.from_pretrained('monologg/kobert')
    model_c = Bert_entity_occurred(bert_model_3, 4)
    model_c.load_state_dict(torch.load("./model_configs/sik_mybest_occc", map_location=torch.device('cpu')))
    re_pair, re_triplet = relation_pair(q_dic, a_dic, b_dic)

    re_triplet_dict = {}
    for tr in re_triplet:
        re_triplet_dict[tr[1], tr[2]] = tr[0]

    try:
        while True:
            ipt = input(">> User: ")
            sentence = str(ipt)
            a = inference_total(sentence, tokenizer, model, model_b,model_c, re_pair, re_pair)

    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('EXIT.')