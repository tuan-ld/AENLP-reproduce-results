import streamlit as st

import torch
import torch.utils.data as data_utils



@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
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
