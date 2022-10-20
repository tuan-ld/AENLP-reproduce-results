import streamlit as st



@st.cache(allow_output_mutation=True)
def tokenizing_(token_list,  tokenizer):
    text_seq = []
    valid_seq = []
    ## target_seq ~ target_text_seq 똑같
   
    for tokens in token_list:
        new_target_text = []
        new_valid = []
        
        for t in tokens:
            tokens_wordpiece = tokenizer.tokenize(t)
            new_v = [1] + [0]*(len(tokens_wordpiece)-1)
            new_target_text.extend(tokens_wordpiece)
            new_valid.extend(new_v)
            
        valid_seq.append(new_valid)
        text_seq.append(new_target_text)
       
    return text_seq, valid_seq


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
def sentence_to_tokens(sentence):
    text_seq = []
    q = sentence.split()
    text_seq.append(q)
    
    return text_seq


@st.cache(allow_output_mutation=True)
def sentence_to_input(sentence, tokenizer):
    tokens = sentence_to_tokens(sentence)
    text_seq, valid_seq = tokenizing_(tokens, tokenizer)
    
    return text_seq, valid_seq


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
def token_label_together(token_list, label_list):
    new_token_list = []
    for token, label in zip(token_list, label_list):
        new_token = []
       
        for tok, lab in zip(token, label):
            if lab != 0 and lab != -1:
                new_token.append(tok +"/"+str(lab))
            else:
                new_token.append(tok)
        new_token_list.append(new_token)
    return new_token_list


@st.cache(allow_output_mutation=True)
def make_pair_by_max(value, ordered_tokens, entities, q_pair):
    lefts = []; rights = []; pairs = []
    entity_pairs =[]
    for i in range(value+1):
        for j in range(value+1):
            if i != j and entitie_check(entities[j], entities[i], q_pair) == True:
                pairs.append((ordered_tokens[j], ordered_tokens[i]))
                lefts.append(j)
                rights.append(i)
                entity_pairs.append((entities[j], entities[i]))
    return lefts, rights, pairs, entity_pairs


@st.cache(allow_output_mutation=True)
def entitie_check(l, r, q_pair):
    booa = False
    if (l, r) in q_pair:
        booa = True
    return booa    


@st.cache(allow_output_mutation=True)
def make_left_rights_sents(lefts, rights, labels):
    left_ids = []; right_ids = []
    for le, ri in zip(lefts, rights):
        left_id = []; right_id = []
        
        for label in labels:
            if label == le:
                left_id.append(1)
                right_id.append(0)
            elif label == ri:
                right_id.append(1)
                left_id.append(0)
            else:
                left_id.append(0)
                right_id.append(0)
        left_ids.append(left_id)
        right_ids.append(right_id)
        
    return left_ids, right_ids