import json
import spacy
from collections import defaultdict
from tqdm import tqdm
import time
import pickle
from json import JSONDecodeError

# initialize language model
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)

dict_query_train_ent = {}
dict_query_dev_ent = {}
dict_query_test_ent = {}
dict_ctx_ent = {}
dict_title2id = defaultdict(list)

with open('./NQ/test.json', 'r') as f:
    NQ_data = json.load(f)
    
for NQ_idx, NQ_ins in enumerate(tqdm(NQ_data)):
    
    #question el
    dict_query_train_ent[NQ_idx] = []
    NQ_question = NQ_ins['question']
    doc_el = nlp(NQ_question)
    for ent in doc_el._.linkedEntities:
        dict_query_train_ent[NQ_idx].append(ent.get_id())
        
    #ctx el
    for NQ_ins_ctx in NQ_ins['ctxs']:
        if NQ_ins_ctx['id'] in dict_ctx_ent:
            continue
        dict_title2id[NQ_ins_ctx['title']].append(NQ_ins_ctx['id'])
        dict_ctx_ent[NQ_ins_ctx['id']] = []
        doc_el = nlp(NQ_ins_ctx['text'])
        for ent in doc_el._.linkedEntities:
            dict_ctx_ent[NQ_ins_ctx['id']].append(ent.get_id())

with open('./dict_query_train_ent.pkl', 'wb') as f:
    pickle.dump(dict_query_train_ent, f)