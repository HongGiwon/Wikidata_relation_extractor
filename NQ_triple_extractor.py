import json
import spacy  # version 3.0.6'
from collections import defaultdict
from tqdm import tqdm
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)
import time
import pickle
import math
from json import JSONDecodeError

k1 = 1.2
b = 0.75
thre_tfidf = 10.0
dataset_name = "test"

# +
with open('./NQ/' + dataset_name + '.json', 'r') as f:
    NQ_data = json.load(f)

with open('./dict_query_train_ent.pkl', 'rb') as f:
    dict_query_train_ent = pickle.load(f)
with open('./dict_query_dev_ent.pkl', 'rb') as f:
    dict_query_dev_ent = pickle.load(f)
with open('./dict_query_test_ent.pkl', 'rb') as f:
    dict_query_test_ent = pickle.load(f)
with open('./dict_ctx_ent.pkl', 'rb') as f:
    dict_ctx_ent = pickle.load(f)
with open('./dict_set_query_train_ent.pkl', 'rb') as f:
    dict_set_query_train_ent = pickle.load(f)
with open('./dict_set_query_dev_ent.pkl', 'rb') as f:
    dict_set_query_dev_ent = pickle.load(f)
with open('./dict_set_query_test_ent.pkl', 'rb') as f:
    dict_set_query_test_ent = pickle.load(f)

# query id set
all_query_set = set()
for qid in tqdm(dict_query_train_ent):
    all_query_set.update(set(dict_query_train_ent[qid]))
for qid in tqdm(dict_query_dev_ent):
    all_query_set.update(set(dict_query_train_ent[qid]))
for qid in tqdm(dict_query_test_ent):
    all_query_set.update(set(dict_query_train_ent[qid]))

# context set
all_context_set = set()
for qid in tqdm(dict_ctx_ent):
    all_context_set.update(set(dict_ctx_ent[qid]))

# +
# TFIDF score for each passage
all_doc_num = len(dict_ctx_ent)
avg_doc_len = 0
entity_DF = defaultdict(int)

for did in tqdm(dict_ctx_ent):
    avg_doc_len += len(dict_ctx_ent[did])
    for eid in set(dict_ctx_ent[did]):
        entity_DF[eid] += 1

avg_doc_len /= all_doc_num

entity_IDF = {}

for eid in tqdm(entity_DF):
    entity_IDF[eid] = math.log((all_doc_num-entity_DF[eid]+0.5) / (entity_DF[eid]+0.5))

entity_TFIDF = {}

for did in tqdm(dict_ctx_ent):
    entity_TFIDF[did] = {}
    for eid in set(dict_ctx_ent[did]):
        f_de = dict_ctx_ent[did].count(eid)
        TF_de = (f_de * (k1 + 1)) / (f_de + k1 * (1 - b + b * (len(dict_ctx_ent[did]) / avg_doc_len)))
        entity_TFIDF[did][eid] = entity_IDF[eid] * TF_de

# +
with open('./entity_TFIDF.pkl', 'wb') as f:
    pickle.dump(entity_TFIDF, f)

avg_ent_num_ori = 0
for did in tqdm(entity_TFIDF):
    for eid in entity_TFIDF[did]:
        avg_ent_num_ori += 1

avg_ent_num_ori /= all_doc_num

# +
dict_set_ctx_ent_tdidf = {}
avg_ent_num = 0
for did in tqdm(entity_TFIDF):
    dict_set_ctx_ent_tdidf[did] = set()
    for eid in entity_TFIDF[did]:
        if entity_TFIDF[did][eid] > thre_tfidf:
            dict_set_ctx_ent_tdidf[did].add(eid)
            avg_ent_num += 1

avg_ent_num /= all_doc_num
print("avg num of ent per doc",avg_ent_num)
print("reduce in comp",(avg_ent_num_ori / avg_ent_num) **2)
# -

dict_propid2value = {}
dict_entityid2value = {}

def wikidata_triple_retrieve(e1, e2):
    time.sleep(0.5)

    entity_s = " ".join(["wd:Q"+str(qid) for qid in e1])
    entity_o = " ".join(["wd:Q"+str(qid) for qid in e2])
    sparql_query = """
    SELECT DISTINCT ?e1 ?e1Label ?e2 ?e2Label ?item ?propLabel
    WHERE 
    {
      VALUES ?e1 { """ + entity_s + """ } 
      VALUES ?e2 { """ + entity_o + """ } 
      ?e1 ?item ?e2.
      SERVICE wikibase:label { 
          bd:serviceParam wikibase:language "en". 
          ?e1 rdfs:label ?e1Label . 
          ?e2 rdfs:label ?e2Label . 
          ?item rdfs:label ?itemLabel . 
          ?prop rdfs:label ?propLabel . 
      }
      ?prop wikibase:directClaim ?item .
    }
    """
    res = return_sparql_query_results(sparql_query)
    return_list = []
    for prop_item in res['results']['bindings']:
        prop_id = prop_item['item']['value'].split('/')[-1]
        prop_value = prop_item['propLabel']['value']
        e1_id = prop_item['e1']['value'].split('/')[-1]
        e1_label = prop_item['e1Label']['value']
        e2_id = prop_item['e2']['value'].split('/')[-1]
        e2_label = prop_item['e2Label']['value']
        
        if not prop_id in dict_propid2value:
            dict_propid2value[prop_id] = prop_value
        if not e1_id in dict_entityid2value:
            dict_entityid2value[e1_id] = e1_label
        if not e2_id in dict_entityid2value:
            dict_entityid2value[e2_id] = e2_label
        return_list.append((e1_id,e2_id,prop_id))
    return return_list


batch_size = 200
NQ_q2c_triples = []
NQ_c2c_triples = []
error_list = []
for NQ_idx, NQ_ins in enumerate(tqdm(NQ_data)):
    try:
        Q_entity = dict_set_query_test_ent[NQ_idx]

        C_entity_batch = set()
        for NQ_ins_ctx_idx, NQ_ins_ctx in enumerate(NQ_data[NQ_idx]['ctxs'][:10]):
            C_entity_batch.update(dict_set_ctx_ent_tdidf[NQ_ins_ctx['id']])

        result_Q = wikidata_triple_retrieve(Q_entity, C_entity_batch)
        result_Q += wikidata_triple_retrieve(C_entity_batch, Q_entity)

        result_C = []
        C_entity_batch_list = list(C_entity_batch)
        for i in range(int(len(C_entity_batch) / batch_size) + bool(len(C_entity_batch) % batch_size)):
            result_C += wikidata_triple_retrieve(C_entity_batch, C_entity_batch_list[i*batch_size:(i+1)*batch_size])
        NQ_c2c_triples.append(result_C)
        NQ_q2c_triples.append(result_Q)
    except (JSONDecodeError, ConnectionError, TimeoutError, MaxRetryError) as error:
        time.sleep(60)
        error_list.append(NQ_idx)
        NQ_c2c_triples.append([])
        NQ_q2c_triples.append([])

# error handling
batch_size = 20
for NQ_idx in tqdm(error_list):
    print(NQ_idx)
    Q_entity = dict_set_query_test_ent[NQ_idx]
    C_entity_batch = set()
    for NQ_ins_ctx_idx, NQ_ins_ctx in enumerate(NQ_data[NQ_idx]['ctxs'][:10]):
        C_entity_batch.update(dict_set_ctx_ent_tdidf[NQ_ins_ctx['id']])
    result_Q = wikidata_triple_retrieve(Q_entity, C_entity_batch)
    result_Q += wikidata_triple_retrieve(C_entity_batch, Q_entity)
    
    result_C = []
    C_entity_batch_list = list(C_entity_batch)
    for i in range(int(len(C_entity_batch) / batch_size) + bool(len(C_entity_batch) % batch_size)):
        result_C += wikidata_triple_retrieve(C_entity_batch, C_entity_batch_list[i*batch_size:(i+1)*batch_size])
    NQ_c2c_triples[NQ_idx] += result_C
    NQ_q2c_triples[NQ_idx] += result_Q
