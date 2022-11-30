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