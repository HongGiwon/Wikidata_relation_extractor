# KB_triple_extractor

## Description

For a given query and n passages, this code extracts triples between (inter/intra) passages and a query. The dataset used in this code is Open NQ, which consists of queries of the original Natural Questions, and evidence documents retrieved through DPR (https://github.com/facebookresearch/DPR). You can get the dataset from: <code>get-data.sh </code> in https://github.com/facebookresearch/FiD. 

What this code does
1. Perform entity linking in given queries and passages (entity_linking.py).
2. Obtain a relaton list that can exist between linked entities, using Wikidata (KB) API (NQ_triple_extractor.py).
3. Filter relations using TFIDF scores. 

## Packages

Entity linker from https://github.com/egerber/spaCy-entity-linker

Wikidata: https://qwikidata.readthedocs.io/en/stable/readme.html

## Installation

Requirements: <code>pip install -r requirements.txt</code>

<code>python -m spacy download en_core_web_md</code>

<code>python -m spacy_entity_linker "download_knowledge_base"</code>

## Run

<code>python entity_linking.py</code>

<code>python NQ_triple_extractor.py</code>

```python

```
