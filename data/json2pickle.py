#coding:utf-8
#author:wup
#description:change json 2 pickle in python3
#usage python3 json2pickle.py
#e-mail:wup@nlp.nju.cn
#date:2018.4.5
import pickle as pkl
import json
dev_data_json = json.load(open("test.data.cfo.json","r"))
from Item import Item
dev_data=[]
for t in dev_data_json:
    it=Item(t['qid'],t['question'],t['subject'],t['relation'],t['object'],t['gold_type'],t['subject_text'],t['anonymous_question'])
    if 'cand_rel' in t:
        it.cand_rel=t['cand_rel']
    if 'cand_sub' in t:
        it.cand_sub=t['cand_sub']
    if 'sub_rels' in t:
        it.sub_rels = t['sub_rels']
    dev_data.append(it)
pkl.dump(dev_data,open("test.data.cfo.pickle","wb"))