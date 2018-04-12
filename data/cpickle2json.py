#coding:utf-8
#author:wup
#description: change cpickle to json!
#usage: python2 cpickle2json.py
#e-mail:wup@nlp.nju.cn
#date:2018.4.5
import cPickle as pkl
dev_data= pkl.load(open("kbqa_data/test.data.cfo.cpickle","rb"))
import json
dev_data = [t.__dict__ for t in dev_data]
json.dump(dev_data,open("test.data.cfo.json","w"))