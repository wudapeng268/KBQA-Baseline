# coding:utf-8
# author:wup
# description: create type vocabulary saving in cpickle and the mapping between entity and their type`
# e-mail:wup@nlp.nju.cn
# date:2018.4.5
import sys
sys.path.append("../")
from util import FileUtil
import pickle as pkl
context = FileUtil.readFile("RawData/FB5M-extra/FB5M.type.txt")
type_voc = {}

alltypes = set()
for c in context:
    triple = c.split("\t")
    for i, t in enumerate(triple):
        triple[i] = t.replace("<fb:", "").replace(">", "")
    alltypes.add(triple[2])
alltypes = list(alltypes)
for i, t in enumerate(alltypes):
    type_voc[t] = i
    type_voc[i] = t

pkl.dump(type_voc, open("fb5m.type_voc.pickle", "wb"))
print("Done!")

entity2type = {}
for t in context:
    id = t.split("\t")[0].replace("<fb:", "").replace(">", "")
    name = t.split("\t")[2].replace("<fb:", "").replace(">", "")
    type_id = type_voc[name]
    if id not in entity2type:
        entity2type[id] = set([type_id])
    else:
        entity2type[id].add(type_id)

pkl.dump(entity2type, open("entity2type.fb5m.pickle", "wb"))

print("Done!")
