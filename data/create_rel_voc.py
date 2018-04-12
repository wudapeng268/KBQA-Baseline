import sys

sys.path.append("../")
from util import FileUtil
import pickle as pkl

rel_voc = FileUtil.readFile("FB5M.relation.vocabulary")
rel_voc_map = {}
for i, t in enumerate(rel_voc):
    if i % 2 == 0:
        ss = t.split("\t")
        rel_voc_map[int(ss[0])] = ss[1]
        rel_voc_map[ss[1]] = int(ss[0])
# print("check it!")
# old_rel_voc = pkl.load(open("/home/user_data/wup/kbqa_data/rel_voc.cpickle", "rb"))
# assert rel_voc_map == old_rel_voc, "relation not equal!\n failed!"

pkl.dump(rel_voc_map, open("rel_voc.pickle", "wb"))

print("Done!")
