import sys
sys.path.append("../")
from Item import Item
from util import FileUtil
import nltk
import pickle as pkl

def get_anonymous(subtext, question):
    q_list = question.split(" ")
    sub_list = subtext.split(" ")
    t = 0
    start = -1
    for i, q in enumerate(q_list):
        if t < len(sub_list):
            if q == sub_list[t]:
                if t == 0:
                    start = i
                t += 1
            else:
                t = 0
        else:
            break
    if t == len(sub_list):
        q_list[start] = "X"
        for i in range(1, t):
            q_list[start + i] = "########"
    s = ""
    for q in q_list:
        if q != "########":
            s += (q + " ")
    return s[:-1]

def read_data(filename):
    context = FileUtil.readFile(filename)
    items = []
    for j, c in enumerate(context):
        questionTriple = c.split("\t")
        question = tokenizer(questionTriple[-1].lower())

        questionTriple[:-1] = [questionTriple[i].replace("www.freebase.com/", "").replace("/", ".") for i in
                               range(0, len(questionTriple) - 1)]
        sub = questionTriple[0]
        relation = rel_voc[questionTriple[1]]
        obj = questionTriple[2]
        if sub == 'm.07s9rl0':
            sub = 'm.02822'
        if obj == 'm.07s9rl0':
            obj = 'm.02822'

        if sub in entityId2type:
            sub_type = entityId2type[sub]
        else:
            sub_type = []
        if sub in enName:
            sub_text = enName[sub]
        else:
            sub_text = ""
        sub_text.split(" ")

        if sub_text in question:
            anonymous_question = get_anonymous(sub_text, question)
        else:
            anonymous_question = question
        it = Item(j, question, sub, relation, obj, list(sub_type), sub_text, anonymous_question)

        items.append(it)
    return items

def tokenizer(sentence):
    tokens = nltk.word_tokenize(sentence)

    # tokens = nltk.word_tokenize(sentence.decode("UTF-8"))
    s = " ".join(tokens)
    # return s.encode("UTF-8")
    return s

entityId2type = pkl.load(open("entity2type.fb5m.cpickle","rb"))

rel_voc = pkl.load(open("rel_voc.pickle","rb"))

enName = {}

def read_en_name():
    context = FileUtil.readFile("RawData/FB5M-extra/FB5M.en-name.txt")
    for c in context:
        tripe = c.split("\t")
        if tripe[1] == "<fb:type.object.en_name>":
            enName[tripe[0].replace("<fb:", "").replace(">", "")] = tokenizer(tripe[2][1:-1].lower())

read_en_name()

train_data = read_data("RawData/SimpleQuestions_v2/annotated_fb_data_train.txt")
pkl.dump(train_data, open("train.data.pickle", "wb"))

print("Done!")

