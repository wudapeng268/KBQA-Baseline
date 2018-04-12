# coding:utf-8
# author:wup
# description:
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:
import nltk

from src.util import FileUtil

Max = 0


def readSimpleQAData(filename):
    context = FileUtil.readFile(filename)
    items = []
    output = []
    for c in context:
        questionTriple = c.split("\t")
        question = tokenizer(questionTriple[-1].lower())
        output.append(questionTriple[0] + "\t" + questionTriple[1] + "\t" + questionTriple[2] + "\t" + question)
    FileUtil.writeFile(output, "../data/" + filename[:-3] + "pre.txt")


def tokenizer(sentence):
    tokens = nltk.word_tokenize(sentence.decode("UTF-8"))
    s = ""
    global Max
    if len(tokens) > Max:
        Max = len(tokens)
    for t in tokens:
        s += t + " "
    s = s[:-1]
    return s.encode("UTF-8")


readSimpleQAData("../data/annotated_fb_data_test.txt")
readSimpleQAData("../data/annotated_fb_data_train.txt")
readSimpleQAData("../data/annotated_fb_data_valid.txt")
