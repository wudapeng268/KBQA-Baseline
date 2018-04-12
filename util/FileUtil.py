# coding:utf-8
# author:wup
# description:读文件，写文件
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:2017-4-16

def readFile(filename):
    context = open(filename).readlines()
    return [c.strip() for c in context]


def writeFile(context, filename, append=False):
    if not append:
        with open(filename, 'w+') as fout:
            for co in context:
                fout.write(co + "\n")
    else:
        with open(filename, 'a+') as fout:
            for co in context:
                fout.write(co + "\n")


def list2str(l, split=" "):
    a = ""
    for li in l:
        a += (str(li) + split)
    a = a[:-len(split)]
    return a
