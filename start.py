#coding:utf-8
#author:wup
#description: start file
#e-mail:wup@nlp.nju.cn
#date: 2018.4.4

import os

GPUCARD = os.environ.get("CUDA_VISIBLE_DEVICES")
if GPUCARD == None:
    print("忘了指定 GPU 了吧")
    print("export CUDA_VISIBLE_DEVICES=")
    os._exit(1)

import tensorflow as tf
from settings import setting
from src.network import BiGRU
from src import run_op
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--run", type=str, default="run")
args = parse.parse_args()
run = args.run
FLAGS = setting()

if not os.path.exists("model"):
    os.mkdir("model")
if not os.path.exists(FLAGS.answer_path):
    os.mkdir(FLAGS.answer_path)
tf.logging.set_verbosity(tf.logging.INFO)

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=tfconfig) as sess:
    model = BiGRU(sess, FLAGS)
    if run == "train":
        run_op.train(model, FLAGS)
    elif run == "test":
        run_op.test(model, FLAGS, True)
    else:
        print("error in run! only accept train or test")
        exit(1)
