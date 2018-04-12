import os
GPUCARD = os.environ.get("CUDA_VISIBLE_DEVICES")
if GPUCARD == None:
    print("forget to choose gpu!!")
    print("export CUDA_VISIBLE_DEVICES=")
    os._exit(1)

import tensorflow as tf
import argparse

from focus_prune import run_op
from focus_prune.network import fp_model
from settings_fp import setting_fp


parser = argparse.ArgumentParser()
parser.add_argument("--run", default="train", help="run/test when run")
args = parser.parse_args()
run = args.run

FLAGS = setting_fp()

if not os.path.exists("fp_model"):
    os.mkdir("fp_model")
if not os.path.exists(FLAGS.answer_path):
    os.mkdir(FLAGS.answer_path)
tf.logging.set_verbosity(tf.logging.INFO)

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=tfconfig) as sess:
    model = fp_model(sess, FLAGS)
    if run == "train":
        run_op.train(model, FLAGS)
    elif run == "test":
        run_op.test(model, FLAGS)
    else:
        print("error in run! only accept train or test")
        exit(1)
