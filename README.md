# Knowledge Based Question Answering

# Update
We update `dev.small.pickle` and `test.data.cfo.pickle` in [baidu pan](https://pan.baidu.com/s/16Qq0RiZAEN3m8LEnMJrRFA)  key: c9mi or [google drive](https://drive.google.com/drive/folders/1cmO7QdcUgNeB8hpsE5Bj6AdtIFKbHyCW?usp=sharing)
# Description
 

This is the code of KBQA baseline following [CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](https://arxiv.org/pdf/1606.01994.pdf). Similar with the CFO, we also use type vector(one-hot) to repesent the entity, But we change something for our purpose.

Different from CFO
1. We are training both subject and relation at the same time.
2. We use log-likelihood to replace relation loss in "CFO", we also calculate all relation in FB5M to get the probability.
3. We use Adam optimizer to replace AdaGrad with momentum in CFO.
4. In CFO, they use alpha to add subject score to relation score, which need to fine-turn this hyper-parameter, we directly add subject score and relation score 1:1, then multiply alpha matrix, which alpha matrix is the binary function of subject to relation in KB. 
5. Also we share the word embedding between subject and relation, from our experiment this is no matter for model.

# Requirement

python 3.5+

tensorflow 1.2+

nltk

# Usage

## Preprocess

please run `data/preprocess.sh` to download raw data and generate train data.
Important! Because of [freebase API was deprecated](https://developers.google.com/freebase/)，we can't use this API to get candidate when focus prune, so we use valid candidate and test candidate from CFO(the path is `data/kbqa_data/dev.small.pickle data/kbqa_data/test.data.cfo.pickle`). Now thanks the support of Zihang Dai. 

## Focus Prune

See example in `fp_train.sh`
See example in `fp_test.sh`
After train and test, you can see `sq.dev.label` `sq.test.label` in `fp_output` directory, your model saved in `fp_model` directory.

## Entity Match

Because of [freebase API was deprecated](https://developers.google.com/freebase/), we only use strict match to get the candidate of each question. 

# Configuration

relation subject network setting in `setting.py`
Focus prune setting in `setting_fp.py`

# Training

See example in `train.sh`

# Testing

See example in `test.sh`

After test, you can see `sq.all.txt` in `output` directory, your model saved in `model` directory

# Performance

## kbqa

View |subject | relation | all
 --- | --- | --- |---
acc| 0.797759 | 0.829391 | 0.732743

## Focus Prune

dataset | pred|recall|f1
---| --- | ---| ---|
dev| 0.9053|0.8129|0.8566
test |0.8982|0.8230|0.8589

