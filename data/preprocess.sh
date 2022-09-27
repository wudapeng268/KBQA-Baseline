#!/usr/bin/env bash

# 1. get raw data
echo "====> Step 1: download raw data"
mkdir RawData
cd RawData

wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz
tar -xzf SimpleQuestions_v2.tgz

wget https://www.dropbox.com/s/dt4i1a1wayks43n/FB5M-extra.tar.gz
tar -xzf FB5M-extra.tar.gz

wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
mv glove.6B/glove.6B.300d.txt ../kbqa_data/

#2. create entity2type type_voc rel_voc cpickle
cd ..
echo "====> Step 2: create entity2type type_voc rel_voc cpickle"
python create_entity2type.py
python create_rel_voc.py

echo "====> Step 3: create train data cpickle"
python prepare_train_data.py

echo "====> Step 4: rm dir RawData"
rm -r RawData
echo "====> Step 5: move generate data to kbqa_data"
mkdir kbqa_data
mv *.pickle kbqa_data/
echo "data preprocess all done!"






