#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -pe smp.pe 4     # 4 core job

module load apps/binapps/anaconda/3/4.2.0
module load apps/gcc/tensorflow/1.2.1-py35-cpu

VOCAB=/path/to/vocabulary
SAVE_DIR=/path/to/save/model/dir
TRAIN_PREFIX=../data/1bw/train
OPTION=options/small128.json
N_TOKENS=30000000
WEIGHT=/path/to/weight/file

python -u bin/train_elmo.py \
--vocab_file $VOCAB \
--train_prefix $TRAIN_PREFIX \
--save_dir $SAVE_DIR \
--options_file $OPTION \
--batch 50 \
--n_epochs 10 \
--n_gpus 1 \
--n_tokens $N_TOKENS

python -u bin/dump_weights.py \
--save_dir $SAVE_DIR \
--outfile $WEIGHT