#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l k20

module load libs/gcc/pycuda/2017.1.1
module load libs/cuDNN/5.1.5
module load apps/binapps/anaconda/3/4.2.0
module load apps/gcc/tensorflow/1.2.1-py35-gpu

# one sentence per line
TARGET=/path/to/target/datafile
VOCAB=/path/to/vocabulary
OPTION=options/small128.json
WEIGHT=/path/to/weight/file
EMBEDDING=/path/to/save/embeddings

python -u bin/dump_cached_from_weights.py \
--target $TARGET \
--vocab_file $VOCAB \
--options_file $OPTION \
--weights $WEIGHT \
--embedding $EMBEDDING
