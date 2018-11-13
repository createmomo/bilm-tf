#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -pe smp.pe 4     # 4 core job

module load apps/binapps/anaconda/3/4.2.0
module load apps/gcc/tensorflow/1.2.1-py35-cpu

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
