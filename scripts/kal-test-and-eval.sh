#!/bin/bash

# ./kal-test-and-eval.sh ahdr /home/ikenaga/liaojd/code/hdr-toolkit/models/kal-ahdr/checkpoint.pth /home/ikenaga/liaojd/data/kal_test /home/ikenaga/student-data/liaojd/data/result/kal/ahdr/ep15/ ~/liaojd/data/kal_test/
python ../hdr_toolkit/test.py --model-type "$1" --checkpoint "$2" --data kalantari --data-with-gt --input-dir "$3" --output-dir "$4" --out-activation sigmoid \
&& python ../hdr_toolkit/evaluation.py --result-dir "$4"  --reference-dir "$5" --dataset kalantari
