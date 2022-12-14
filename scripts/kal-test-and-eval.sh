#!/bin/bash

# model-type checkpoint-path input-dir output-dir
# ./kal-test-and-eval.sh ahdr /home/ikenaga/liaojd/code/hdr-toolkit/models/kal-ahdr/checkpoint.pth /home/ikenaga/liaojd/data/kal_test /home/ikenaga/student-data/liaojd/data/result/kal/ahdr/ep15/
python ../hdr_toolkit/test.py --model-type "$1" --checkpoint "$2" --data kalantari --data-with-gt --input-dir "$3" --output-dir "$4" --out-activation sigmoid \
&& python ../hdr_toolkit/evaluation.py --result-dir "$4/last"  --reference-dir "$3" --dataset kalantari \
&& python ../hdr_toolkit/evaluation.py --result-dir "$4/best-l"  --reference-dir "$3" --dataset kalantari \
&& python ../hdr_toolkit/evaluation.py --result-dir "$4/best-t"  --reference-dir "$3" --dataset kalantari
