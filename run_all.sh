#!/bin/bash

LIST_OF_VIDEOS="/home/vidit/models/UnVOS/STM/data/challenge/DAVIS"

set `ls $LIST_OF_VIDEOS`

for i in $*
do
	echo "evaluating ${i}"
	python eval_DAVIS_crit1.py -g '1' -s challenge -D /home/vidit/models/UnVOS/STM/data/challenge/DAVIS  -v ${i}
done