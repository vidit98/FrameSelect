#!/bin/bash

LIST_OF_VIDEOS="data/DAVIS/JPEGImages/480p"

set `ls $LIST_OF_VIDEOS`

for i in $*
do
	echo "evaluating ${i}"
	python eval_DAVIS_crit1.py -g '2' -s challenge2 -D data/DAVIS  -v ${i}
done
