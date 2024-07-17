#!/bin/bash

cd transformer-ds/
for ds in *; do
  echo $ds
  cd $ds
  tfs predict -vk boxes --exp . > preds.ndjson
  lds add -k preds -l $(wc -l pgn.txt | awk '{print $1}')
  cd ..
done