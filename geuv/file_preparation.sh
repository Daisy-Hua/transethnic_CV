#!/usr/bin/env bash

cat $1 |while read line;
do
  ID=$(echo $line | awk '{print $1}')
  name=$(echo $line | awk '{print $2}')
  ./preparation.sh $ID $name
done


