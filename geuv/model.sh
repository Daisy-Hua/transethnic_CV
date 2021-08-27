#!/usr/bin/env bash

python code/D_solver/CV_model.py $1 $2 >> result/refined.txt
echo "$2 Done"