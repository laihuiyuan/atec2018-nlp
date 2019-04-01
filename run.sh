#!/bin/bash

python predict_cnn.py $1 
python predict_dssm.py $1
python predict_bidaf.py $1
python predict_deattn.py $1
python predict_lgb.py
python stack_lgb.py $2
