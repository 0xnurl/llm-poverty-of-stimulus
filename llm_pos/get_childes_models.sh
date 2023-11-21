#!/bin/bash

CHILDES_DIR=../models/yedetore
mkdir -p $CHILDES_DIR || true
cd $CHILDES_DIR

wget http://www.adityayedetore.com/childes-project/vocab.txt # Correct file, the vocab.txt file provided in pretraining.zip is incorrect.

mkdir lstm || true
cd lstm


for i in {1..10}
  do
  if [ $i -lt 10 ]; then
    num="0$i"
  else
    num="$i"
  fi
  wget --no-clobber "http://www.adityayedetore.com/childes-project/models/LSTM_final/2-800-10-20-0.4-10$num-LSTM-model.pt"
  wget --no-clobber "http://www.adityayedetore.com/childes-project/models/LSTM_final/2-800-10-20-0.4-10$num-LSTM-log.txt"
done

cd ../transformer/

for i in {1..10}
  do
  if [ $i -lt 10 ]; then
    num="0$i"
  else
    num="$i"
  fi
  wget --no-clobber "http://www.adityayedetore.com/childes-project/models/Transformer_final_state_dict/04-500-800-10-0.2-5.0-04-10${num}-Transformer-state-dict.pt"
  wget --no-clobber "http://www.adityayedetore.com/childes-project/models/Transformer_final/04-500-800-10-0.2-5.0-04-10${num}-Transformer-log.txt"
done
