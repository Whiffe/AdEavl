#!/bin/bash
# bash DyGen.sh mmlu_sample_300 2 Qwen mmlu
# bash DyGen.sh mmlu_sample_300 2 Doub mmlu


# bash DyGen.sh arc_sample_300 2 Doub ai2_arc
name=$1
kn_num=$2
model=$3
dataset=$4

python 01DyGenKN.py \
    --prompt ./promptEn/choiceKN.txt \
    --dataset ./dataset/formatDataset/$dataset/$name.json \
    --output ./output/$dataset/KN_$name.json \
    --fewshot ./promptEn/few-shot/choiceKN.txt
    
python 02DyGenPurport.py \
    --prompt ./promptEn/choicePurport.txt \
    --dataset ./output/$dataset/KN_$name.json \
    --output ./output/$dataset/Purport_$name.json \
    --fewshot ./promptEn/few-shot/choicePurport.txt
    
python 03DyGenQnet.py \
    --prompt ./promptEn/KNexplain.txt \
    --dataset ./output/$dataset/Purport_$name.json \
    --output ./output/$dataset/KNexplain_$name.json \
    --knowledge_points_num $kn_num \
    --fewshot ./promptEn/few-shot/KNexplain.txt
    
python 04DyGenQsetQs.py \
    --prompt ./promptEn/setCQ.txt \
    --dataset ./output/$dataset/KNexplain_$name.json \
    --model $model \
    --outputCQ ./output/$dataset/DyCQ_$name.json \
    --fewshot ./promptEn/few-shot/setCQ.txt