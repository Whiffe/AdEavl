#!/bin/bash
# bash Reconstruct.sh mmlu_sample_300 -1

name=$1
delta=$2

# 执行第一个Python脚本
python 30ExtractDataReconstruct.py \
    --delta $delta \
    --dataset ./output/mmlu/DyCQ_Answer/Answer_DyCQ_$name.json \
    --output_select ./output/mmlu/Reconstruct_Complex/Reconstruct_$name.json \
    --output_remain ./output/mmlu/Reconstruct_Complex/Remain_$name.json


# 执行第二个Python脚本
python 31DyReconstructQs.py \
    --prompt ./promptEn/reconstructCQUp.txt \
    --dataset ./output/mmlu/Reconstruct_Complex/Reconstruct_$name.json \
    --outputCQ ./output/mmlu/Reconstruct_Complex/Reconstructed_$name.json \
    --fewshot ./promptEn/few-shot/setCQ_reconstruct.txt


python 32MergeJson.py \
    --json1 ./output/mmlu/Reconstruct_Complex/Reconstructed_$name.json \
    --json2 ./output/mmlu/Reconstruct_Complex/Remain_$name.json \
    --output ./output/mmlu/Reconstruct_Complex/DyCQ_Reconstructed_$name.json

echo "所有脚本执行完毕。"