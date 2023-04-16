#!/bin/bash
var1=/storage/LabJob/Projects/PipelineBuild/EricDataset/dst/audiopaths.tsv
var2=/storage/LabJob/Projects/PipelineBuild/outlabels
optvar1=/storage/LabJob/Projects/PipelineBuild/hubertfeat


. \
/storage/LabJob/Projects/PipelineBuild/scripts/ENVS.sh

ckpt_path=``$HUBERT_MODEL_PATH
layer=`    `9
km_path=`  `$HUBERT_KMEANS_PATH

# input
tsv_dir=`  `"$(dirname $var1)"
split=`    `"$(basename $var1 .tsv)"

# output
feat_dir=` `$optvar1
lab_dir=`  `$var2

# engineering
nshard=`   `3

if [[ ! -d $feat_dir ]]; then mkdir -p $feat_dir; fi
if [[ ! -d $lab_dir ]]; then
 >&2 echo 'No '$lab_dir' found! Make one...'
 mkdir -p $lab_dir; fi

for rank in $(seq 0 $((nshard - 1))); do

# TODO: RESAMPLE, shard 要寫！
python fairseq_utils/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
python fairseq_utils/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
# rm ${feat_dir}/${split}_${rank}_${nshard}.len  # 可留！ FIXME
rm ${feat_dir}/${split}_${rank}_${nshard}.npy
# TODO: 移除新的音檔

done




# for rank in $(seq 0 $((nshard - 1))); do
#   cat $lab_dir/${split}_${rank}_${nshard}.km
# done > $lab_dir/${split}.km





# ＃ＴＯＤＯ：MFA! (好像要 wav)
用起始點當標記，以不至於混淆！
