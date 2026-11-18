#!/usr/bin/env bash
# Exp 11 Part B on a GPU pod: rebuild Exp 04 data, pretrain a prior per size, then
# 5 sizes x 6 orders x 3 seeds with the path ensemble. Status lines go to /workspace/out/STATUS.
cd /workspace
OUT=/workspace/out; mkdir -p $OUT
log() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/STATUS; }
log "exp11 started"
[ -d learning-order ] || git clone -q https://github.com/inny0127/learning-order.git
cd learning-order && git pull -q; log "code $(git log --oneline -1)"
pip install -q tokenizers matplotlib > /dev/null 2>&1
RAW=/workspace/raw; mkdir -p $RAW
HF=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
[ -f $RAW/valid.txt ] || curl -sSL -o $RAW/valid.txt $HF/TinyStoriesV2-GPT4-valid.txt
[ -f $RAW/train200.txt ] || curl -sSL -r 0-209715199 -o $RAW/train200.txt $HF/TinyStoriesV2-GPT4-train.txt
cd exp04_curriculum
[ -f data/people.json ] || python curriculum.py prep --raw $RAW > $OUT/prep.log 2>&1
if (cd data && md5sum -c ../data_md5.txt) > $OUT/md5.txt 2>&1; then log "data identical"; else log "DATA MISMATCH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in m1 m4 m16 m64 m128; do echo $s; done | xargs -P 3 -I{} sh -c '[ -f runs_prior/{}/reference.pt ] || python curriculum.py prior --size {} --device cuda --runs runs_prior/{} > runs_prior_{}.log 2>&1'
log "priors done: $(ls runs_prior/*/reference.pt | wc -l)/5"
cd ../exp05_order_compression && mkdir -p runs11
{ for z in m1 m4 m16 m64 m128; do for s in 1 2 3; do for o in random easy_first hard_first maxloss rho lp; do echo "$z $o $s"; done; done; done; } | \
  xargs -P ${WORKERS:-4} -n 3 sh -c '[ -f runs11/order/$1_$0_s$2.json ] || python order.py --size $0 --order $1 --seed $2 --device cuda --threads 1 --runs4 ../exp04_curriculum/runs_prior/$0 --runs runs11 --path_ensemble --tag _$0 > runs11/log_$1_$0_s$2.txt 2>&1'
log "EXP11 DONE: $(ls runs11/order/*.json | wc -l)/90"
