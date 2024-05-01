# 18751-Speech-Recognition-Project
Data Augmentation with Various Distortions Prepared by Torchaudio.

## Environment
```
cd espnet-path/egs2/librispeech_100/asr1
git clone https://github.com/chxw20/11751-Speech-Recognition-Project.git project
cd project
conda create -n aug python=3.9
conda activate aug
pip install tqdm
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Run
```
python main.py --config [path/to/config] --nprocs [number of processes]
python normalize.py --src_dset [dataset] --nprocs [number of processes]
cd ../
bash utils/combine_data.sh [combined_path] [normalized_path] [aug_path]
```
The normalization of original training, dev, and test datasets can be run only once. An example of using `noise.json` can be:
```
srun -p RM-shared -t 5:00:00 --ntasks-per-node=1 run.sh # python main.py --config noise.json --nprocs 8
srun -p RM-shared -t 5:00:00 --ntasks-per-node=1 run1.sh # normalize dev
srun -p RM-shared -t 5:00:00 --ntasks-per-node=1 run2.sh # normalize test
srun -p RM-shared -t 5:00:00 --ntasks-per-node=1 run3.sh # normalize train_clean_100
cd ../
bash utils/combine_data.sh data/train_aug-noise data/train_clean_100-noise data/train_clean_100-normalized # combine train
bash utils/combine_data.sh data/dev_norm data/dev_clean-normalized data/dev_other-normalized # combine dev
```
Then in `run_aug-noise.sh`:
```
#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_aug-noise"
valid_set="dev_norm"
test_sets="test_clean-normalized test_other-normalized dev_clean-normalized dev_other-normalized"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --stage 3 --stop-stage 13 \
    --batch_size 1 \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --gpu_inference true \
    --inference_nj 8 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_stats_dir "exp/${train_set}_stats" \
    --asr_exp "exp/${train_set}_train" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
```
## Experimental Configurations
| Config | CER (dev_clean) | CER (dev_other) | CER (test_clean) | CER (test_other) |
|:------:|:---------------:|:---------------:|:----------------:|:----------------:|
|Baseline|3.1              |10.8             |3.2               |10.8              |
|Effect  |2.9              |9.2              |3.0               |9.3               |
|Reverb  |3.5              |10.2             |3.4               |10.1              |
|Noise   |2.8              |9.1              |2.8               |9.2               |
|Scenes  |3.0              |9.3              |3.1               |9.4               |
