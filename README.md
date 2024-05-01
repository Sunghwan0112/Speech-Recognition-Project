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
## Data Augmentation Preparation

Baseline was simply the original libre speech data without any augmenataion techniques.
First time, I used specaugmenation,
Second time, I used specaugmentation + noise
Lastly, I used specaugmentation + noise + noise
Espnet already has built-in speed pertubation data augmentation.
So I used the speed perturbation (x0.9, 1, 1.1) for the augmented data. 

## Run
train_new = the original train data + augmented data from the original train data.
I Combined the both data for training. 
Combination of the files are then stored in train_new folder.

Then in `run.sh`:
```
#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_new"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
```

## Experimental Configurations
1st augmenatation technique = Speed + Specaugmentation 

| Config | CER (dev_clean) | CER (dev_other) | CER (test_clean) | CER (test_other) |
|:------:|:---------------:|:---------------:|:----------------:|:----------------:|
|Baseline|3.1              |10.8             |3.2               |10.8              |
|Spe  |2.9              |9.2              |3.0               |9.3               |
|Reverb  |3.5              |10.2             |3.4               |10.1              |
|Noise   |2.8              |9.1              |2.8               |9.2               |
|Scenes  |3.0              |9.3              |3.1               |9.4               |
