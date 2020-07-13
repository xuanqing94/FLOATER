# FLOATER
*This is the official implementation of *Learning to encode position for transformer with continuous dynamical model* in ICML 2020*

The codebase is modified upon [fairseq](https://github.com/pytorch/fairseq).

## Install the dependencies
In what follows I assume PyTorch(>=1.3) is installed.
```
# Install torchdiffeq
pip install git+git@github.com:xuanqing94/torchdiffeq.git

# Install fairseq
git clone https://github.com/xuanqing94/FLOATER.git
cd FLOATER
pip install -e .
```


## Run WMT14 En-De and En-Fr
First, download and preprocess the dataset:
```
# Download & bpe tokenize
bash examples/translation/prepare-wmt14en2de.sh --icml17
bash examples/translation/prepare-wmt14en2fr.sh
```
Then binarize the dataset:
```
TEXT=ende32k_wmt14
fairseq-preprocess \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEST/test \
    --source-lang en --target-lang de \
    --destdir data-bin/tokenized.en-de \
    --joined-dictionary \
    --workers 20

# Binarize En-Fr similarly.
```

Then, you can choose to train FLOATER from scratch but it will take a pretty long time until convergence. So, I recommend to follow the pretraining and finetuning approach discussed in the paper:
```
# Pretraining
fairseq-train \
    data-bin/wmt14.en-de/ \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --min-lr 1.0e-9 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2000 --save-dir ./checkpoints/base-flow \
    --warmup-init-lr 1.0e-7 --warmup-updates 4000 \
    --find-unused-parameters \
    --update-freq 2 \
    --reset-optimizer \
    --save-interval-updates 5 \
    --max-update 8 \

# initialize checkpoint_last.pt with original transformer model
python model_migration.py

# fine-tune FLOATER model for 10 more epochs
fairseq-train \
    data-bin/wmt14.en-de/ \
    --arch flow_transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 3.5e-4 --min-lr 1.0e-9 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2000 --save-dir ./checkpoints/base-flow \
    --warmup-init-lr 1.0e-7 --warmup-updates 4000 \
    --find-unused-parameters \
    --update-freq 2 \
    --reset-optimizer \
    --max-epoch 10 \

```
