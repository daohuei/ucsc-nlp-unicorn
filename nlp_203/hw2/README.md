# NLP 203 Homework 2

## Prerequisite

-   Python 3.8.2
-   [Fairseq](https://github.com/pytorch/fairseq) Need to install Fairseq for producing the result
-   Need to download iwslt13_fr_en dataset

## Output the dataset from raw data

Use the `output_dataset.ipynb` for extracting the data.

## Tokenize the data

```bash
# with bpe
bash prepare_data.sh

# no bpe
bash prepare_data_no_bpe.sh
```

## Preprocess the data

```bash
# with bpe
TEXT=iwslt13_fr_en ; mkdir -p data-bin ; fairseq-preprocess --source-lang fr --target-lang en     --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test     --destdir data-bin/iwslt13_fr_en     --workers 20

# no bpe
TEXT=iwslt13_fr_en_no_bpe ; mkdir -p data-bin ; fairseq-preprocess --source-lang fr --target-lang en     --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test     --destdir data-bin/iwslt13_fr_en_no_bpe     --workers 20

# share vocab
TEXT=iwslt13_fr_en ; mkdir -p data-bin ; fairseq-preprocess --source-lang fr --target-lang en     --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test     --destdir data-bin/iwslt13_fr_en_shared_vocab     --workers 20 --joined-dictionary
```

## Training the model

```bash
# Training Transformer:

CUDA_VISIBLE_DEVICES=5 fairseq-train     data-bin/iwslt13_fr_en     --arch transformer --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 3000     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/tf_bpe --patience 10

# no bpe:

CUDA_VISIBLE_DEVICES=0 fairseq-train     data-bin/iwslt13_fr_en_no_bpe     --arch transformer --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 3000    --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/tf_no_bpe --patience 10

# share_vocab:
CUDA_VISIBLE_DEVICES=0 fairseq-train     data-bin/iwslt13_fr_en_shared_vocab     --arch transformer --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 3000    --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/tf_shared_vocab --patience 10


# Training CNN:

CUDA_VISIBLE_DEVICES=1 fairseq-train \
    data-bin/iwslt13_fr_en \
    --arch fconv \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 3000 \
    --save-dir checkpoints/cnn_bpe \
    --fp16 --patience 10

# no bpe:

CUDA_VISIBLE_DEVICES=1 fairseq-train     data-bin/iwslt13_fr_en_no_bpe     --arch fconv \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 3000    --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/cnn_no_bpe \
    --fp16 --patience 10

# share_vocab:

CUDA_VISIBLE_DEVICES=0 fairseq-train     data-bin/iwslt13_fr_en_shared_vocab     --arch fconv \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 3000    --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/cnn_shared_vocab \
    --fp16 --patience 10
```

## Evaluation

```bash
# Transformer
bpe
valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu --sacrebleu --results-path result/tf_bpe.pred --gen-subset valid ; \

Generate valid with beam=5: BLEU = 31.91 62.7/38.6/26.1/18.0 (BP = 0.976 ratio = 0.977 hyp_len = 20232 ref_len = 20717)


test
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu --sacrebleu --results-path result/tf_bpe.pred ; \


Generate test with beam=5: BLEU = 37.13 64.9/42.8/30.5/22.4 (BP = 1.000 ratio = 1.012 hyp_len = 35566 ref_len = 35154)

no bpe:
valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_no_bpe \
    --path checkpoints/tf_no_bpe/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --scoring sacrebleu --sacrebleu --results-path result/tf_no_bpe.pred --gen-subset valid ; \

Generate valid with beam=5: BLEU = 29.04 59.3/35.7/23.8/16.2 (BP = 0.966 ratio = 0.967 hyp_len = 21925 ref_len = 22676)

test
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_no_bpe \
    --path checkpoints/tf_no_bpe/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --scoring sacrebleu --sacrebleu --results-path result/tf_no_bpe.pred ; \

Generate test with beam=5: BLEU = 35.33 63.4/41.4/29.1/21.0 (BP = 0.993 ratio = 0.993 hyp_len = 35756 ref_len = 36006)


shared vocab:
valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_shared_vocab \
    --path checkpoints/tf_shared_vocab/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe  \
    --scoring sacrebleu --sacrebleu --results-path result/tf_shared_vocab.pred --gen-subset valid ; \

Generate valid with beam=5: BLEU = 31.98 63.2/39.2/26.4/18.3 (BP = 0.968 ratio = 0.968 hyp_len = 20025 ref_len = 20682)


test
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_shared_vocab \
    --path checkpoints/tf_shared_vocab/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe  \
    --scoring sacrebleu --sacrebleu --results-path result/tf_shared_vocab.pred ; \

Generate test with beam=5: BLEU = 36.94 65.2/42.8/30.3/22.0 (BP = 1.000 ratio = 1.011 hyp_len = 35517 ref_len = 35120)




# CNN
bpe
valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en \
    --path checkpoints/fconv_iwslt13_fr_en/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_bpe.pred --gen-subset valid ; \

Generate valid with beam=5: BLEU = 29.64 60.5/36.2/23.6/15.7 (BP = 0.988 ratio = 0.988 hyp_len = 20469 ref_len = 20717)


test
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en \
    --path checkpoints/fconv_iwslt13_fr_en/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_bpe.pred ; \

Generate test with beam=5: BLEU = 34.21 62.8/40.1/27.7/19.6 (BP = 1.000 ratio = 1.026 hyp_len = 36080 ref_len = 35154)


no bpe

valid

CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_no_bpe \
    --path checkpoints/cnn_no_bpe/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_no_bpe.pred --gen-subset valid

Generate valid with beam=5: BLEU = 26.72 60.5/35.4/22.9/15.3 (BP = 0.909 ratio = 0.913 hyp_len = 20702 ref_len = 22676)


test

CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_no_bpe \
    --path checkpoints/cnn_no_bpe/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_no_bpe.pred

Generate test with beam=5: BLEU = 32.59 62.0/38.9/26.4/18.4 (BP = 0.990 ratio = 0.990 hyp_len = 35663 ref_len = 36006)

shared vocab:
valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_shared_vocab \
    --path checkpoints/cnn_shared_vocab/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe  \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_shared_vocab.pred --gen-subset valid ; \

Generate valid with beam=5: BLEU = 29.25 62.7/37.3/24.6/16.6 (BP = 0.935 ratio = 0.937 hyp_len = 19379 ref_len = 20682)


test
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt13_fr_en_shared_vocab \
    --path checkpoints/cnn_shared_vocab/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe  \
    --scoring sacrebleu --sacrebleu --results-path result/cnn_shared_vocab.pred ; \

Generate test with beam=5: BLEU = 34.19 63.9/40.5/27.9/19.8 (BP = 0.990 ratio = 0.990 hyp_len = 34759 ref_len = 35120)

```
