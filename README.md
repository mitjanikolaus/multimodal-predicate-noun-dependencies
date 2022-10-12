# Multimodal Predicate-Noun Dependencies



## Installation

- `git clone --recursive <repo>`

## Data

All images with image ids occurring in [data/sentence-semantics/eval_set.json](eval_set.json) have to be downloaded to
~/data/multimodal_evaluation/images .

### Python Environments:

#### vl-eval-pytorch-1.8.1
For LXMERT, UNITER, ViLT
```
conda env create --file environment_vl-eval-pytorch-1.8.1.yml 
cd src ViLT && pip install . && cd -
```

#### vl-eval-vilbert

```
conda env create --file environment_vl-eval-vilbert
conda activate vl-eval-vilbert
cd src/vilbert-multi-task && python setup.py build develop && cd -
```

#### vl-eval-vinvl

```
conda env create --file environment_vl-eval-vinvl
conda activate vl-eval-vinvl
cd src/Oscar && python setup.py build develop && cd -
python -m pip install git+https://github.com/facebookresearch/maskrcnn-benchmark.git
```

#### clip

```
conda env create --file environment_vl-eval-clip.yml
conda activate vl-eval-clip
pip install git+https://github.com/openai/CLIP.git
```

#### volta

For models trained in controlled conditions ([VOLTA framework](https://github.com/e-bug/volta))

```
conda env create --file environment_vl-eval-volta.yml
conda activate vl-eval-volta
cd src/volta && python setup.py develop && cd -
```

## Models

### LXMERT

- Environment: vl-eval-pytorch-1.8.1
- Checkpoint: unc-nlp/lxmert-base-uncased
- Image features: Bottom-up (36 boxes)

```
conda activate vl-eval-pytorch-1.8.1
python eval_sentence_semantics.py --model LXMERT --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_features_2048.tsv
python eval_sentence_semantics.py --model LXMERT --cropped --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_cropped_features_2048.tsv
```

### UNITER

- Environment: vl-eval-pytorch-1.8.1
- Checkpoint: https://github.com/ChenRocks/UNITER/blob/master/scripts/download_pretrained.sh
- Image features: Bottom-up (36 boxes)

```
conda activate vl-eval-pytorch-1.8.1
python eval_sentence_semantics.py --model UNITER --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_features_2048.tsv
python eval_sentence_semantics.py --model UNITER --cropped --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_cropped_features_2048.tsv
```

### ViLT

- Environment: vl-eval-pytorch-1.8.1
- Checkpoint: https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt
- Image features: extracted within model

```
conda activate vl-eval-pytorch-1.8.1
python eval_sentence_semantics.py --model VILT --images-dir ~/data/multimodal_evaluation/images
python eval_sentence_semantics.py --model VILT --cropped --images-dir ~/data/multimodal_evaluation/images_cropped
```

### Oscar

- Environment: vl-eval-vinvl
- Checkpoint: https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/base/checkpoint.zip
- Image features: Bottom-up (10 to 100 boxes)

```
conda activate vl-eval-vinvl
python eval_sentence_semantics.py --model Oscar --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_features_2048_10_100.tsv
python eval_sentence_semantics.py --model Oscar --cropped --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_cropped_features_2048_10_100.tsv
```

### VinVL

- Environment: vl-eval-vinvl
- Checkpoint: https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md
- Image features: extracted using Vision Transformer (ViT)

```
conda activate vl-eval-vinvl
python eval_sentence_semantics.py --model VINVL --img-features-path ~/data/multimodal_evaluation/image_features_vinvl/
python eval_sentence_semantics.py --model VINVL --cropped --img-features-path ~/data/multimodal_evaluation/image_features_vinvl_cropped/
```

### ViLBERT

- Environment: vl-eval-vilbert
- Checkpoint: https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin
- Image features: from faster r-cnn (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)


```
conda activate vl-eval-vilbert
python eval_sentence_semantics.py --model VILBERT --img-features-path ~/data/multimodal_evaluation/image_features_vilbert/img_features.p
python eval_sentence_semantics.py --model VILBERT --cropped --img-features-path ~/data/multimodal_evaluation/image_features_vilbert/img_cropped_features.p
```

### CLIP

- Environment: vl-eval-clip
- Checkpoint: ViT-B/32
- Image features: extracted within model

```
conda activate vl-eval-clip
python eval_sentence_semantics.py --model CLIP --images-dir ~/data/multimodal_evaluation/images
python eval_sentence_semantics.py --model CLIP --cropped --images-dir ~/data/multimodal_evaluation/images_cropped
```

### VOLTA

- Environment: vl-eval-volta
- Image features: Bottom-up (36 boxes)

Example eval for VisualBERT:
```
conda activate vl-eval-volta
python eval_sentence_semantics_volta.py --from_pretrained ~/data/volta/VisualBERT --config_file src/volta/config/ctrl_visualbert_base.json --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_features_2048.tsv
python eval_sentence_semantics_volta.py --from_pretrained ~/data/volta/VisualBERT --config_file src/volta/config/ctrl_visualbert_base.json --cropped --img-features-path ~/data/multimodal_evaluation/image_features_2048/img_cropped_features_2048.tsv
```

## Analyses

Analysis scripts can be run after model results have been saved to `runs/sentence_semantics`.

### Detailed results

Generate per-concept results plots (and more):
```
python plot_sentence_semantics_results.py --input-file runs/sentence-semantics/LXMERT/results.csv
```

### Correlations
(First, download [Train_GCC-training.tsv](https://ai.google.com/research/ConceptualCaptions/download)
and safe it to data/conceptual_captions/.)

Run correlations between common predictors and model performance: 
```
python sentence_semantics_correlations.py --models LXMERT UNITER VILBERT VILT VINVL CLIP
```
