---
tags:
- generated_from_trainer
model-index:
- name: falcon40b-mini-shakespeare
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# falcon40b-mini-shakespeare

This model was trained from scratch on "tinyshakespeare" text file.

## Model description

The configuration and code is adapted from tiiuae/falcon-40b, with configuration parameters changed to make it a very tiny model.

- **License:** Apache 2.0.

## Intended uses & limitations

Intended just to aid debugging efforts of a GGML port of Falcon.

## Training and evaluation data

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Training procedure

Just used the single tinyshakespeare text file as both the training and validation set (split up into paragraphs). See:

https://colab.research.google.com/drive/1dl7Jko78CX1y_EYZJWcRW9b_7X-0Bst8?usp=sharing

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 256
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 10

### Framework versions

- Transformers 4.28.0
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3
