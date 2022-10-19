# AENLP
This repository contains the implementation of inference models from "Extraction of Comprehensive Drug Safety Information from Adverse Drug Event Narratives in Spontaneous Reporting System"

# Requirements and Installation
``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout dd7499
pip install .
```

# Getting Started
## Pre-trained models
We provide our pre-trained KAERS-BERT model checkpoint along with fine-tuned checkpoints for downstream tasks

It contains following folders:
* KAERS-BERT: The pre-trained KAERS-BERT model checkpoint
* ner-KEARS-BERT: The fine-tuned checkpoint for name entity recognition task
* rel-KEARS-BERT: The fine-tuned checkpoint for relation extraction task
* lp-KEARS-BERT: The fine-tuned checkpoint for labe prediction task

# License
KAERS-BERT is MIT-licensed.
The license applies to the pre-trained models as well.
