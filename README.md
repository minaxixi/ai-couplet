# AI-couplet-writer

Welcome! This is our fun project of using AI to write couplets (对对联). Please check out our website!

https://ai-poet.com

## Introduction
We are a Chinese couple living in California. We love traditional Chinese poems, and we had this idea of training an AI model to write couplets, an easier version of poems per se.

There are several projects on AI couplet writing (see references below), but this project is the first of its kind written in TensorFlow 2.x, to the best of our knowledge.

## Model
This is a sequence-to-sequence model with Encoder + Decoder + Attention, schematically shown as below.

![The AI Couplet Model](/doc/schematics.png)

Notable features
- Embedding layer pretrained via Word2Vec
- Beam search decoder
- Format polishing to prevent repeated characters, etc.

## Data
The training/test data are around 70k couplets from [this](https://github.com/wb14123/couplet-dataset) repo

## How to Run
Set up the config at the **baseconfig.ini**, and run as **python train.py**
Our model on the website was trained on Google Colab Pro for 12 epochs (~1 day)

Package pre-requisites
- TensorFlow 2.2.0
- Python 3.6.9
- Numpy 1.18.5
- Gensim 3.6.0

## Reference
- Seq2seq-couplet by Bin Wang ([Github](https://github.com/wb14123/seq2seq-couplet), [Demo website](https://ai.binwang.me/couplet))
- Seq2seq chinese poetry generation by Simon and Vera ([Github](https://github.com/Disiok/poetry-seq2seq), [Related Paper](https://arxiv.org/abs/1610.09889))
- Microsoft Research/微软亚洲研究院电脑对联系统 ([Website](https://duilian.msra.cn/app/couplet.aspx))
- Open Couplet/中文对联AI ([Github](https://github.com/neoql/open_couplet), [Website](https://couplet.neoql.me/))
- AICHPOEM.COM/诗三百-人工智能诗歌平台 ([Website](https://www.aichpoem.com/#/shisanbai/poem))
- 清华九歌-人工智能诗歌写作系统 ([Website](http://jiuge.thunlp.org/jueju.html))
