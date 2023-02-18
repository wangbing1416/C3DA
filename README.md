# C3DA

Code and datasets of our paper “A Contrastive Cross-Channel Data Augmentation Framework for Aspect-based Sentiment Analysis”



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.



## Generating

To generate data items, run:

`python C3DA/generate.py`



## Training

To train the C3DA model, run:

`sh C3DA/run.sh`

and `C3DA/start.sh`, `C3DA/start1.sh` is used to adjust our hyper-parameters.



## Logs

Logs are saved under `C3DA/C3DA/log`



## Credits

The code and datasets in this repository are based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA).


## Cite

```
@inproceedings{wang2022a,
  author    = {Bing Wang and Liang Ding and Qihuang Zhong and Ximing Li and Dacheng Tao},
  title     = {A Contrastive Cross-Channel Data Augmentation Framework for Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics, {COLING} 2022, Gyeongju, Republic of Korea, October 12-17,
               2022},
  pages     = {6691--6704},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.581},
  timestamp = {Thu, 13 Oct 2022 17:29:38 +0200},
  biburl    = {https://dblp.org/rec/conf/coling/Wang0ZLT22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


