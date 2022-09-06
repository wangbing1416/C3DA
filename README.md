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


## Citation

If our method, data, and code help you, please considering cite our work:

```bibtex
@inproceedings{wang2022c3da,
    title = {A Contrastive Cross-Channel Data Augmentation Framework for Aspect-Based Sentiment Analysis},
    author = {Bing Wang, Liang Ding, Qihuang Zhong, Ximing Li, Dacheng Tao},
    booktitle = {COLING},
    year = {2022}
}
```




