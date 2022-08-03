# RECON
This is the code for the paper ['RECON: Relation Extraction using Knowledge Graph Context in a Graph Neural Network'](https://arxiv.org/abs/2009.08694).

Please note, we recieved multiple queries regarding why we have not used BERT as context aggregator instead of GNN. We tried it and performance was quite low (in the range of 0.76 F1Score on Wikidata dataset). Hence we decided to ignore BERT.

# Citation
If you use our work kindly consider citing

```
@inproceedings{bastos2020recon,
  title={RECON: Relation Extraction using Knowledge Graph Context in a Graph Neural Network},
  author={Bastos, Anson and Nadgeri, Abhishek and Singh, Kuldeep and Mulang, Isaiah Onando and Shekarpour, Saeedeh and Hoffart, Johannes and Kaul, Manohar},
  booktitle={Proceedings of The Web Conference (WWW) (long papers)},
  pages={:N/A},
  year={2021}
}
```

# Dataset
The datasets can be downloaded from the below link

[Wikidata](https://drive.google.com/file/d/1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU/view?usp=sharing)

[NYT](https://drive.google.com/file/d/1VhY3i3SoLS3XZvtsk_BmpZgN5cjITlij/view?usp=sharing)

Glove embedding can be obtained from below link:

[glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)

# Setup
Requires `python3`, `torch>=1.4.0`

# Running
### Train

The parameters are in the config file. 
Download the datasets, embeddings etc. and place in the expected locations.

> mkdir data \
> cd data \
> unzip WikipediaWikidataDistantSupervisionAnnotations.v1.0.zip
> unzip glove.6B.zip \
> cp glove.6B/glove.6B.50d.txt WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/


In case nltk modules are not downloaded please run the below commands in a python interpreter:
> import nltk \
> nltk.download('punkt')

If there are any relations to not be trained place them in `../resources/property_blacklist.txt` else ignore the error of this file not being present.

Please ensure all the file paths are appropriate and run below commands for training the desired configurations of RECON

RECON_EAC
set the 
`python train.py`



### Test

Ensure the right model is being loaded(model/file name needs to be edited in code) and run

`python test.py -sf=<path to the model folder>`


# Acknowledgements
The code has been adapted from [ContextAware](https://github.com/UKPLab/emnlp2017-relation-extraction) and [GPGNN](https://github.com/thunlp/GP-GNN) repositories. Many thanks to the authors for sharing the code.