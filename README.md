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

# Setup
Requires `python3`, `torch>=1.4.0`

# Running
### Train

The parameters are in the config file. Please ensure all the file paths are appropriate and run

`python train.py`

### Test

Ensure the right model is being loaded and run

`python test.py`

