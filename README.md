# DySPred [![DOI](https://zenodo.org/badge/736617244.svg)](https://doi.org/10.5281/zenodo.14058681)
Code for "Leveraging real-world pharmacovigilance data with deep learning to predict population-scale toxicity landscape of immune checkpoint inhibitors"
![](https://github.com/ZhoulabCPH/DySPred/edit/master/Workflow.png)

# Requirements
`pip install -r requirements.txt`

# Example for disproportionality analyses
``` python
# ./main_demo.py
args_preprocessing = {'original_data_path': './Data_ref/',
                      'out_data_path': './Data/',
                      'label_folder': 'split_year',
                      'graph_type': 'Directed',
                      'generate_core': True
                      }
preprocessing(args_preprocessing)
```
``` python
## Input: Raw format for drug-AEs 
Drug	Reac	a	drug_num	reac_num	n
ABALOPARATIDE	ABDOMINAL DISCOMFORT	39	1930	9332	701394
ABATACEPT	ABDOMINAL DISCOMFORT	12	1934	9332	701394
ABEMACICLIB	ABDOMINAL DISCOMFORT	8	437	9332	701394
ABIRATERONE	ABDOMINAL DISCOMFORT	8	1498	9332	701394
```
``` python
## Output: Disproportionality format for drug-AEs 
Drug	Reac	signal	ROR	EBGM	N	ROR_025	EBGM_025	b	c	d
ABALOPARATIDE	ABDOMINAL DISCOMFORT	3	2.313830868	1.94	39	1.684364199	1.41	1894	13562	1523950
ABATACEPT	ABDOMINAL DISCOMFORT	1	0.924888543	0.9	59	0.715438044	0.69	7154	13542	1518690
ABEMACICLIB	ABDOMINAL DISCOMFORT	1	1.337541854	1	9	0.693087343	0.53	755	13592	1525089
ABIRATERONE	ABDOMINAL DISCOMFORT	1	0.593398787	0.58	21	0.38633161	0.38	3966	13580	1521878
ACETAMINOPHEN	ABDOMINAL DISCOMFORT	1	0.270034145	0.29	11	0.149400835	0.16	4560	13590	1521284
```

# DySPred Training and Outputs
``` python
# ./main_demo.py
args_model = {'base_path': './Data',
              'origin_path': 'all',
              'core_folder': 'Cores',
              'label_folder': 'split_year',
              'learn_type': 'regression',
              'model_type': ['DySPred'],
              'dic_level': 'PT',
              'duration': 5,
              'train': True,
              'seed': 20,
              'embed_dim': 128,
              'use_cuda': True,
              'min_epoch': 1000,
              'max_epoch': 10000,
              'patient': 10,
              'lr': 0.001,
              'export': True,
              'min_N': 10,
              'trans_layer_num': 1,
              'diffusion_layer_num': 1,
              'rnn_type': 'LSTM',
              'trans_activate_type': 'N',
              'bias': True,
              'weight_decay': 1e-3,
              'neg_num': 20,
              'cls_file': "cls_01",
              'cls_bias': True,
              'cls_layer_num': 2,
              'cls_output_dim': 5,
              'cls_activate_type': 'N',
              'train_ratio': 0.8,
              'val_ratio': 0.1,
              'test_ratio': 0.1,
              'test_stamp_idx': [0, -1],
              'has_cuda': True if torch.cuda.is_available() else False
              }
model(args_model)
```
``` python
## Output: predicted risk score format for drug-AEs 
Drug	Reac	Score
PD-1/PD-L1_MEL	ABDOMINAL ABSCESS	3.082800865
PD-1/PD-L1_MEL	ABDOMINAL ADHESIONS	3.825821877
PD-1/PD-L1_MEL	ABDOMINAL DISCOMFORT	1.241694212
PD-1/PD-L1_MEL	ABDOMINAL DISTENSION	0.710710168
PD-1/PD-L1_MEL	ABDOMINAL HERNIA	3.600068092
PD-1/PD-L1_MEL	ABDOMINAL INFECTION	3.206662893
PD-1/PD-L1_MEL	ABDOMINAL PAIN	0.927628994
``` 
