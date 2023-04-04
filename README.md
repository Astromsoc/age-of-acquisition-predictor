# Age-of-Acquisition Predictor

A fine-tuned BERT model to predict the age of acquisition for words.

## Environmental Setup

Run the following line in local terminal to create a conda env called `aoapred`:

```bash
source setup.sh
```

## Execution

### Dataset Split

Run the following line to split datasets from `data/aoa-data.csv`:

```bash
python3 -m src.train -f data/aoa-data.csv
```

### Model Training

Run the following line to train models with configs specified in `cfg/sample-train-configs.yaml`:

```bash
python3 -m src.train -c cfg/sample-train-configs.yaml
```

### Model Inference

Run the following line to train models with configs specified in `cfg/sample-infer-configs.yaml`:

```bash
python3 -m src.infer -c cfg/sample-infer-configs.yaml
```


## Repo Structure

```bash
.
├── LICENSE
├── README.md
├── cfg
│   ├── sample-infer-configs.yaml
│   └── sample-train-configs.yaml
├── data
│   ├── aoa-data.csv
│   ├── aoapred-test.json
│   ├── aoapred-train.json
│   └── aoapred-val.json
├── requirements.txt
├── setup.sh
└── src
    ├── infer.py
    ├── models.py
    ├── split.py
    ├── train.py
    └── utils.py

3 directories, 15 files
```
