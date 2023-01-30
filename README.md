## Kaggle Toolbox

Type-aware library for solving Kaggle competitions.

### Key components

1. Training loop, extendable via hooks.
2. Prediction loop.
3. Custom device system.
4. Out of the box adversarial techniques (AWP, FGM) implemented as iteration hooks.
5. Ensembling strategies.
6. Custom loss collection.
7. Custom metric and metric criteria collection.
8. Logging system.
9. Feature generation framework.
10. NLP framework for transformer models.

### Installation

Core version:
```
pip install "git+https://github.com/andrei-papou/kaggle-toolbox.git@v0.3.0#egg=kaggle_toolbox[remote]"
```

The following extras are available besides `remote`:
- `tensorboard` - dependencies for the Tensorboard logger.
- `wandb` - dependencies for the W&B logger.
- `nlp` - NLP dependencies: mainly Huggingface Transformers library.


### Development

Clone the repo.

```
git clone https://github.com/andrei-papou/kaggle-toolbox.git
```

Install all development dependencies.

```
pip install -r requirements/local.txt
```

Run the tests, mypy and flake8.

```
./bin/dev/test.sh && mypy && flake8
```

### Deployment

Make sure `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables are exported before running Kaggle deployment scripts.

Run the following command to create a dataset in your Kaggle account. This should only be done once.

```
./kaggle/bin/init_dataset.sh
```

Then you can deploy a new version of the dataset by running the following command:

```
./kaggle/bin/create_dataset_version.sh "New version"
```
