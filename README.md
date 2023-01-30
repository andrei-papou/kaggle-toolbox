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
