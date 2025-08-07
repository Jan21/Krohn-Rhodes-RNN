# RNN Parity Generalization Experiment

This project tests whether RNNs can generalize to different sequence lengths when trained on a streaming parity task.

## Task Description

The streaming parity task requires the model to predict at each timestep whether there is an even number of ones in the sequence so far:
- Input: Binary sequence (0s and 1s)
- Output: 0 if odd number of ones, 1 if even number of ones

## Model Architecture

- RNN with 100 hidden units and ReLU activation
- Readout head that predicts parity at every timestep
- Trained on sequences of length 10
- Tested on sequences of length 50 (and other lengths)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Evaluate generalization:
```bash
python evaluate.py
```

## Configuration

The project uses Hydra for configuration management. You can modify:
- `conf/model/parity_rnn.yaml`: Model hyperparameters
- `conf/data/parity_data.yaml`: Data generation parameters
- `conf/trainer/default.yaml`: Training parameters
- `conf/logger/wandb.yaml`: Logging configuration

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── rnn_model.py          # RNN model definition
│   └── lightning_module.py   # PyTorch Lightning wrapper
├── data/
│   ├── __init__.py
│   ├── parity_dataset.py     # Dataset for parity sequences
│   └── datamodule.py         # PyTorch Lightning DataModule
├── conf/
│   ├── config.yaml           # Main config
│   ├── model/
│   ├── data/
│   ├── trainer/
│   └── logger/
├── train.py                  # Main training script
├── evaluate.py              # Evaluation script
└── requirements.txt
```

## Expected Results

The experiment will show how well the RNN generalizes from sequences of length 10 to sequences of length 50 and beyond. Perfect generalization would maintain the same accuracy across all lengths.