# Thesis Code: CRISPR Off-Target Cleavage Activity Prediction

## Overview

This repository contains the code associated with a thesis on computational prediction of CRISPR/Cas off-target cleavage activity.

The repository consolidates three model families investigated in the thesis:

- CRISPR-DBA;
- CRISPR-ParaXNet;
- CRISPR-DE.

## Repository Structure

```text
thesis/
├── DBA/
│   ├── PNN model/
│   │   ├── FNN5_PNN.py
│   │   ├── CNN5_PNN.py
│   │   ├── CrisprNet_PNN.py
│   │   ├── CrisprIP_PNN.py
│   │   └── HW_PNN.py
│   └── data/
│
├── ParaXNet/
│   ├── models/
│   │   └── CRISPR-ParaXNet.py
│   └── data/
│
└── DE/
    ├── models/
    │   └── CRISPR-DE.py
    └── data/
```

## Model Implementations

### CRISPR-DBA

CRISPR-DBA applies Monte Carlo dropout to established neural-network architectures for off-target cleavage activity prediction.

The repository includes probabilistic neural-network implementations based on five backbone architectures:

- FNN;
- CNN;
- CRISPR-Net;
- CRISPR-IP;
- CRISPR-HW.

Each implementation performs stochastic forward passes with dropout enabled to estimate a distribution of predicted cleavage activities.

### CRISPR-ParaXNet

CRISPR-ParaXNet combines a multilayer perceptron branch with a three-layer Deep & Cross Network branch.

The two feature representations are concatenated before the final regression layers. The cross layers are implemented using TensorFlow Recommenders.

### CRISPR-DE

CRISPR-DE combines predictions from four neural-network subnetworks:

- multilayer perceptron;
- convolutional neural network;
- bidirectional LSTM;
- bidirectional LSTM with self-attention.

The implementation also includes equal-weight subnetwork ablation analyses and an additional noisy-MLP configuration.

## Usage

The following pseudocode summarises the general execution procedure.

### CRISPR-DBA

```text
BEGIN

CHANGE_DIRECTORY_TO <DBA data directory>

SELECT <a compatible encoded data package>

SELECT <one PNN model script>

EXECUTE_WITH_PYTHON <selected PNN model script>

END
```

The selected data package must have the orientation required by the corresponding model.

### CRISPR-ParaXNet

```text
BEGIN

CHANGE_DIRECTORY_TO <repository or ParaXNet directory>

SELECT <the CRISPR sequence dataset>

SELECT <the ParaXNet model script>

EXECUTE_WITH_PYTHON <selected ParaXNet model script>

END
```

### CRISPR-DE

```text
BEGIN

CHANGE_DIRECTORY_TO <repository or DE directory>

SELECT <the CRISPR sequence dataset>

SELECT <the CRISPR-DE model script>

EXECUTE_WITH_PYTHON <selected CRISPR-DE model script>

END
```

## Software Environment

The current code was tested with the following software environment:

- Python 3.8.18
- TensorFlow 2.13.0
- Keras 2.13.1
- TensorFlow Recommenders 0.7.3
- NumPy 1.24.3
- pandas 2.0.3
- SciPy 1.10.1
- scikit-learn 1.1.3
- Matplotlib 3.7.2

## License

This repository is distributed under the Apache License 2.0.
