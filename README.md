# Neural Network Intrusion Detection System (DDoS)

## Project Objective
A custom PyTorch model built to detect DDoS attacks in network traffic.

In network security, missing an attack (a False Negative) can crash the server, while accidentally blocking a normal user (a False Positive) is just a minor annoyance. Because of this, the model is specifically optimized for Recall and the F2-Score to ensure no threats slip through.

## Architecture & MLOps Features
This project implements a bare-metal PyTorch pipeline, completely decoupling the training state machine from the production inference script.

* **Neural Network:** Custom multi-layer perceptron utilizing `BCEWithLogitsLoss` for numerical stability.
* **Optimization:** Adam optimizer paired with a dynamic `ReduceLROnPlateau` scheduler to navigate complex loss landscapes.
* **Automated Watchdog (Early Stopping):** Custom training loop interrupt that monitors the validation loss delta, automatically saving the `.pth` checkpoint and killing the training loop to prevent overfitting.
* `train.py`: The compiler. Handles data ingestion, gradient calculation, scheduler stepping, and weight generation.

## System Performance (Model Card)
The engine was trained and evaluated on network traffic data (e.g., CICIDS2017), achieving the following system metrics on the locked test set:

* **Recall (0.9986):** 99.86% of all incoming DDoS attacks successfully intercepted.
* **Precision (0.9995):** 99.95% accuracy when triggering the firewall, ensuring benign user traffic remains undisrupted.
* **F2-Score (0.9988):** A highly balanced metric confirming the model safely favors threat containment without triggering mass false alarms.


##  Quick Start

### 1. Environment Setup
Install the exact system dependencies (CUDA-enabled GPU recommended):
```bash
pip install -r requirements.txt
```
### 2. Compile & Train the Engine
To train the model from scratch, initialize the optimizer, and run the Early Stopping watchdog:

```Bash
python src/train.py
```
This will output the validation metrics per epoch and save the best_ddos_model.pth weights to the data/processed/ directory.

### 3.Run Production Inference
To evaluate new, unseen network traffic without initializing the training loops:

```Bash
python src/predict.py --input path/to/new_traffic.csv
```

