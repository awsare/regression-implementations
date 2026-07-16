# Regression Implementations

Implementations of linear, logistic, and softmax regression from scratch in Python using NumPy. This project demonstrates:

- Batch/stochastic gradient descent
- Multiclass classification
- Cross-entropy loss
- Forward/backward pass structure

## Setup

Requires Python 3 with `numpy` and `pytest`.

```bash
pip install numpy pytest
```

## Run Tests

Run a single file's tests:

```bash
pytest -v test_linear_regression.py
pytest -v test_logistic_regression.py
pytest -v test_softmax_regression.py
```

Or run everything at once:

```bash
pytest -v
```