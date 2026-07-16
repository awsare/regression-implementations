# Regression Implementations

Implementations of linear, logistic, and softmax regression from scratch in Python using NumPy. This project demonstrates machine learning concepts including gradient descent optimization, multiclass classification, and forward/backward pass structure.

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

## Concepts

The three files build on each other conceptually:
1. **Linear regression** introduces the design matrix and vectorized batch gradient descent.
2. **Logistic regression** introduces nonlinear activation, cross-entropy loss, and explicit chain-rule backpropagation with per-sample SGD.
3. **Softmax regression** generalizes logistic regression from binary to multi-class, where each output class's activation depends on every logit, so the activation gradient is a full matrix of cross-terms instead of a single scalar.