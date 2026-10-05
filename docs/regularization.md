# RawGrad Regularization

## 1. Overview

Regularization is a technique used to reduce overfitting by adding an
additional penalty to the training objective.

RawGrad implements L2 regularization for trainable Dense layer weights.

The regularized training objective is:

```text
Total Objective
      =
Data Loss
      +
L2 Penalty