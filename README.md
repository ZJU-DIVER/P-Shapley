# P-Shapley

Code for implementation of "P-Shapley: Shapley value on probabilistic classification".

## Motivation

Simple accuracy is not sufficient for evaluting the performance of a classifier.


### Prerequisites

- Python, NumPy, Scikit-learn, PyTorch

### Datasets

- Covertype
- Wind
- Fashion-MNIST
- CIFAR-10

The preprocessing procedure of the above dataset is mentioned in Section 5.1 in the original paper.

## Experiment

```
.
├── data_preprocess  # Extract features from image datasets
├── case_study       # Case study for Section 3.3
└── experiment       # Experiments for Section 5
    ├── dataeval               # Algorithms for P-Shapley, baselines, and other required utils.
    ├── computation_stability  # Computation stability experiment for Section 5.3
    ├── data_removal           # Data removal experiment for Section 5.4
    └── noise_detection        # Noise detection experiment for Section 5.5
```

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
