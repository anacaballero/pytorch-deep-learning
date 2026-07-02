# PyTorch Deep Learning

A collection of deep learning experiments in PyTorch covering initialization strategies, regularization techniques, optimization dynamics, convolutional architectures, and transfer learning.

## Notebooks

### Initialization & Regularization
| Notebook | Description |
|----------|-------------|
| [`weight_initialization_relu_benchmark.ipynb`](weight_initialization_relu_benchmark.ipynb) | Benchmarks He (Kaiming), Xavier, Uniform, and default initialization with ReLU activations on MNIST. Quantifies how each scheme affects convergence speed and validation accuracy. |
| [`weight_initialization_tanh_benchmark.ipynb`](weight_initialization_tanh_benchmark.ipynb) | Xavier vs He vs default initialization under Tanh activations. Includes per-layer activation distribution analysis. |
| [`batch_normalization_analysis.ipynb`](batch_normalization_analysis.ipynb) | Comparative study of networks with/without Batch Normalization. Analyzes training stability, convergence speed, and generalization gap. |
| [`logistic_initialization_sensitivity.ipynb`](logistic_initialization_sensitivity.ipynb) | Demonstrates how poor initialization causes gradient saturation in Sigmoid-activated networks. |

### Optimization Dynamics
| Notebook | Description |
|----------|-------------|
| [`momentum_optimization_analysis.ipynb`](momentum_optimization_analysis.ipynb) | Compares SGD with momentum against standard gradient descent. Visualizes loss surface trajectories and convergence under varying learning rates. |
| [`minibatch_gradient_descent_analysis.ipynb`](minibatch_gradient_descent_analysis.ipynb) | Analyzes how batch size interacts with learning rate stability. Loss surface visualization across full-batch, mini-batch, and stochastic configurations. |

### Network Architectures
| Notebook | Description |
|----------|-------------|
| [`nonlinear_boundary_capacity_analysis.ipynb`](nonlinear_boundary_capacity_analysis.ipynb) | Empirical capacity analysis: minimum hidden neurons needed to separate noisy XOR data. Tests 1, 2, and 3 neurons systematically. |
| [`hidden_width_approximation.ipynb`](hidden_width_approximation.ipynb) | Effect of hidden layer width on function approximation power. Compares decision boundaries across neuron counts. |
| [`deep_network_modulelist.ipynb`](deep_network_modulelist.ipynb) | Constructs arbitrarily deep networks using `nn.ModuleList`. Enables dynamic architecture configuration without hardcoded layer definitions. |
| [`activation_function_comparison.ipynb`](activation_function_comparison.ipynb) | Empirical comparison of Sigmoid, Tanh, and ReLU. Analyzes gradient saturation, convergence speed, and final accuracy. |
| [`activation_maxpooling_cnn.ipynb`](activation_maxpooling_cnn.ipynb) | Activation functions combined with MaxPooling in a convolutional pipeline. Analyzes spatial downsampling effects on feature retention and gradient flow. |

### Softmax & Classification
| Notebook | Description |
|----------|-------------|
| [`softmax_regression_pytorch.ipynb`](softmax_regression_pytorch.ipynb) | Softmax regression from first principles: traces how linear projections map to class probabilities. |
| [`softmax_multiclass_classifier.ipynb`](softmax_multiclass_classifier.ipynb) | Single-layer Softmax classifier on MNIST. Cross-entropy loss, probability calibration, per-class accuracy analysis. |

### Convolutional Networks & Transfer Learning
| Notebook | Description |
|----------|-------------|
| [`convolutional_feature_maps_analysis.ipynb`](convolutional_feature_maps_analysis.ipynb) | Multi-channel convolutional layers and learned feature map analysis. Explores kernel depth, stride, and padding interactions with MaxPooling. |
| [`transfer_learning_image_classifier.ipynb`](transfer_learning_image_classifier.ipynb) | VGG16 and ResNet50 (ImageNet weights) applied to Fashion MNIST with limited per-class samples. Freezes convolutional layers, replaces classifier head, uses `DataParallel` for multi-GPU training. |

### Data Pipelines
| Notebook | Description |
|----------|-------------|
| [`pytorch_tensor_operations.ipynb`](pytorch_tensor_operations.ipynb) | Tensor construction, shape manipulation, broadcasting, and autograd. Covers in-place operations and the computational graph. |
| [`image_dataset_transforms_pipeline.ipynb`](image_dataset_transforms_pipeline.ipynb) | Full image augmentation pipeline with torchvision: normalization, random cropping, flipping, and custom transforms. |
| [`custom_dataset_dataloader.ipynb`](custom_dataset_dataloader.ipynb) | Custom `Dataset` class with `__len__` and `__getitem__`, integrated with `DataLoader` for batching, shuffling, and on-the-fly transforms. |
| [`demand_forecasting_numpy_nn.ipynb`](demand_forecasting_numpy_nn.ipynb) | Feedforward neural network built from scratch in NumPy (no frameworks) for demand forecasting. Manual forward pass, backpropagation, and gradient descent. |

## Stack
- PyTorch, torchvision
- NumPy, Matplotlib
- MNIST, Fashion MNIST datasets
