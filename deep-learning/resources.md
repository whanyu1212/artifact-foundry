# Deep Learning Resources

## Books

### Core Deep Learning Textbooks

- [Deep Learning](https://www.deeplearningbook.org/) - Ian Goodfellow, Yoshua Bengio, Aaron Courville - Comprehensive textbook covering feedforward networks (Ch 6), optimization (Ch 8), CNNs (Ch 9), RNNs (Ch 10), and practical methodology (Ch 11-12)
- [Dive into Deep Learning](https://d2l.ai/) - Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola - Interactive book with code implementations in PyTorch, TensorFlow, and JAX
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen - Accessible introduction focusing on fundamentals and backpropagation
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) - Simon J.D. Prince - Modern textbook (2023) with focus on transformers and attention mechanisms

## Papers

### Foundational Papers

- [Hinton, G. E., & Salakhutdinov, R. R. (2006) "Reducing the Dimensionality of Data with Neural Networks"](https://www.science.org/doi/10.1126/science.1127647) - Seminal paper on deep learning and autoencoders
- [LeCun, Y., Bengio, Y., & Hinton, G. (2015) "Deep Learning"](https://www.nature.com/articles/nature14539) - Nature review article on deep learning by the pioneers
- [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986) "Learning Representations by Back-Propagating Errors"](https://www.nature.com/articles/323533a0) - Original backpropagation paper

### Activation Functions

- [Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015) "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"](https://arxiv.org/abs/1511.07289) - ELU activation function paper
- [Glorot, X., Bordes, A., & Bengio, Y. (2011) "Deep Sparse Rectifier Neural Networks"](http://proceedings.mlr.press/v15/glorot11a.html) - Analysis of ReLU activation and sparse representations
- [Hendrycks, D., & Gimpel, K. (2016) "Gaussian Error Linear Units (GELUs)"](https://arxiv.org/abs/1606.08415) - GELU activation (used in BERT, GPT)
- [Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017) "Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515) - SELU activation for self-normalizing networks
- [Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013) "Rectifier Nonlinearities Improve Neural Network Acoustic Models"](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) - Leaky ReLU and analysis
- [Nair, V., & Hinton, G. E. (2010) "Rectified Linear Units Improve Restricted Boltzmann Machines"](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) - Early ReLU paper
- [Ramachandran, P., Zoph, B., & Le, Q. V. (2017) "Searching for Activation Functions"](https://arxiv.org/abs/1710.05941) - Swish activation discovery via neural architecture search

### Optimization

- [Kingma, D. P., & Ba, J. (2014) "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - Adam optimizer (most popular)
- [Loshchilov, I., & Hutter, F. (2017) "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) - AdamW optimizer
- [Robbins, H., & Monro, S. (1951) "A Stochastic Approximation Method"](https://www.jstor.org/stable/2236626) - Foundational stochastic gradient descent paper
- [Ruder, S. (2016) "An Overview of Gradient Descent Optimization Algorithms"](https://arxiv.org/abs/1609.04747) - Comprehensive survey of optimizers
- [Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013) "On the Importance of Initialization and Momentum in Deep Learning"](http://proceedings.mlr.press/v28/sutskever13.html) - Momentum and initialization

### Regularization

- [Ioffe, S., & Szegedy, C. (2015) "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167) - Batch normalization paper
- [Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/v15/srivastava14a.html) - Dropout regularization

### Initialization

- [Glorot, X., & Bengio, Y. (2010) "Understanding the Difficulty of Training Deep Feedforward Neural Networks"](http://proceedings.mlr.press/v9/glorot10a.html) - Xavier/Glorot initialization
- [He, K., Zhang, X., Ren, S., & Sun, J. (2015) "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"](https://arxiv.org/abs/1502.01852) - He initialization (for ReLU)

### Convolutional Neural Networks (CNNs)

- [He, K., Zhang, X., Ren, S., & Sun, J. (2016) "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) - ResNet with skip connections
- [Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012) "ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) - AlexNet, breakthrough CNN
- [LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998) "Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - LeNet-5, foundational CNN
- [Simonyan, K., & Zisserman, A. (2014) "Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) - VGGNet (16-19 layers)
- [Szegedy, C., Liu, W., Jia, Y., et al. (2015) "Going Deeper with Convolutions"](https://arxiv.org/abs/1409.4842) - GoogLeNet/Inception

### Recurrent Neural Networks (RNNs)

- [Cho, K., Van Merriënboer, B., Gulcehre, C., et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078) - GRU (Gated Recurrent Unit)
- [Hochreiter, S., & Schmidhuber, J. (1997) "Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf) - LSTM architecture

### Transformers and Attention

- [Vaswani, A., Shazeer, N., Parmar, N., et al. (2017) "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Original Transformer architecture
- [Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020) "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) - Vision Transformer (ViT)
- [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018) "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) - BERT model

## Articles & Tutorials

### Deep Learning Fundamentals

- [A Beginner's Guide to Neural Networks](https://towardsdatascience.com/a-beginners-guide-to-neural-networks-d5cf7e369a13) - Towards Data Science - Introduction to neural network basics
- [Backpropagation Algorithm Explained](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd) - Towards Data Science - Step-by-step backpropagation explanation
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) - Chris Olah - Visual explanation of backpropagation
- [How Neural Networks Learn](https://ml4a.github.io/ml4a/how_neural_networks_are_trained/) - ml4a - Interactive guide to training neural networks
- [Neural Networks and Deep Learning](https://www.3blue1brown.com/topics/neural-networks) - 3Blue1Brown - Video series on neural networks with excellent visualizations
- [Understanding Deep Learning](https://www.deeplearningbook.org/) - Deep Learning Book Website - Free online textbook

### Activation Functions

- [Activation Functions Explained](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) - Towards Data Science - Comprehensive guide to activation functions
- [Activation Functions in Neural Networks](https://www.v7labs.com/blog/neural-networks-activation-functions) - V7 Labs - Detailed comparison of activation functions
- [GELU Activation Function](https://paperswithcode.com/method/gelu) - Papers With Code - GELU explanation and papers
- [ReLU and its Variants](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) - Machine Learning Mastery - Complete guide to ReLU family
- [Understanding Activation Functions in Deep Learning](https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/) - Analytics Vidhya - Activation functions with derivatives
- [Visualizing Activation Functions](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/) - David Sheehan - Interactive visualizations

### Optimization

- [Adam Optimizer Explained](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) - Towards Data Science - Deep dive into Adam
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) - Sebastian Ruder - Comprehensive optimizer comparison
- [Gradient Descent, How Neural Networks Learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 3Blue1Brown - Visual explanation of gradient descent
- [Learning Rate Schedules](https://www.deeplearning.ai/ai-notes/optimization/) - DeepLearning.AI - Guide to learning rate strategies
- [Understanding Gradient Descent](https://machinelearningmastery.com/gradient-descent-for-machine-learning/) - Machine Learning Mastery - Practical guide to gradient descent

### Regularization

- [Batch Normalization Explained](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338) - Towards Data Science - Multi-level explanation
- [Dropout in Deep Learning](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) - Machine Learning Mastery - Practical dropout guide
- [L1 and L2 Regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) - Towards Data Science - Regularization techniques
- [Understanding Batch Normalization](https://www.learnopencv.com/batch-normalization-in-deep-networks/) - LearnOpenCV - Detailed batch norm tutorial

### Training Best Practices

- [How to Configure the Learning Rate](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/) - Machine Learning Mastery - Learning rate configuration
- [How to Initialize Neural Network Weights](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) - Machine Learning Mastery - Weight initialization guide
- [Practical Tips for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) - Andrej Karpathy - Expert advice on training
- [Training Neural Networks](https://cs231n.github.io/neural-networks-3/) - Stanford CS231n - Comprehensive training guide

### Architectures

- [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) - Towards Data Science - CNN fundamentals
- [Illustrated Guide to Transformers](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar - Visual transformer explanation
- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah - Understanding LSTM architecture
- [ResNet Explained](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8) - Towards Data Science - ResNet and skip connections
- [Understanding LSTM Networks](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/) - Machine Learning Mastery - LSTM introduction

## Documentation

### PyTorch

#### Core Components

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation
- [torch.nn - Neural Network Modules](https://pytorch.org/docs/stable/nn.html) - Neural network building blocks
- [torch.optim - Optimization Algorithms](https://pytorch.org/docs/stable/optim.html) - Optimizers (SGD, Adam, etc.)
- [Autograd - Automatic Differentiation](https://pytorch.org/docs/stable/autograd.html) - Automatic gradient computation

#### Activation Functions

- [ReLU - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) - ReLU activation module
- [LeakyReLU - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) - Leaky ReLU module
- [ELU - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) - ELU activation module
- [SELU - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html) - SELU activation module
- [GELU - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) - GELU activation module
- [SiLU/Swish - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) - Swish/SiLU activation module
- [Sigmoid - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) - Sigmoid activation module
- [Tanh - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) - Tanh activation module
- [Softmax - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) - Softmax activation module

#### Loss Functions

- [Loss Functions - PyTorch](https://pytorch.org/docs/stable/nn.html#loss-functions) - Overview of loss functions
- [BCELoss - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) - Binary cross-entropy loss
- [CrossEntropyLoss - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - Cross-entropy loss (combines softmax + log loss)
- [MSELoss - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) - Mean squared error loss

#### Optimizers

- [SGD - PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) - Stochastic gradient descent
- [Adam - PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) - Adam optimizer
- [AdamW - PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) - Adam with decoupled weight decay
- [RMSprop - PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) - RMSprop optimizer

#### Regularization

- [Dropout - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - Dropout layer
- [BatchNorm1d - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) - Batch normalization for 1D inputs
- [BatchNorm2d - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Batch normalization for 2D inputs (CNNs)
- [LayerNorm - PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) - Layer normalization

### TensorFlow / Keras

#### Core Components

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) - Official TensorFlow documentation
- [Keras API Reference](https://keras.io/api/) - Official Keras documentation
- [tf.keras.layers - Layer API](https://www.tensorflow.org/api_docs/python/tf/keras/layers) - Neural network layers

#### Activation Functions

- [Activations - Keras](https://keras.io/api/layers/activations/) - Overview of activation functions
- [ReLU - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU) - ReLU activation layer
- [LeakyReLU - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU) - Leaky ReLU layer
- [ELU - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ELU) - ELU activation layer
- [Activation Functions - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/activations) - Functional activation API

#### Loss Functions

- [Losses - Keras](https://keras.io/api/losses/) - Overview of loss functions
- [BinaryCrossentropy - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) - Binary cross-entropy
- [CategoricalCrossentropy - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) - Categorical cross-entropy
- [MeanSquaredError - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError) - MSE loss

#### Optimizers

- [Optimizers - Keras](https://keras.io/api/optimizers/) - Overview of optimizers
- [SGD - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) - SGD with momentum
- [Adam - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) - Adam optimizer
- [AdamW - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/AdamW) - Adam with weight decay

#### Regularization

- [Dropout - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) - Dropout layer
- [BatchNormalization - Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) - Batch normalization layer

### JAX

- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX documentation for high-performance numerical computing
- [Flax - Neural Network Library for JAX](https://flax.readthedocs.io/) - Neural network library built on JAX
- [Optax - Gradient Processing and Optimization for JAX](https://optax.readthedocs.io/) - Optimizer library for JAX

## Courses

### Online Courses

- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Andrew Ng (Coursera) - 5-course specialization covering fundamentals to advanced topics
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) - Stanford - Comprehensive course on CNNs and computer vision
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - Jeremy Howard - Practical, top-down approach to deep learning
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/) - MIT - Introductory deep learning course with lectures on YouTube

## Cheatsheets

- [Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning) - Stanford CS-229 - Comprehensive deep learning cheatsheet
- [Deep Learning Cheatsheet](https://github.com/afshinea/stanford-cs-230-deep-learning) - Stanford CS-230 - Cheatsheets for CNN, RNN, tips and tricks
