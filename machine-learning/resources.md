# Machine Learning Resources

## Cheatsheets

- [Ensemble Methods in Machine Learning: Random Forests Cheatsheet](https://www.codecademy.com/learn/mle-ensembling/modules/mle-random-forests/cheatsheet) - Codecademy - Quick reference for Random Forest concepts and implementation
- [Stanford CS-229 Machine Learning Tips and Tricks](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks) - Shervine Amidi - Comprehensive guide to classification/regression metrics, model selection, cross-validation, regularization, and bias-variance tradeoff
- [Stanford CS-229 Supervised Learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning) - Shervine Amidi - Reference for supervised learning algorithms including linear models, SVM, generative learning, and ensemble methods
- [Stanford CS-229 Unsupervised Learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning) - Shervine Amidi - Guide to clustering (K-means, EM, hierarchical) and dimensionality reduction (PCA, ICA)

## Books

### Core Machine Learning Textbooks

- [An Introduction to Statistical Learning](https://www.statlearning.com/) - James, Witten, Hastie, Tibshirani - Accessible introduction to statistical learning with R examples; essential chapters on linear regression (Ch 3), logistic regression and GLMs (Ch 4), regularization methods (Ch 6), cross-validation (Ch 5), and tree-based methods (Ch 8)
- [The Elements of Statistical Learning](https://hastie.su.stanford.edu/ElemStatLearn/) - Hastie, Tibshirani, Friedman - Comprehensive mathematical treatment of statistical learning; authoritative reference for linear methods (Ch 3), basis expansions (Ch 5), model assessment (Ch 7), trees and ensemble methods (Ch 9-10, 15, 16)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) - Aurélien Géron - Practical Python implementations using scikit-learn; excellent for ensemble methods, decision trees, and hyperparameter tuning
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Christopher Bishop - Bayesian perspective on machine learning with strong theoretical foundations; comprehensive coverage of model selection and evaluation

## Papers

### Linear Models and Regularization

- [Hoerl & Kennard (1970) "Ridge Regression: Biased Estimation for Nonorthogonal Problems"](https://www.jstor.org/stable/1267351) - Original Ridge regression paper introducing L2 regularization
- [Tibshirani (1996) "Regression Shrinkage and Selection via the Lasso"](https://www.jstor.org/stable/2346178) - Seminal Lasso paper introducing L1 regularization for sparse solutions
- [Zou & Hastie (2005) "Regularization and Variable Selection via the Elastic Net"](https://hastie.su.stanford.edu/Papers/B67.2%20%282005%29%20301-320%20Zou%20&%20Hastie.pdf) - Elastic Net paper combining L1 and L2 penalties for grouped variable selection
- [Nelder & Wedderburn (1972) "Generalized Linear Models"](https://www.jstor.org/stable/2344614) - Foundational GLM paper establishing the framework for exponential family distributions

### Loss Functions and Optimization

- [Huber, P. J. (1964) "Robust Estimation of a Location Parameter"](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) - Original Huber loss paper introducing robust M-estimation
- [Koenker, R., & Bassett, G. (1978) "Regression Quantiles"](https://www.jstor.org/stable/1913643) - Foundational paper on quantile regression and pinball loss
- [Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017) "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) - Focal loss addressing class imbalance in object detection (RetinaNet)
- [Vapnik, V. N. (1995) "The Nature of Statistical Learning Theory"](https://link.springer.com/book/10.1007/978-1-4757-2440-0) - Comprehensive theory of loss functions and risk minimization

### Probabilistic Models

- [Fisher, R. A. (1936) "The Use of Multiple Measurements in Taxonomic Problems"](https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x) - Original LDA paper introducing linear discriminant analysis
- [Friedman, J. H. (1989) "Regularized Discriminant Analysis"](https://www.jstor.org/stable/2289860) - RDA paper introducing regularization for discriminant analysis

### Instance-Based Learning (K-Nearest Neighbors)

- [Cover, T. M., & Hart, P. E. (1967) "Nearest Neighbor Pattern Classification"](https://ieeexplore.ieee.org/document/1053964) - Foundational paper on KNN algorithm and error bounds
- [Fix, E., & Hodges, J. L. (1951) "Discriminatory Analysis: Nonparametric Discrimination"](https://apps.dtic.mil/sti/citations/ADA800276) - Early work on nearest neighbor classification

### Support Vector Machines

- [Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992) "A Training Algorithm for Optimal Margin Classifiers"](https://dl.acm.org/doi/10.1145/130385.130401) - Original SVM paper introducing optimal margin classifier
- [Cortes, C., & Vapnik, V. (1995) "Support-Vector Networks"](https://link.springer.com/article/10.1007/BF00994018) - Seminal SVM paper introducing soft margin and kernel methods
- [Platt, J. (1998) "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf) - SMO algorithm for efficient SVM training
- [Schölkopf, B., & Smola, A. J. (2002) "Learning with Kernels"](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) - Comprehensive book on kernel methods and SVM theory

### Evaluation Metrics

- [Davis, J., & Goadrich, M. (2006) "The Relationship Between Precision-Recall and ROC Curves"](https://www.biostat.wisc.edu/~page/rocpr.pdf) - Analysis of when to use PR curves vs ROC curves, especially for imbalanced data
- [Hand, D. J., & Till, R. J. (2001) "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems"](https://link.springer.com/article/10.1023/A:1010920819831) - Extension of AUC to multi-class classification
- [Matthews, B. W. (1975) "Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme"](https://www.sciencedirect.com/science/article/abs/pii/0005279575901099) - Original paper introducing Matthews Correlation Coefficient
- [Powers, D. M. W. (2011) "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation"](https://arxiv.org/abs/2010.16061) - Comprehensive analysis of classification metrics relationships
- [Willmott, C. J., & Matsuura, K. (2005) "Advantages of the Mean Absolute Error (MAE) over the Root Mean Square Error (RMSE)"](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.1183) - Discussion of MAE vs RMSE for model evaluation

### Unsupervised Learning (Clustering and Dimensionality Reduction)

- [Arthur, D., & Vassilvitskii, S. (2007) "K-Means++: The Advantages of Careful Seeding"](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) - K-Means++ initialization improving convergence and results
- [Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977) "Maximum Likelihood from Incomplete Data via the EM Algorithm"](https://www.jstor.org/stable/2984875) - Foundational EM algorithm paper for Gaussian Mixture Models
- [Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996) "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - Original DBSCAN paper
- [Hotelling, H. (1933) "Analysis of a Complex of Statistical Variables into Principal Components"](https://psycnet.apa.org/record/1934-00645-001) - Early work on Principal Component Analysis
- [Jolliffe, I. T. (2002) "Principal Component Analysis"](https://link.springer.com/book/10.1007/b98835) - Comprehensive book on PCA theory and applications
- [Lloyd, S. P. (1982) "Least Squares Quantization in PCM"](https://ieeexplore.ieee.org/document/1056489) - Original K-Means algorithm (Lloyd's algorithm)
- [MacQueen, J. (1967) "Some Methods for Classification and Analysis of Multivariate Observations"](https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/Some-methods-for-classification-and-analysis-of-multivariate-observations/bsmsp/1200512992) - K-Means clustering method
- [Pearson, K. (1901) "On Lines and Planes of Closest Fit to Systems of Points in Space"](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) - Original PCA paper
- [Rousseeuw, P. J. (1987) "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis"](https://www.sciencedirect.com/science/article/pii/0377042787901257) - Silhouette method for cluster validation
- [van der Maaten, L., & Hinton, G. (2008) "Visualizing Data using t-SNE"](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) - t-SNE for high-dimensional data visualization

### Tree-Based Methods and Ensembles

- [Breiman, L. (1996) "Bagging Predictors"](https://link.springer.com/article/10.1023/A:1018054314350) - Original paper introducing Bootstrap Aggregating (bagging) ensemble method
- [Breiman, L. (2001) "Random Forests"](https://link.springer.com/article/10.1023/A:1010933404324) - Seminal paper introducing Random Forest algorithm with random feature selection
- [Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System"](https://arxiv.org/abs/1603.02754) - XGBoost paper introducing second-order optimization and regularization
- [Freund & Schapire (1997) "A Decision-Theoretic Generalization of On-Line Learning"](https://cseweb.ucsd.edu/~yfreund/papers/adaboost.pdf) - Original AdaBoost paper
- [Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full) - Foundational gradient boosting paper

## Courses

## Articles & Tutorials

### Linear Regression

- [A Complete Guide to Linear Regression](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86) - Towards Data Science - Comprehensive explanation of linear regression with mathematical foundations
- [Introduction to Linear Regression](https://machinelearningmastery.com/linear-regression-for-machine-learning/) - Machine Learning Mastery - Practical guide to linear regression with Python examples
- [Linear Regression Explained](https://mlu-explain.github.io/linear-regression/) - MLU-Explain - Interactive visualization of linear regression concepts
- [Simple and Multiple Linear Regression in Python](https://realpython.com/linear-regression-in-python/) - Real Python - Comprehensive Python tutorial covering statsmodels, scikit-learn, and manual implementation

### Regularization (Ridge, Lasso, Elastic Net)

- [L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) - Towards Data Science - Clear explanation of L1 vs L2 regularization
- [Lasso and Ridge Regression Explained](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression) - DataCamp - Practical tutorial on regularization techniques
- [Regularization in Machine Learning](https://www.geeksforgeeks.org/regularization-in-machine-learning/) - GeeksforGeeks - Overview of regularization methods with code examples
- [Ridge and Lasso Regression: A Complete Guide with Python Scikit-Learn](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b) - Towards Data Science - Practical implementation guide
- [Understanding the Bias-Variance Tradeoff and Visualizing it with Example and Python Code](https://www.kdnuggets.com/2020/09/understanding-bias-variance-tradeoff.html) - KDnuggets - Visualization of bias-variance tradeoff with regularization

### Logistic Regression and GLMs

- [Logistic Regression Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) - Towards Data Science - Mathematical foundation and implementation
- [Logistic Regression for Machine Learning](https://machinelearningmastery.com/logistic-regression-for-machine-learning/) - Machine Learning Mastery - Comprehensive guide to logistic regression
- [Understanding Logistic Regression](https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/) - Analytics Vidhya - Beginner-friendly explanation
- [Generalized Linear Models](https://www.statsmodels.org/stable/glm.html) - Statsmodels Documentation - Technical guide to GLMs with Python

### Probabilistic Models (Naive Bayes, LDA/QDA)

- [6 Easy Steps to Learn Naive Bayes Algorithm](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/) - Analytics Vidhya - Step-by-step guide to Naive Bayes with examples
- [Linear Discriminant Analysis Explained](https://towardsdatascience.com/linear-discriminant-analysis-explained-f88be6c1e00b) - Towards Data Science - LDA for classification and dimensionality reduction
- [Naive Bayes Classifier Explained](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c) - Towards Data Science - Mathematical foundations and applications
- [Naive Bayes for Machine Learning](https://machinelearningmastery.com/naive-bayes-for-machine-learning/) - Machine Learning Mastery - Comprehensive guide to all Naive Bayes variants
- [Understanding Linear Discriminant Analysis (LDA)](https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/) - GeeksforGeeks - LDA concepts with Python implementation

### K-Nearest Neighbors (KNN)

- [A Complete Guide to K-Nearest Neighbors (with Python code)](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/) - Analytics Vidhya - Comprehensive KNN guide with distance metrics and implementation
- [K-Nearest Neighbors Algorithm in Machine Learning](https://www.geeksforgeeks.org/k-nearest-neighbours/) - GeeksforGeeks - Detailed explanation with code examples
- [K-Nearest Neighbors for Machine Learning](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/) - Machine Learning Mastery - Complete guide to KNN algorithm
- [KNN Algorithm: An Overview](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn) - DataCamp - Practical tutorial on KNN classification
- [Understanding K-Nearest Neighbors](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) - Towards Data Science - KNN fundamentals and use cases

### Support Vector Machines (SVM)

- [A Beginner's Guide to Support Vector Machines](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47) - Towards Data Science - Introduction to SVM concepts
- [Support Vector Machines (SVM) Algorithm Explained](https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python) - DataCamp - Practical SVM tutorial with scikit-learn
- [Support Vector Machines for Machine Learning](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/) - Machine Learning Mastery - Comprehensive guide to SVM
- [Support Vector Machines in Machine Learning](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) - GeeksforGeeks - SVM explanation with code examples
- [Understanding Support Vector Machine (SVM) Algorithm](https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/) - Analytics Vidhya - Complete beginner's guide to SVM
- [Understanding the Kernel Trick](https://towardsdatascience.com/understanding-the-kernel-trick-e0bc6112ef78) - Towards Data Science - Explanation of kernel methods in SVM

### Loss Functions and Optimization

- [5 Regression Loss Functions All Machine Learners Should Know](https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0) - Heartbeat (Comet ML) - Comprehensive guide to MSE, MAE, Huber, Log-Cosh, and Quantile losses
- [A Guide to Loss Functions in Machine Learning](https://builtin.com/data-science/loss-functions-in-machine-learning) - Built In - Overview of classification and regression loss functions
- [Binary Cross Entropy/Log Loss for Binary Classification](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) - Towards Data Science - Visual explanation of log loss
- [Common Loss Functions in Machine Learning](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23) - Towards Data Science - Comparison of MSE, Cross-Entropy, and Hinge Loss
- [Huber Loss in Machine Learning](https://www.geeksforgeeks.org/huber-loss-in-machine-learning/) - GeeksforGeeks - Explanation and implementation of Huber loss
- [Loss Functions Explained](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) - ML Cheatsheet - Comprehensive reference for common loss functions
- [Loss Functions in Machine Learning: The Ultimate Guide](https://neptune.ai/blog/loss-functions-in-machine-learning) - Neptune.ai - Detailed guide covering classification, regression, and ranking losses
- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/) - Raul Gomez - Clarification of cross-entropy terminology
- [Understanding Focal Loss: A Quick Read](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075) - Towards Data Science - Focal loss for imbalanced classification
- [Understanding Quantile Loss](https://towardsdatascience.com/quantile-loss-function-1f0b007c75cf) - Towards Data Science - Quantile regression and pinball loss

### Machine Learning Evaluation Metrics

- [20 Popular Machine Learning Metrics](https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce) - Towards Data Science - Comprehensive guide to classification and regression metrics
- [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) - Towards Data Science - When and why to use precision and recall
- [Classification Metrics Explained](https://www.datacamp.com/tutorial/tutorial-classification-metrics-machine-learning) - DataCamp - Practical tutorial on classification metrics
- [Evaluation Metrics for Classification Problems](https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/) - Analytics Vidhya - Detailed guide to choosing metrics
- [How to Choose the Right Evaluation Metric for Machine Learning Models](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/) - Machine Learning Mastery - Comprehensive metric selection guide
- [Precision-Recall Curves: What Are They and How Are They Used?](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) - Machine Learning Mastery - Understanding PR and ROC curves
- [Regression Metrics Explained](https://www.datacamp.com/tutorial/tutorial-regression-metrics-machine-learning) - DataCamp - Guide to MSE, RMSE, MAE, R², and MAPE
- [Understanding AUC-ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) - Towards Data Science - Comprehensive guide to ROC curves
- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) - Towards Data Science - Foundation of classification metrics
- [When to Use ROC vs Precision-Recall Curves?](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/) - Machine Learning Mastery - Choosing metrics for imbalanced data

### Unsupervised Learning (Clustering and Dimensionality Reduction)

- [A Comprehensive Guide to K-Means Clustering](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a) - Towards Data Science - K-Means algorithm, applications, and limitations
- [Clustering Performance Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) - Scikit-Learn - Metrics for evaluating clustering quality
- [DBSCAN Clustering Algorithm in Machine Learning](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html) - KDnuggets - DBSCAN theory and applications
- [DBSCAN: Density-Based Clustering Explained](https://towardsdatascience.com/dbscan-clustering-explained-97556a2ad556) - Towards Data Science - Comprehensive DBSCAN guide
- [Hierarchical Clustering Explained](https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec) - Towards Data Science - Agglomerative and divisive clustering
- [K-Means Clustering: Algorithm, Applications, Evaluation Methods](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/) - Analytics Vidhya - Complete K-Means guide
- [PCA: Principal Component Analysis Explained](https://builtin.com/data-science/step-step-explanation-principal-component-analysis) - Built In - Step-by-step PCA explanation
- [Principal Component Analysis (PCA) in Python](https://www.datacamp.com/tutorial/principal-component-analysis-in-python) - DataCamp - Practical PCA tutorial
- [Principal Component Analysis Explained Visually](https://setosa.io/ev/principal-component-analysis/) - Setosa - Interactive PCA visualization
- [The Complete Guide to Clustering Analysis: K-Means and Hierarchical](https://www.datacamp.com/tutorial/k-means-clustering-python) - DataCamp - Clustering algorithms comparison
- [Understanding DBSCAN and Implementation with Python](https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/) - Analytics Vidhya - DBSCAN implementation guide
- [Understanding K-Means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) - Towards Data Science - K-Means fundamentals

### Bagging and Ensemble Learning

- [A Comprehensive Guide to Ensemble Techniques: Bagging and Boosting](https://medium.com/@abhishekjainindore24/a-comprehensive-guide-to-ensemble-techniques-bagging-and-boosting-fa276e28da9f) - Abhishek Jain - Comprehensive overview of bagging and boosting techniques
- [A Guide to Bagging in Machine Learning: Ensemble Method to Reduce Variance and Improve Accuracy](https://www.datacamp.com/tutorial/what-bagging-in-machine-learning-a-guide-with-examples) - DataCamp - Practical guide to bagging with examples
- [Bagging, Boosting and Stacking: Ensemble Learning in ML Models](https://www.analyticsvidhya.com/blog/2023/01/ensemble-learning-methods-bagging-boosting-and-stacking/) - Analytics Vidhya (Updated April 2025) - Tutorial covering all three major ensemble techniques
- [Ensemble learning: Bagging and Boosting](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205) - Towards Data Science (Jan 2025) - Comparison of bagging and boosting methods
- [How to Develop a Bagging Ensemble with Python](https://machinelearningmastery.com/bagging-ensemble-with-python/) - MachineLearningMastery.com - Step-by-step implementation guide

### Random Forest

- [A Practical Guide to Random Forests in Machine Learning](https://www.digitalocean.com/community/tutorials/random-forest-in-machine-learning) - DigitalOcean (June 2025) - Practical guide with Python implementation
- [Random Forest Algorithm in Machine Learning](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/) - GeeksforGeeks (Dec 2025) - Comprehensive explanation with examples
- [Random Forest Algorithm in Machine Learning](https://www.simplilearn.com/tutorials/machine-learning-tutorial/random-forest-algorithm) - Simplilearn (May 2025) - Tutorial covering terminologies and use cases
- [Random Forest Classification in Python With Scikit-Learn: Step-by-Step Guide](https://www.datacamp.com/tutorial/random-forests-classifier-python) - DataCamp (Oct 2025) - Practical scikit-learn implementation guide
- [Random Forest: A Complete Guide for Machine Learning](https://builtin.com/data-science/random-forest-algorithm) - Built In - Complete guide to Random Forest algorithm
- [Random Forest, Explained: A Visual Guide with Code Examples](https://towardsdatascience.com/random-forest-explained-a-visual-guide-with-code-examples-9f736a6e1b3c/) - Towards Data Science - Visual explanations with code
- [Random Forest](https://mlu-explain.github.io/random-forest/) - MLU-Explain - Interactive visualization of Random Forest

### Gradient Boosting and XGBoost

- [A Guide to The Gradient Boosting Algorithm](https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm) - DataCamp - Comprehensive guide to gradient boosting concepts
- [A Tutorial and Use Case Example of the eXtreme Gradient Boosting (XGBoost) Algorithm for Drug Development Applications](https://ascpt.onlinelibrary.wiley.com/doi/full/10.1111/cts.70172) - Wiens et al., Clinical and Translational Science (2025) - Academic tutorial with practical use case
- [Complete Guide to Gradient Boosting and XGBoost in R](https://www.appsilon.com/post/r-xgboost) - Appsilon - Comprehensive R implementation guide
- [Extreme Gradient Boosting with XGBoost in Python](https://tilburgsciencehub.com/topics/analyze/machine-learning/supervised/xgboost_python/) - Tilburg Science Hub - Python implementation guide
- [Implementation of XGBoost (eXtreme Gradient Boosting)](https://www.geeksforgeeks.org/machine-learning/implementation-of-xgboost-extreme-gradient-boosting/) - GeeksforGeeks - Implementation tutorial
- [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) - XGBoost Official Documentation - Official introduction to boosted trees
- [Mastering XGBoost: The Ultimate Guide to Extreme Gradient Boosting](https://medium.com/@abhaysingh71711/mastering-xgboost-the-ultimate-guide-to-extreme-gradient-boosting-ac7fa2828047) - Abhay Singh - Comprehensive XGBoost guide
- [XGBoost: Complete Guide to Extreme Gradient Boosting with Mathematical Foundations, Optimization Techniques & Python Implementation](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation) - Michael Brenndoerfer (July 2025) - Mathematical deep dive with second-order optimization, regularization, and code examples

## Documentation

### Scikit-learn

#### Linear Models

- [Elastic-Net - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) - API reference for Elastic Net regression
- [Lasso - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) - API reference for Lasso regression
- [Linear Models - scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html) - User guide for all linear models including regression and classification
- [LinearRegression - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - API reference for ordinary least squares
- [LogisticRegression - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - API reference for logistic regression classification
- [Ridge - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) - API reference for Ridge regression

#### Probabilistic Models

- [BernoulliNB - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) - API reference for Bernoulli Naive Bayes (binary features)
- [GaussianNB - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - API reference for Gaussian Naive Bayes (continuous features)
- [LinearDiscriminantAnalysis - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) - API reference for LDA classifier and transformer
- [MultinomialNB - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) - API reference for Multinomial Naive Bayes (count data, text)
- [Naive Bayes - scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html) - User guide for all Naive Bayes variants
- [QuadraticDiscriminantAnalysis - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html) - API reference for QDA

#### Instance-Based Methods (K-Nearest Neighbors)

- [KNeighborsClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - API reference for KNN classification
- [KNeighborsRegressor - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - API reference for KNN regression
- [Nearest Neighbors - scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html) - User guide for nearest neighbors algorithms including KNN, radius neighbors, and nearest centroid

#### Support Vector Machines

- [LinearSVC - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) - API reference for linear SVM using liblinear (scalable for large datasets)
- [NuSVC - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) - API reference for Nu-Support Vector Classification
- [SVC - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - API reference for C-Support Vector Classification with kernel methods
- [SVR - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - API reference for Support Vector Regression
- [Support Vector Machines - scikit-learn](https://scikit-learn.org/stable/modules/svm.html) - User guide for SVM classification, regression, and outlier detection

#### Loss Functions

- [hinge_loss - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html) - API reference for hinge loss computation (SVM)
- [log_loss - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) - API reference for log loss (binary/multi-class cross-entropy)
- [mean_absolute_error - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - API reference for MAE (L1 loss)
- [mean_squared_error - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - API reference for MSE (L2 loss) and RMSE

#### Evaluation Metrics

- [Classification Metrics - scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) - User guide for classification metrics including accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- [Regression Metrics - scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) - User guide for regression metrics including MSE, MAE, R², MAPE
- [accuracy_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) - API reference for accuracy calculation
- [confusion_matrix - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - API reference for confusion matrix computation
- [f1_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) - API reference for F1-score (harmonic mean of precision and recall)
- [matthews_corrcoef - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) - API reference for Matthews Correlation Coefficient
- [precision_recall_curve - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) - API reference for precision-recall curve computation
- [precision_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) - API reference for precision calculation
- [r2_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) - API reference for R² (coefficient of determination)
- [recall_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) - API reference for recall (sensitivity)
- [roc_auc_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - API reference for ROC-AUC calculation
- [roc_curve - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) - API reference for ROC curve computation

#### Unsupervised Learning

- [calinski_harabasz_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) - API reference for Calinski-Harabasz clustering metric
- [Clustering - scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) - User guide for clustering algorithms and evaluation
- [davies_bouldin_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) - API reference for Davies-Bouldin clustering metric
- [DBSCAN - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) - API reference for DBSCAN density-based clustering
- [Decomposition - scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) - User guide for dimensionality reduction (PCA, NMF, ICA)
- [GaussianMixture - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) - API reference for Gaussian Mixture Models
- [KMeans - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - API reference for K-Means clustering
- [PCA - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - API reference for Principal Component Analysis
- [silhouette_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) - API reference for silhouette clustering metric
- [TSNE - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) - API reference for t-SNE dimensionality reduction

#### Model Selection and Evaluation

- [Cross-validation - scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) - Official guide to cross-validation strategies, model evaluation, and the model_selection module

#### Tree-Based Models

- [Decision Trees - scikit-learn](https://scikit-learn.org/stable/modules/tree.html) - User guide for decision tree algorithms
- [DecisionTreeClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - API reference for decision tree classification
- [DecisionTreeRegressor - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - API reference for decision tree regression
- [Ensemble Methods - scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html) - User guide covering bagging, boosting, and voting methods
- [GradientBoostingClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) - API reference for gradient boosting classification
- [GridSearchCV - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - API reference for exhaustive grid search with cross-validation
- [RandomForestClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - API reference for random forest classification
- [RandomizedSearchCV - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) - API reference for randomized hyperparameter search

### Ensemble Libraries

- [BaggingClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) - Official scikit-learn bagging implementation documentation
- [CatBoost Documentation](https://catboost.ai/) - Official CatBoost documentation for categorical boosting
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) - Official LightGBM documentation
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Official XGBoost documentation

### Hyperparameter Optimization

- [Optuna Documentation](https://optuna.readthedocs.io/) - Official documentation for Optuna hyperparameter optimization framework with pruning, visualization, and distributed optimization

### Statistical Modeling

- [Generalized Linear Models - Statsmodels](https://www.statsmodels.org/stable/glm.html) - Comprehensive GLM documentation for various distributions and link functions
- [Linear Regression - Statsmodels](https://www.statsmodels.org/stable/regression.html) - Statistical linear regression with hypothesis testing and diagnostics
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html) - Official statsmodels documentation for statistical modeling in Python
