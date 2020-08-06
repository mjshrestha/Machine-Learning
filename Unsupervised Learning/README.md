## Unsupervised Learning and Dimensionality Reduction

This project is about unsupervised learning, especially exploring the clustering and dimensionality reduction algorithms and implementing them with Neural Networks. The clustering algorithms explored are Kmeans and expectation Maximization (EM), and the dimensionality reduction algorithms used were principal component analysis (PCA), independent component analysis (ICA), random projection (RP) and Random Forest Classifier (RFC). It is divided into three parts. The first part applies clustering algorithm to two datasets – “Pima Indian Diabetes” and “Wine Quality-red”, the second part includes dimensionality reduction and re-clustering on the same datasets, and the third part is applying Neural Networks on the data, pre-processed with clustering and dimensionality reduction algorithms. This assignment performs many experiments to understand unsupervised learning with limited analysis as it is more of an exploratory in nature. The project uses Scikit-learn library for the implementation of the algorithms.

### Datasets
1. Pima Indian Diabetes - https://www.openml.org/d/37

2. Wine Quality-red -  https://www.openml.org/d/40691


### Requires tools/Software
1. Python 3.6 with following packages:
- pandas, numpy, scikit-learn, matplotlib
2. Jupyter Notebook


### Instructions for running the code:
Run the following in Jupyter Notebook:
- **ML_Assignment3_Diabetes.ipynb** for the Pima Diabetes dataset includes
##### Clustering:
1. Kmeans on dataset
2. Expectation Maximization 
##### Dimensionlaity Reduction:
1. PCA
2. ICA
3. Randomized Projections
4. Random Forest Classifer
##### Neural Networks:
1. Clustering
2. Dimensionlaity Reduction

- **ML_Assignment3_Wine.ipynb** for the Wine Quality dataset includes
##### Clustering:
1. Kmeans on dataset
2. Expectation Maximization 
##### Dimensionlaity Reduction:
1. PCA
2. ICA
3. Randomized Projections
4. Random Forest Classifer
