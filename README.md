# Multiclass Classiffication. Dry Bean dataset.
Publicly available Dry Bean dataset is used in this experiment: [Dry Bean](https://archive.ics.uci.edu/dataset/602)

### Notebook structure
#### 1_beans_project_main.ipyn
- Import and examine the dataset.
- Create histograms and density plots to understand data distribution.
- Divide the dataset into training, validation, and test sets.
- Apply transformations to prepare the data for modeling.
- Train SVM, DT, KNN, RF, and MLP models with default parameters and validate them on the validation set.
- Tune RF, MLP, and SVM models; evaluate the best-performing models on the test set.

#### 2_beans_default_models_test.ipynb
- Train SVM, RF, and MLP models with default parameters and evaluate their performance on the test set.
- Display the confusion matrix for the best-performing model to analyze its accuracy and misclassifications.

#### 3_beans_PCA.ipynb
- Apply Principal Component Analysis (PCA) to the training and validation datasets to reduce dimensionality.
- Train SVM, DT, KNN, RF, and MLP models using the top 4 principal components (describe 95% variance) derived from PCA. Validate the performance of these models on the validation set.


