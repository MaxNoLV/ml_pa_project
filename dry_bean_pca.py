import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

arff_file = arff.loadarff(
    "/Users/maksimnoskov/Documents/tsi_term2/ml_pa_project/Dry_Bean_Dataset.arff"
)
df = pd.DataFrame(arff_file[0])


df.rename(columns={"AspectRation": "AspectRatio"}, inplace=True)

df["Class"] = df["Class"].str.decode("utf-8")


labels = df["Class"]
features = df.drop("Class", axis=1)

# Split into training (80%) and test (20%) sets using Stratified Sampling

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.20, random_state=42, stratify=labels
)

train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size=0.10, random_state=42, stratify=train_labels
)

# PCA implementation

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ]
)

pipeline.fit(train_features)

pca = pipeline.named_steps["pca"]
explained_variance_ratio = pca.explained_variance_ratio_
for index, variance in enumerate(explained_variance_ratio):
    print(f"Principal Component {index + 1}: {variance:.4f} ({variance * 100:.2f}%)")

pca.n_components_

pca_train_features = pipeline.transform(train_features)
pca_val_features = pipeline.transform(val_features)
pca_test_features = pipeline.transform(test_features)


# Train SVM, DT and KNN models

svm_clf = SVC()
dt_clf = DecisionTreeClassifier(random_state=42)
knn_clf = KNeighborsClassifier()

svm_clf.fit(pca_train_features, train_labels)
dt_clf.fit(pca_train_features, train_labels)
knn_clf.fit(pca_train_features, train_labels)

# Predictions for SVM, DT and KNN

svm_predictions = svm_clf.predict(pca_val_features)
dt_predictions = dt_clf.predict(pca_val_features)
knn_predictions = knn_clf.predict(pca_val_features)

print("SVM Accuracy:", accuracy_score(val_labels, svm_predictions))
print("Decision Tree Accuracy:", accuracy_score(val_labels, dt_predictions))
print("KNN Accuracy:", accuracy_score(val_labels, knn_predictions))

# Detailed classification report
print("\nClassification Report for SVM:")
print(classification_report(val_labels, svm_predictions, digits=4))

print("Classification Report for Decision Tree:")
print(classification_report(val_labels, dt_predictions, digits=4))

print("Classification Report for KNN:")
print(classification_report(val_labels, knn_predictions, digits=4))

# Train RF and MLP

rf_clf = RandomForestClassifier(random_state=42)
mlp_clf = MLPClassifier(max_iter=500, random_state=42)

rf_clf.fit(pca_train_features, train_labels)
mlp_clf.fit(pca_train_features, train_labels)

# Predictions for RF and MLP

rf_predictions = rf_clf.predict(pca_val_features)
mlp_predictions = mlp_clf.predict(pca_val_features)

print("Random Forest Accuracy:", accuracy_score(val_labels, rf_predictions))
print("MLP Neural Network Accuracy:", accuracy_score(val_labels, mlp_predictions))

# Detailed classification reports
print("\nClassification Report for Random Forest:")
print(classification_report(val_labels, rf_predictions, digits=4))

print("Classification Report for MLP Neural Network:")
print(classification_report(val_labels, mlp_predictions, digits=4))
