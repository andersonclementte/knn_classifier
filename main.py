import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import label_binarize, StandardScaler

# Load data
with open('mini_gm_public_v0.1.p', 'rb') as file:
    data = pickle.load(file)

# Flatten the data into a DataFrame
flattened_data = []
for syndrome_id, subjects in data.items():
    for subject_id, images in subjects.items():
        for image_id, encoding in images.items():
            flattened_data.append({
                'syndrome_id': syndrome_id,
                'subject_id': subject_id,
                'image_id': image_id,
                'encoding': encoding
            })
df = pd.DataFrame(flattened_data)

# Prepare embeddings and labels
embeddings = np.array(df['encoding'].tolist())
labels = np.array(df['syndrome_id'])

# -------------------- t-SNE Visualization --------------------

# Scale the embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Reduce dimensions with PCA before t-SNE
pca = PCA(n_components=50, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_scaled)

# t-SNE parameters
perplexity = 30
learning_rate = 200
n_iter = 1000

# Run t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=learning_rate,
    n_iter=n_iter,
    random_state=42,
    metric='cosine'
)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'syndrome_id': labels
})

# Plot the t-SNE result
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='x', y='y',
    hue='syndrome_id',
    palette='bright',
    data=tsne_df,
    legend='full',
    alpha=0.7,
    s=50
)
plt.title('t-SNE Visualization of Embeddings')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
# plt.show()

# -------------------- KNN Classification --------------------

# Set up 10-fold cross-validation
n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize variables for ROC AUC and metrics
mean_fpr = np.linspace(0, 1, 100)
tprs_cosine, aucs_cosine, tprs_euclidean, aucs_euclidean = [], [], [], []
top1_accuracies_cosine, top5_accuracies_cosine = [], []
top1_accuracies_euclidean, top5_accuracies_euclidean = [], []
fold_numbers, mean_auc_cosine_per_fold, mean_auc_euclidean_per_fold = [], [], []

# Number of classes
n_classes = len(np.unique(labels))

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(cv.split(embeddings, labels), 1):
    train_embeddings, test_embeddings = embeddings[train_index], embeddings[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Standardize the embeddings
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)

    # --------------------- Cosine Distance ---------------------
    # Compute cosine distances
    cosine_train_distances = np.array([[cosine(vec1, vec2) for vec2 in train_embeddings_scaled] for vec1 in train_embeddings_scaled])
    cosine_test_distances = np.array([[cosine(test_vec, train_vec) for train_vec in train_embeddings_scaled] for test_vec in test_embeddings_scaled])

    # KNN classifier using precomputed cosine distances
    knn_cosine = KNeighborsClassifier(metric='precomputed', n_neighbors=5)
    knn_cosine.fit(cosine_train_distances, train_labels)

    # Predict probabilities and labels for test set
    cosine_probabilities = knn_cosine.predict_proba(cosine_test_distances)
    cosine_pred_labels = knn_cosine.predict(cosine_test_distances)

    # Binarize labels for ROC computation
    test_labels_bin = label_binarize(test_labels, classes=np.unique(labels))

    # Calculate metrics for cosine
    top1_accuracies_cosine.append(accuracy_score(test_labels, cosine_pred_labels))
    top5_accuracies_cosine.append(top_k_accuracy_score(test_labels, cosine_probabilities, k=5, labels=np.unique(labels)))
    fold_numbers.append(fold)
    per_fold_aucs_cosine = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(test_labels_bin[:, i], cosine_probabilities[:, i])
        tprs_cosine.append(np.interp(mean_fpr, fpr, tpr))
        tprs_cosine[-1][0] = 0.0
        per_fold_aucs_cosine.append(auc(fpr, tpr))
    mean_auc_cosine_per_fold.append(np.mean(per_fold_aucs_cosine))

    # --------------------- Euclidean Distance ---------------------
    # Compute euclidean distances
    euclidean_train_distances = np.array([[euclidean(vec1, vec2) for vec2 in train_embeddings_scaled] for vec1 in train_embeddings_scaled])
    euclidean_test_distances = np.array([[euclidean(test_vec, train_vec) for train_vec in train_embeddings_scaled] for test_vec in test_embeddings_scaled])

    # KNN classifier using precomputed euclidean distances
    knn_euclidean = KNeighborsClassifier(metric='precomputed', n_neighbors=5)
    knn_euclidean.fit(euclidean_train_distances, train_labels)

    # Predict probabilities and labels for test set
    euclidean_probabilities = knn_euclidean.predict_proba(euclidean_test_distances)
    euclidean_pred_labels = knn_euclidean.predict(euclidean_test_distances)

    # Calculate metrics for euclidean
    top1_accuracies_euclidean.append(accuracy_score(test_labels, euclidean_pred_labels))
    top5_accuracies_euclidean.append(top_k_accuracy_score(test_labels, euclidean_probabilities, k=5, labels=np.unique(labels)))
    per_fold_aucs_euclidean = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(test_labels_bin[:, i], euclidean_probabilities[:, i])
        tprs_euclidean.append(np.interp(mean_fpr, fpr, tpr))
        tprs_euclidean[-1][0] = 0.0
        per_fold_aucs_euclidean.append(auc(fpr, tpr))
    mean_auc_euclidean_per_fold.append(np.mean(per_fold_aucs_euclidean))

# -------------------- Plot ROC Curves --------------------

fig, ax = plt.subplots(figsize=(8, 6))

# Plot Mean ROC Curve for Cosine
mean_tpr_cosine = np.mean(tprs_cosine, axis=0)
mean_tpr_cosine[-1] = 1.0
mean_auc_cosine = auc(mean_fpr, mean_tpr_cosine)
std_auc_cosine = np.std(mean_auc_cosine_per_fold)
ax.plot(
    mean_fpr,
    mean_tpr_cosine,
    color="blue",
    label=r"Mean Cosine ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_cosine, std_auc_cosine),
    lw=2,
    alpha=0.8,
)

# Plot Mean ROC Curve for Euclidean
mean_tpr_euclidean = np.mean(tprs_euclidean, axis=0)
mean_tpr_euclidean[-1] = 1.0
mean_auc_euclidean = auc(mean_fpr, mean_tpr_euclidean)
std_auc_euclidean = np.std(mean_auc_euclidean_per_fold)
ax.plot(
    mean_fpr,
    mean_tpr_euclidean,
    color="green",
    label=r"Mean Euclidean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_euclidean, std_auc_euclidean),
    lw=2,
    alpha=0.8,
)

# Plot settings
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Mean ROC Curve Comparison (Cosine vs Euclidean)",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_auc_comparison_multiclass.png", dpi=300)
# plt.show()

# -------------------- Save Results --------------------

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Fold': fold_numbers,
    'Top-1 Accuracy (Cosine)': top1_accuracies_cosine,
    'Top-5 Accuracy (Cosine)': top5_accuracies_cosine,
    'AUC (Cosine)': mean_auc_cosine_per_fold,
    'Top-1 Accuracy (Euclidean)': top1_accuracies_euclidean,
    'Top-5 Accuracy (Euclidean)': top5_accuracies_euclidean,
    'AUC (Euclidean)': mean_auc_euclidean_per_fold
})

# Calculate mean and standard deviation
mean_results = results_df.mean(numeric_only=True)
std_results = results_df.std(numeric_only=True)

# Append mean and std to the DataFrame
mean_std_df = pd.DataFrame({
    'Fold': ['Mean', 'Std'],
    'Top-1 Accuracy (Cosine)': [mean_results['Top-1 Accuracy (Cosine)'], std_results['Top-1 Accuracy (Cosine)']],
    'Top-5 Accuracy (Cosine)': [mean_results['Top-5 Accuracy (Cosine)'], std_results['Top-5 Accuracy (Cosine)']],
    'AUC (Cosine)': [mean_results['AUC (Cosine)'], std_results['AUC (Cosine)']],
    'Top-1 Accuracy (Euclidean)': [mean_results['Top-1 Accuracy (Euclidean)'], std_results['Top-1 Accuracy (Euclidean)']],
    'Top-5 Accuracy (Euclidean)': [mean_results['Top-5 Accuracy (Euclidean)'], std_results['Top-5 Accuracy (Euclidean)']],
    'AUC (Euclidean)': [mean_results['AUC (Euclidean)'], std_results['AUC (Euclidean)']]
})

final_results_df = pd.concat([results_df, mean_std_df], ignore_index=True)

# Save results to a TXT file
with open('knn_comparison_results.txt', 'w') as f:
    f.write(final_results_df.to_string(index=False))

# Save the table as a PDF
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=final_results_df.values, colLabels=final_results_df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
fig.tight_layout()
fig.savefig('knn_comparison_results.pdf', dpi=300)
plt.close(fig)
