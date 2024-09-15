# KNN Classification and t-SNE Visualization on Syndrome Dataset

This script performs KNN classification using cosine and Euclidean distances, evaluates the model using metrics such as top-1 accuracy, top-5 accuracy, and ROC-AUC scores, and visualizes data using t-SNE. The results are saved as both text and PDF files.

## Prerequisites

Before running this script, make sure you have Python installed on your machine. This script has been tested with Python 3.9.

## Installation

1. Clone this repository or download the script files.
2. Place the data file `mini_gm_public_v0.1.p` in the root of the project directory.
3. Create a virtual environment to manage dependencies:
    ```bash
    python3 -m venv env
    ```
4. Activate the virtual environment:
    - On macOS/Linux:
        ```bash
        source env/bin/activate
        ```
    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
5. Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Script

1. Make sure the virtual environment is activated.
2. Run the script using Python:
    ```bash
    python main.py
    ```

## Deactivating the Virtual Environment

After running the script, you can deactivate the virtual environment by running:
```bash
deactivate
```



## Introduction

This report summarizes the work performed on a multi-class classification task involving syndrome identification based on image encodings. The objectives of the project were:

- To explore the dataset and understand its characteristics.
- To visualize high-dimensional embeddings using t-SNE.
- To perform K-Nearest Neighbors (KNN) classification using both cosine and Euclidean distances.
- To compare the performance of the classifiers and analyze the results.

## Data Exploration

The dataset consists of image encodings representing different syndromes. Each entry in the dataset includes:

- **syndrome_id**: The identifier for the syndrome (target variable).
- **subject_id**: The unique identifier for each subject.
- **image_id**: The unique identifier for each image.
- **encoding**: A list representing the high-dimensional feature vector extracted from each image.

### Class Distribution

An examination of the class distribution revealed an imbalance in the number of samples per syndrome.

| Syndrome ID | Number of Samples |
|-------------|-------------------|
| 300000034   | 210               |
| 300000080   | 198               |
| 100192430   | 136               |
| 300000007   | 115               |
| 300000082   | 98                |
| 100610443   | 89                |
| 300000018   | 74                |
| 100180860   | 67                |
| 100610883   | 65                |
| 700018215   | 64                |

### Missing Values and Encoding Dimensions

- **Missing Values**: There are no missing values in the dataset.
- **Encoding Lengths**: All encoding vectors have a consistent length of 320.

**Conclusion**: The dataset is complete with uniform feature dimensions, but there is a noticeable class imbalance.

## Data Preprocessing

Given the high dimensionality of the data and the imbalance in class distribution, the following preprocessing steps were taken:

1. **Standardization**: The embeddings were standardized using `StandardScaler` to ensure each feature contributes equally to the distance calculations.
2. **Dimensionality Reduction with PCA**: Before applying t-SNE, Principal Component Analysis (PCA) was used to reduce the embeddings to 50 dimensions. This step helps in speeding up the t-SNE computation and potentially enhances the visualization by focusing on the most significant components.

## t-SNE Visualization

t-SNE (t-distributed Stochastic Neighbor Embedding) was employed to visualize the high-dimensional embeddings in a 2D space.

### Parameter Choices

- **Perplexity**: Set to 30.
  - **Reasoning**: Perplexity is a measure of the effective number of local neighbors. It should be less than the size of the smallest class. A value of 30 balances between local and global aspects of the data.
- **Learning Rate**: Set to 200.
  - **Reasoning**: A learning rate between 10 and 1000 is typical. A value of 200 is a good starting point for datasets of this size.
- **Number of Iterations (n_iter)**: Set to 1000.
  - **Reasoning**: This is the default value. Increasing n_iter can lead to a more stable embedding but at the cost of computation time.
- **Metric**: Set to 'cosine'.
  - **Reasoning**: Cosine distance is often more appropriate for high-dimensional data where the magnitude of vectors is less informative than their orientation.

### Visualization

The t-SNE plot revealed clusters corresponding to different syndromes. Some overlap between clusters was observed, indicating that certain syndromes may share similar features in the embedding space.

**Insights**:

- **Cluster Separation**: Some syndromes are well-separated, suggesting the embeddings capture distinguishing features effectively.
- **Overlap**: Overlapping clusters may indicate similarities between certain syndromes or limitations in the embedding's ability to discriminate between them.

## Classification Task

A K-Nearest Neighbors classifier was implemented using both cosine and Euclidean distances to classify images into their respective syndromes.

### Cross-Validation Setup

- **Stratified 10-Fold Cross-Validation**: Ensures each fold has a representative class distribution, addressing the class imbalance issue.
- **Standardization**: Applied to both training and test embeddings within each fold to maintain consistency.

### Parameter Choices

- **Number of Neighbors (n_neighbors)**: Set to 5.
  - **Reasoning**: A common default value that considers a moderate number of neighbors. It balances between noise reduction (higher k) and capturing local structure (lower k).
- **Distance Metrics**:
  - **Cosine Distance**: Captures the orientation between vectors, which is useful in high-dimensional spaces.
  - **Euclidean Distance**: Measures the straight-line distance between points in space.

## Discussion

### Why Cosine Distance Performed Better

- **High-Dimensional Data**: In high-dimensional spaces, cosine similarity can be more informative as it measures the angle between vectors rather than their magnitude.
- **Feature Representation**: The embeddings may encode information in such a way that the orientation (direction) is more discriminative for classification.

### Parameter Choices Justification

- **t-SNE Parameters**:
  - **Perplexity (30)**: Chosen to balance local and global data structure representation, suitable for the dataset size.
  - **Learning Rate (200)**: A moderate value to ensure convergence without overshooting minima.
  - **Iterations (1000)**: Provides sufficient optimization steps for the embedding to stabilize.
  - **Metric ('cosine')**: Aligns with the distance metric that yielded better classification performance, and is suitable for high-dimensional data.
- **Classification Parameters**:
  - **Number of Neighbors (5)**: Balances sensitivity to noise and the ability to capture local structure in the data.
  - **Distance Metrics**:
    - **Cosine Distance**: Chosen based on its effectiveness in the t-SNE visualization and its suitability for high-dimensional feature spaces.
    - **Euclidean Distance**: Included for comparison, as it is a standard distance metric in many applications.

### Data Exploration Considerations

- **Class Imbalance**: The variability in class sizes could affect the classifier's ability to learn minority classes effectively.
- **Uniform Encoding Dimensions**: Consistent feature dimensions simplify preprocessing and ensure compatibility across different computational steps.
- **No Missing Values**: Eliminates the need for imputation strategies, allowing focus on modeling.
