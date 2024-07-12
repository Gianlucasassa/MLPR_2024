import os

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from Preprocess.PCA import *


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def compute_class_means(D, L):
    """
    Computes the means of each class.

    Args:
    D (numpy.ndarray): The data matrix.
    L (numpy.ndarray): The label vector.

    Returns:
    list of numpy.ndarray: The list of means for each class.
    """
    classes = np.unique(L)
    means = []
    for cls in classes:
        class_data = D[:, L == cls]
        means.append(class_data.mean(axis=1).reshape(-1, 1))
    return means


def compute_within_class_covariance(D, L):
    """
    Computes the within-class covariance matrix.

    Args:
    D (numpy.ndarray): The data matrix.
    L (numpy.ndarray): The label vector.

    Returns:
    numpy.ndarray: The within-class covariance matrix.
    """
    classes = np.unique(L)
    SW = np.zeros((D.shape[0], D.shape[0]))
    for cls in classes:
        class_data = D[:, L == cls]
        centered_data, _ = center_data(class_data)
        SW += np.dot(centered_data, centered_data.T)
    return SW / D.shape[1]


def compute_between_class_covariance(D, L):
    """
    Computes the between-class covariance matrix.

    Args:
    D (numpy.ndarray): The data matrix.
    L (numpy.ndarray): The label vector.

    Returns:
    numpy.ndarray: The between-class covariance matrix.
    """
    mu = vcol(D.mean(axis=1))
    class_means = compute_class_means(D, L)
    SB = np.zeros((D.shape[0], D.shape[0]))
    classes = np.unique(L)
    for cls, mu_cls in zip(classes, class_means):
        n_cls = D[:, L == cls].shape[1]
        diff = mu_cls - mu
        SB += n_cls * np.dot(diff, diff.T)
    return SB / D.shape[1]


def compute_lda_projection_matrix(D, L, m):
    """
    Computes the LDA projection matrix.

    Args:
    D (numpy.ndarray): The data matrix.
    L (numpy.ndarray): The label vector.
    m (int): The number of LDA components.

    Returns:
    numpy.ndarray: The LDA projection matrix.
    """
    SW = compute_within_class_covariance(D, L)
    SB = compute_between_class_covariance(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    return U[:, ::-1][:, :m]


def apply_lda(D, P_lda):
    """
    Applies LDA to the data.

    Args:
    D (numpy.ndarray): The data matrix.
    P_lda (numpy.ndarray): The LDA projection matrix.

    Returns:
    numpy.ndarray: The LDA-transformed data.
    """
    return np.dot(P_lda.T, D)


def plot_lda_histogram(D, L, output_dir='Output/LDA_Histogram'):
    W_lda = compute_lda_projection_matrix(D, L, 1)
    D_lda = apply_lda(D, W_lda)

    plt.figure()
    for cls in np.unique(L):
        plt.hist(D_lda[0, L == cls], bins=30, alpha=0.5, label=f'Class {cls}')
    plt.xlabel('LDA Component 1')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of LDA Component 1')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'LDA_Component_1_Histogram.png'))
    plt.close()


# Classification
def pca_lda_classification(D_train, L_train, D_val, L_val, pca_dim, lda_dim=1):
    """
    Applies PCA followed by LDA for classification.

    Args:
    D_train (numpy.ndarray): The training data matrix.
    L_train (numpy.ndarray): The training label vector.
    D_val (numpy.ndarray): The validation data matrix.
    L_val (numpy.ndarray): The validation label vector.
    pca_dim (int): The number of PCA components.
    lda_dim (int): The number of LDA components.

    Returns:
    tuple: The error rate on the validation set and the threshold.
    """
    P_pca = estimate_pca(D_train, pca_dim)
    D_train_pca = apply_pca(D_train, P_pca)
    D_val_pca = apply_pca(D_val, P_pca)

    error_rate, threshold = lda_classification(D_train_pca, L_train, D_val_pca, L_val, lda_dim)
    return error_rate, threshold


def lda_classification(D_train, L_train, D_val, L_val, lda_dim=1):
    """
    Applies LDA for classification.

    Args:
    D_train (numpy.ndarray): The training data matrix.
    L_train (numpy.ndarray): The training label vector.
    D_val (numpy.ndarray): The validation data matrix.
    L_val (numpy.ndarray): The validation label vector.
    lda_dim (int): The number of LDA components.

    Returns:
    tuple: The error rate on the validation set and the threshold.
    """
    W_lda = compute_lda_projection_matrix(D_train, L_train, lda_dim)
    D_train_lda = apply_lda(D_train, W_lda)
    D_val_lda = apply_lda(D_val, W_lda)

    class_0 = D_train_lda[0, L_train == 0]
    class_1 = D_train_lda[0, L_train == 1]
    if len(class_0) == 0 or len(class_1) == 0:
        raise ValueError("Error: One of the classes is not present in the training set.")

    threshold = (class_0.mean() + class_1.mean()) / 2.0

    P_val = np.zeros(L_val.shape)
    P_val[D_val_lda[0] >= threshold] = 1
    P_val[D_val_lda[0] < threshold] = 0

    error_rate = np.mean(P_val != L_val)
    return error_rate, threshold


def estimate_threshold(D_lda, L):
    class_0 = D_lda[0, L == 0]
    class_1 = D_lda[0, L == 1]
    if len(class_0) == 0 or len(class_1) == 0:
        raise ValueError("Error: One of the classes is not present in the training set.")
    return (class_0.mean() + class_1.mean()) / 2.0


def classify(D_lda, threshold):
    P_val = np.zeros(D_lda.shape[1])
    P_val[D_lda[0] >= threshold] = 1
    P_val[D_lda[0] < threshold] = 0
    return P_val


def find_best_threshold(D_lda, L):
    """
    Finds the best threshold for classification.

    Args:
    D_lda (numpy.ndarray): The LDA-transformed data matrix.
    L (numpy.ndarray): The label vector.

    Returns:
    tuple: The best threshold and the corresponding error rate.
    """
    class_0 = D_lda[0, L == 0]
    class_1 = D_lda[0, L == 1]
    thresholds = np.linspace(min(min(class_0), min(class_1)), max(max(class_0), max(class_1)), 100)
    best_error_rate = float('inf')
    best_threshold = None

    for threshold in thresholds:

        P = np.zeros(L.shape)
        P[D_lda[0] >= threshold] = 1
        P[D_lda[0] < threshold] = 0
        error_rate = np.mean(P != L)
        print(f"trying threshold {threshold}, has error rate of {error_rate}")
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold

    return best_threshold, best_error_rate
