import os

import numpy as np
import matplotlib.pyplot as plt

from Preprocess.DatasetPlots import center_data, split_dataset, plot_feature_distributions, plot_feature_pairs
from Preprocess.LDA import lda_classification, compute_lda_projection_matrix, apply_lda, plot_lda_histogram, \
    find_best_threshold


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def calculate_covariance(D):
    """
    Calculates the covariance matrix for the data matrix D.

    Args:
    D (numpy.ndarray): The data matrix.

    Returns:
    numpy.ndarray: The covariance matrix.
    """
    mu = D.mean(1)
    C = 0
    for i in range(D.shape[1]):
        C += np.dot(D[:, i:i + 1] - mu, (D[:, i:i + 1] - mu).T)
    C /= float(D.shape[1])
    return C


def compute_covariance_matrix(DC):
    """
    Computes the covariance matrix for the centered data.

    Args:
    DC (numpy.ndarray): The centered data matrix.

    Returns:
    numpy.ndarray: The covariance matrix.
    """
    return np.dot(DC, DC.T) / DC.shape[1]


def calculate_centered_covariance(DC):
    """
    Calculates the covariance matrix for centered data.

    Args:
    DC (numpy.ndarray): The centered data matrix.

    Returns:
    numpy.ndarray: The covariance matrix.
    """
    C = 0
    for i in range(DC.shape[1]):
        C += np.dot(DC[:, i:i + 1], DC[:, i:i + 1].T)
    C /= float(DC.shape[1])
    return C



def calculate_pca_projection_matrix(C, m):
    """
    Calculates the PCA projection matrix using the covariance matrix.

    Args:
    C (numpy.ndarray): The covariance matrix.
    m (int): The number of principal components.

    Returns:
    numpy.ndarray: The PCA projection matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    sorted_indices = np.argsort(eigenvalues)[::-1][:m]  # Select the top m components
    P = eigenvectors[:, sorted_indices]
    return P


def apply_pca(D, P_pca):
    """
    Applies PCA to the data.

    Args:
    D (numpy.ndarray): The data matrix.
    P_pca (numpy.ndarray): The PCA projection matrix.

    Returns:
    numpy.ndarray: The PCA-transformed data.
    """
    return np.dot(P_pca.T, D)


def plot_pca_explained_variance(D, output_dir='Output/PCA'):
    """
    Plots the cumulative explained variance of the principal components.

    Args:
    D (numpy.ndarray): The data matrix.
    output_dir (str): The directory to save the plot.
    """
    DC, _ = center_data(D)
    C = calculate_centered_covariance(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = eigenvalues[::-1]
    explained_variance = eigenvalues / np.sum(eigenvalues)

    plt.figure()
    plt.plot(np.arange(1, len(eigenvalues) + 1), np.cumsum(explained_variance), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'PCA_Explained_Variance.png'))
    plt.close()


def estimate_pca(D, m):
    """
    Estimates the PCA projection matrix.

    Args:
    D (numpy.ndarray): The data matrix.
    m (int): The number of principal components.

    Returns:
    numpy.ndarray: The PCA projection matrix.
    """
    DC, _ = center_data(D)
    C = compute_covariance_matrix(DC)
    P_pca = calculate_pca_projection_matrix(C, m)
    return P_pca


def plot_pca_histograms(D, L, pca_dim=6, output_dir='Output/PCA_Histograms'):
    P_pca = estimate_pca(D, pca_dim)
    D_pca = apply_pca(D, P_pca)

    for i in range(pca_dim):
        plt.figure()
        for cls in np.unique(L):
            plt.hist(D_pca[i, L == cls], bins=30, alpha=0.5, label=f'Class {cls}')
        plt.xlabel(f'PCA Component {i + 1}')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Histogram of PCA Component {i + 1}')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'PCA_Component_{i + 1}_Histogram.png'))
        plt.close()

def PCA_LDA_analysis(DTE, DTR, LTR):
    # Split the dataset
    D_train, L_train, D_val, L_val = split_dataset(DTR, LTR)
    # Apply PCA and then LDA, and plot results
    pca_dim_values = [2, 4, 6]
    for pca_dim in pca_dim_values:
        P_pca = estimate_pca(D_train, pca_dim)
        D_train_pca = apply_pca(D_train, P_pca)
        D_val_pca = apply_pca(D_val, P_pca)

        pca_lda_error_rate, pca_lda_threshold = lda_classification(D_train_pca, L_train, D_val_pca, L_val)
        print(f"PCA (dim={pca_dim}) + LDA Classification Error Rate: {pca_lda_error_rate}")
    # Set the number of principal components
    mp = 6
    # PCA implementation
    P_pca = estimate_pca(DTR, mp)
    DTR_pca = apply_pca(DTR, P_pca)
    DTE_pca = apply_pca(DTE, P_pca)
    # Plot PCA results
    plot_feature_pairs(DTR_pca, LTR, output_dir='Output/PCA_FeaturePairPlots')
    plot_feature_distributions(DTR_pca, LTR, output_dir='Output/PCA_FeatureDistributions')
    # Plot explained variance
    plot_pca_explained_variance(DTR)
    # LDA implementation
    lda_dim = 1  # For binary classification
    W_lda = compute_lda_projection_matrix(DTR, LTR, lda_dim)
    DTR_lda = apply_lda(DTR, W_lda)
    DTE_lda = apply_lda(DTE, W_lda)
    # Plot LDA histogram
    plot_lda_histogram(DTR, LTR)
    # Plot LDA histogram
    plot_lda_histogram(DTR, LTR)
    # Plot PCA and LDA histograms
    plot_pca_histograms(DTR, LTR)
    plot_lda_histogram(DTR, LTR)
    # Apply LDA as classifier
    lda_error_rate, lda_threshold = lda_classification(D_train, L_train, D_val, L_val)
    print(f"LDA Classification Error Rate: {lda_error_rate}")
    # Find the best threshold
    best_threshold, best_error_rate = find_best_threshold(D_val, L_val)
    print(f"Best Threshold: {best_threshold}, Best Error Rate: {best_error_rate}")
