import os

import numpy
import numpy as np
import matplotlib.pyplot as plt

from Preprocess.DatasetPlots import center_data, split_dataset, plot_feature_distributions, plot_feature_pairs
from Preprocess.LDA import lda_classification, compute_lda_projection_matrix, apply_lda, plot_lda_histogram, \
    find_best_threshold, classify


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


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


def plot_pca_explained_variance(D, output_dir='Output/PCA'):
    DC, _ = center_data(D)
    C = calculate_covariance(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = eigenvalues[::-1]
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure()
    plt.plot(np.arange(1, len(eigenvalues) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.grid()

    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    num_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
    plt.scatter(num_components_95, 0.95, color='red')
    plt.annotate(f'{num_components_95} components',
                 xy=(num_components_95, 0.95),
                 xytext=(num_components_95 + 1, 0.90),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend(loc='lower right')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'PCA_Explained_Variance_Improved.png'))
    plt.close()


def plot_pca_histograms(D, L, pca_dim=6, output_dir='Output/PCA_Histograms', mode='Train'):
    D_pca, P_pca = apply_PCA_from_dim(D, pca_dim)

    for i in range(pca_dim):
        plt.figure()
        for cls in np.unique(L):
            plt.hist(D_pca[i, L == cls], bins=30, alpha=0.5, label=f'Class {cls}')
        plt.xlabel(f'PCA Component {i + 1}')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Histogram of PCA Component {i + 1} - {mode}in Set')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'PCA_Component_{i + 1}_Histogram_{mode}.png'))
        plt.close()


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def calculate_covariance(D):
    mu, C = compute_mu_C(D)
    return C


def calculate_pca_projection_matrix(C, m):
    U, s, Vh = np.linalg.svd(C)
    P_pca = U[:, :m]
    return P_pca


def apply_pca(D, P_pca):
    return P_pca.T @ D


def apply_PCA_from_dim(D, m):
    DC, _ = center_data(D)
    C = calculate_covariance(DC)
    P_pca = calculate_pca_projection_matrix(C, m)
    D_pca = apply_pca(D, P_pca)
    return D_pca, P_pca


def PCA_LDA_analysis(DTE, DTR, LTR, LTE):
    # Split the dataset
    D_train, L_train, D_val, L_val = split_dataset(DTR, LTR)

    # Apply PCA and then LDA, and plot results for different PCA dimensions
    pca_dim_values = [2, 4, 6]
    for pca_dim in pca_dim_values:
        D_train_pca, P_pca = apply_PCA_from_dim(D_train, pca_dim)
        D_val_pca = apply_pca(D_val, P_pca)
        pca_lda_error_rate, pca_lda_threshold = pca_lda_classification(DTR, LTR, DTE, LTE, pca_dim)#(D_train_pca, L_train, D_val_pca, L_val, pca_dim)
        print(f"PCA (dim={pca_dim}) + LDA Classification Error Rate: {pca_lda_error_rate}")
        visualize_pca_data(D_train_pca, L_train, pca_dim)
        # Plot PCA histograms for both training and test datasets
        plot_pca_histograms(DTR, LTR, pca_dim, output_dir='Output/PCA_Histogram', mode='Train')
        plot_pca_histograms(DTE, LTE, pca_dim, output_dir='Output/PCA_Histogram', mode='Test')

    # Set the number of principal components
    mp = 6
    # PCA implementation
    DTR_pca, P_pca = apply_PCA_from_dim(DTR, mp)
    DTE_pca = apply_pca(DTE, P_pca)
    # Plot PCA results
    plot_feature_pairs(DTR_pca, LTR, output_dir='Output/PCA_FeaturePairPlots', origin=f'PCA_{mp}')
    plot_feature_distributions(DTR_pca, LTR, output_dir='Output/PCA_FeatureDistributions', origin=f'PCA_{mp}')
    # Plot explained variance
    plot_pca_explained_variance(DTR)

    # LDA implementation
    lda_dim = 1  # For binary classification
    W_lda = compute_lda_projection_matrix(DTR, LTR, lda_dim)
    DTR_lda = apply_lda(DTR, W_lda)
    DTE_lda = apply_lda(DTE, W_lda)



    # Plot LDA histograms for both training and test datasets
    plot_lda_histogram(DTR, LTR, output_dir='Output/LDA_Histogram', origin='LDA_Train')
    plot_lda_histogram(DTE, LTE, output_dir='Output/LDA_Histogram', origin='LDA_Test')

    # Apply LDA as classifier and find the best threshold
    lda_error_rate, lda_threshold = lda_classification(D_train, L_train, D_val, L_val)
    print(f"LDA Classification Error Rate: {lda_error_rate}")

    # Find the best threshold and check its impact
    D_val_lda = apply_lda(D_val, W_lda)
    best_threshold, best_error_rate = find_best_threshold(D_val_lda, L_val)
    print(f"Best Threshold for LDA: {best_threshold}, Best Error Rate: {best_error_rate}")

    # Test different thresholds
    thresholds = np.linspace(best_threshold - 1, best_threshold + 1, 10)
    for threshold in thresholds:
        predictions = classify(D_val_lda, threshold)
        error_rate = np.mean(predictions != L_val)
        print(f"Threshold: {threshold}, Error Rate: {error_rate}")

    # Analyze performance as a function of the number of PCA dimensions m
    pca_dim_values = [2, 4, 6]
    pca_lda_error_rates = []
    for pca_dim in pca_dim_values:
        D_train_pca, P_pca = apply_PCA_from_dim(D_train, pca_dim)
        D_val_pca = apply_pca(D_val, P_pca)
        pca_lda_error_rate, _ = pca_lda_classification(D_train_pca, L_train, D_val_pca, L_val, pca_dim)
        pca_lda_error_rates.append(pca_lda_error_rate)
        print(f"PCA (dim={pca_dim}) + LDA Classification Error Rate: {pca_lda_error_rate}")

    # Plot LDA results with the best PCA dimension
    best_pca_dim = pca_dim_values[np.argmin(pca_lda_error_rates)]
    DTR_pca, P_pca = apply_PCA_from_dim(DTR, best_pca_dim)
    DTE_pca = apply_pca(DTE, P_pca)
    plot_lda_histogram(DTR_pca, LTR, output_dir='Output/LDA_Histogram', origin=f'LDA_PCA_{best_pca_dim}_Train')
    plot_lda_histogram(DTE_pca, LTE, output_dir='Output/LDA_Histogram', origin=f'LDA_PCA_{best_pca_dim}_Test')

    print(f"Best PCA dimension: {best_pca_dim}")


def visualize_pca_data(D, L, pca_dim):
    plt.figure()
    for cls in np.unique(L):
        plt.scatter(D[0, L == cls], D[1, L == cls], label=f'Class {cls}', alpha=0.5)
    plt.xlabel(f'PCA Component 1')
    plt.ylabel(f'PCA Component 2')
    plt.legend()
    plt.title(f'PCA (dim={pca_dim}) Visualization')
    os.makedirs('Output/PCA/test', exist_ok=True)
    plt.savefig(os.path.join('Output/PCA/test', f'PCA_Visualization_{pca_dim}.png'))
    plt.close()


# Classification
def pca_lda_classification(D_train, L_train, D_val, L_val, pca_dim, lda_dim=1):
    # Apply PCA
    D_train_pca, P_pca = apply_PCA_from_dim(D_train, pca_dim)
    D_val_pca = apply_pca(D_val, P_pca)

    # Apply LDA
    W_lda = compute_lda_projection_matrix(D_train_pca, L_train, lda_dim)
    D_train_lda = apply_lda(D_train_pca, W_lda)
    D_val_lda = apply_lda(D_val_pca, W_lda)

    # Compute the threshold
    class_0 = D_train_lda[0, L_train == 0]
    class_1 = D_train_lda[0, L_train == 1]
    threshold = (class_0.mean() + class_1.mean()) / 2.0

    # Classify validation data
    P_val = np.zeros(L_val.shape)
    P_val[D_val_lda[0] >= threshold] = 1
    P_val[D_val_lda[0] < threshold] = 0

    # Compute the error rate
    error_rate = np.mean(P_val != L_val)
    return error_rate, threshold
