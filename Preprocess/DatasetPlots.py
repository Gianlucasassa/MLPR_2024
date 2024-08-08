import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from load import data_statistics_light


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_distributions(D, L, output_dir='Output/FeatureDistributions', origin='main', num_bins=10):
    """
    Plots and saves histograms of feature distributions separated by class.

    Args:
    D (numpy.ndarray): 2D data array with features in rows and samples in columns.
    L (numpy.ndarray): 1D array of labels corresponding to the columns of D.
    output_dir (str): The directory where plots will be saved.
    num_bins (int): Number of bins for the histograms.
    """
    num_features = D.shape[0]  # Number of features

    if origin.startswith("PCA"):
        add_on = f" - PCA with {origin.split('_')[1]}"
        add_on_save = f"-PCA{origin.split('_')[1]}"
        print("found Pca fd")
    else:
        add_on = ""
        add_on_save = ""

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Masks to separate the data based on the class labels
    genuine_mask = (L == 1)
    fake_mask = (L == 0)

    # Separating the data into genuine and fake based on the masks
    D_genuine = D[:, genuine_mask]
    D_fake = D[:, fake_mask]

    # Create histograms for each feature
    for i in range(num_features):
        plt.figure(figsize=(8, 5))  # Smaller figure size

        # Plot histograms for genuine and fake classes
        plt.hist(D_genuine[i, :], bins=num_bins, alpha=0.5, label='Genuine', density=True, color='blue')
        plt.hist(D_fake[i, :], bins=num_bins, alpha=0.5, label='Fake', density=True, color='orange')

        plt.title(f'Distribution of Feature {i + 1}{add_on}')
        plt.xlabel(f'Feature {i + 1} Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plot_filename = os.path.join(output_dir, f'Feature_{i + 1}{add_on_save}_Distribution.png')
        plt.savefig(plot_filename)
        plt.close()


def plot_feature_pairs(D, L, output_dir='Output/FeaturePairPlots', origin='main'):
    """
    Plots and saves scatter plots of feature pairs, separated by class.

    Args:
    D (numpy.ndarray): 2D data array with features in rows and samples in columns.
    L (numpy.ndarray): 1D array of labels corresponding to the columns of D.
    output_dir (str): The directory where scatter plots will be saved.
    """
    num_features = D.shape[0]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define colors for classes
    colors = {0: 'orange', 1: 'blue'}

    if origin.startswith("PCA"):
        add_on = f" - PCA with {origin.split('_')[1]}"
        add_on_save = f"-PCA{origin.split('_')[1]}"

    elif origin.startswith("LDA"):
        add_on = " - LDA"
        add_on_save = "-LDA"

    else:
        add_on = ""
        add_on_save = ""

    # Create scatter plots for each pair of features
    for i in range(num_features):
        for j in range(i + 1, num_features):
            plt.figure(figsize=(8, 6))

            for class_value in [0, 1]:
                class_mask = L == class_value
                plt.scatter(D[i, class_mask], D[j, class_mask], alpha=0.5,
                            c=colors[class_value], label=f'{"Genuine" if class_value else "Fake"}')

            plt.title(f'Feature {i + 1} vs Feature {j + 1}{add_on}')
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel(f'Feature {j + 1}')
            plt.legend()
            plt.grid(True)

            # Save the figure
            plot_filename = os.path.join(output_dir, f'Feature_{i + 1}_vs_Feature_{j + 1}_Scatter{add_on_save}.png')
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to free memory


def compute_mean(D):
    # Compute the mean for each feature and reshape it
    mean = D.mean(1).reshape(D.shape[0], 1)
    return mean


def compute_covariance_matrix(D):
    # Compute the centered data
    mu = compute_mean(D)
    DC = D - mu
    # Compute the covariance matrix using the centered data
    covariance_matrix = (DC @ DC.T) / float(D.shape[1])
    return covariance_matrix


def compute_variance(D):
    # Compute the variance for each feature
    variance = D.var(axis=1).reshape(D.shape[0], 1)
    return variance


def compute_standard_deviation(D):
    # Compute the standard deviation for each feature
    std_dev = D.std(axis=1).reshape(D.shape[0], 1)
    return std_dev


def plot_histograms(D, L, output_dir='Output/Dataset/Histograms'):
    os.makedirs(output_dir, exist_ok=True)
    features = D.shape[0]
    classes = np.unique(L)
    for i in range(features):
        plt.figure()
        for cls in classes:
            plt.hist(D[i, L == cls], alpha=0.5, label=f'Class {cls}')
        plt.title(f'Feature {i + 1} Histogram')
        plt.xlabel(f'Feature {i + 1}')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plot_filename = os.path.join(output_dir, f'Feature_{i + 1}_Histogram.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory


def plot_pairwise_scatter(D, L, output_dir='Output/Dataset/ScatterPlots'):
    os.makedirs(output_dir, exist_ok=True)
    features = D.shape[0]
    classes = np.unique(L)
    for i in range(features):
        for j in range(i + 1, features):
            plt.figure()
            for cls in classes:
                plt.scatter(D[i, L == cls], D[j, L == cls], alpha=0.5, label=f'Class {cls}')
            plt.title(f'Feature {i + 1} vs Feature {j + 1}')
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel(f'Feature {j + 1}')
            plt.legend(loc='upper right')
            plot_filename = os.path.join(output_dir, f'Feature_{i + 1}_vs_Feature_{j + 1}_Scatter.png')
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to free memory


def mcol(v):
    return v.reshape((v.size, 1))


def featurePlot(D, L, m, type):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    print(m)
    for i in range(m):
        plt.figure()
        # plt.xlabel("Feature " + str(i+1))
        plt.ylabel("Number of elements")
        plt.hist(D0[i, :], bins=60, density=True, alpha=0.7, label="Male")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Female")
        plt.legend()
        # plt.show()
        path = f'Images/Dataset/Features_{type}/'
        os.makedirs(path, exist_ok=True)
        if type == "PCA":
            plt.title(f' PCA ')
        elif type == "LDA":
            plt.title(f' LDA ')
        else:
            plt.title(f'Feature {i + 1}')

        plt.savefig(os.path.join(path, f'Feature {i + 1}.png'))
        plt.close()


def generalPlot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.figure()
    plt.xlabel("Feature")
    plt.ylabel("Number of elements")
    plt.hist(D0[:, :], density=True, alpha=0.7, label="Male")
    plt.hist(D1[:, :], density=True, alpha=0.7, label="Female")
    plt.legend()
    path = f'Images/Dataset/General/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Dataset Features')
    plt.savefig(os.path.join(path, f'General.png'))
    plt.close()


def mixedPlot(D, L, m, type):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            plt.figure()
            plt.xlabel("Feature " + str(i + 1))
            plt.ylabel("Feature " + str(j + 1))
            plt.scatter(D0[i, :], D0[j, :], label="Male")
            plt.scatter(D1[i, :], D1[j, :], label="Female")
            plt.legend()
            # plt.show()
            path = f'Images/Dataset/Cross_{type}/Feature{i}/'
            os.makedirs(path, exist_ok=True)
            plt.title(f'Feature {i + 1}, {j + 1}')
            plt.savefig(os.path.join(path, f'Feature {i + 1}, {j + 1}.png'))
            plt.close()


# P=Pearson
def computeCorrelationP(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    diff_x = x - mean_x
    diff_y = y - mean_y
    diff_prod = diff_x * diff_y
    sum_diff_squares = np.sqrt(np.sum(diff_x ** 2) * np.sum(diff_y ** 2))
    correlation = np.sum(diff_prod) / sum_diff_squares
    return correlation


def correlationPlotP(data, labels, target_class):
    target_data = data[:, labels == target_class]
    non_target_data = data[:, labels != target_class]
    num_features = data.shape[0]
    correlations_target = np.zeros((num_features, num_features))
    correlations_non_target = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            correlations_target[i, j] = computeCorrelationP(target_data[:, i], target_data[:, j])
            correlations_non_target[i, j] = computeCorrelationP(non_target_data[:, i], non_target_data[:, j])

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    im1 = axs[0].matshow(correlations_target, cmap='coolwarm', vmin=-1, vmax=1)
    axs[0].set_title('Target Class')
    im2 = axs[1].matshow(correlations_non_target, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].set_title('Non-Target Class')
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    fig.suptitle('Pearson Correlation Coefficient')
    # plt.show()
    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Pearson Correlation')
    plt.savefig(os.path.join(path, 'PearsonCorrelation.png'))
    plt.close()


def maleFemaleFeaturesPlot(DTR, LTR, m=2, appendToTitle=''):
    correlationPlot(DTR, "Dataset" + appendToTitle, cmap="Greys")
    correlationPlot(DTR[:, LTR == 0], "Male" + appendToTitle, cmap="Blues")
    correlationPlot(DTR[:, LTR == 1], "Female" + appendToTitle, cmap="Reds")


def heatmapPlot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    hFea = {
        0: 'Feature_0',
        1: 'Feature_1',
        2: 'Feature_2',
        3: 'Feature_3',
        4: 'Feature_4',
        5: 'Feature_5',
        6: 'Feature_6',
        7: 'Feature_7',
        8: 'Feature_8',
        9: 'Feature_9',
        10: 'Feature_10',
        11: 'Feature_11',
        12: 'Feature_12',
    }

    corr_matrix = np.corrcoef(D1)

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(corr_matrix, cmap='seismic')
    plt.colorbar()

    ax.set_xticks(np.arange(len(corr_matrix)))
    ax.set_yticks(np.arange(len(corr_matrix)))
    ax.set_xticklabels(np.arange(len(corr_matrix)))
    ax.set_yticklabels(np.arange(len(corr_matrix)))

    # plt.title('Pearson Correlation Heatmap - \'Females\' training set')
    # plt.savefig('heatmap_training_set_authentic.png', dpi=300)
    # plt.show()

    plt.legend()
    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Pearson Heatmap V2')
    plt.savefig(os.path.join(path, f'Pearson Heatmap V2.png'), dpi=300)
    plt.close()


def computeCorrelation(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def correlationPlot(DTR, title, cmap):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = computeCorrelation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Features")

    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'Pearson_Heatmap_{title}.png'), dpi=300)
    plt.savefig("./Images/Dataset/" + title + ".svg")


# FOR PCA_LDA
def split_dataset(D, L, train_ratio=0.67, seed=0):
    """
    Splits the dataset into training and validation sets.

    Args:
    D (numpy.ndarray): The data matrix.
    L (numpy.ndarray): The label vector.
    train_ratio (float): The ratio of training samples.
    seed (int): The random seed.

    Returns:
    tuple: Training data, training labels, validation data, validation labels.
    """
    np.random.seed(seed)
    n_train = int(D.shape[1] * train_ratio)
    indices = np.random.permutation(D.shape[1])
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    D_train, L_train = D[:, train_idx], L[train_idx]
    D_val, L_val = D[:, val_idx], L[val_idx]
    return D_train, L_train, D_val, L_val


def center_data(D, mean=None):
    if mean is None:
        mean = D.mean(axis=1, keepdims=True)
    centered_data = D - mean
    return centered_data, mean


def data_analysis(DTR, LTR):
    # Print data statistics using panda's df
    data_statistics_light(DTR, LTR)
    # Additional Analysis
    DC, _ = center_data(DTR)
    plot_feature_pairs(DC, LTR, output_dir='Output/Dataset/CenteredFeaturePairPlots', origin='main')
    plot_feature_distributions(DC, LTR, output_dir='Output/Dataset/CenteredFeatureDistributions', origin='main')
    for i in range(2):
        print(f"Class {i}")
        variance = DTR.var(i)
        std_dev = DTR.std(i)
        print(f"Variance = {variance}")
        print(f"Standard Deviation = {std_dev}")
        print(f"Variance centered = {DC.var(i)}")
        print(f"Standard Deviation centered = {DC.std(i)}")
    # Compute mean, covariance matrix, variance, and standard deviation
    mean = compute_mean(DTR)
    covariance_matrix = compute_covariance_matrix(DTR)
    variance = compute_variance(DTR)
    std_dev = compute_standard_deviation(DTR)
    print("Mean:\n", mean)
    print("Covariance Matrix:\n", covariance_matrix)
    print("Variance:\n", variance)
    print("Standard Deviation:\n", std_dev)
    # Centered data statistics
    centered_variance = compute_variance(DC)
    centered_std_dev = compute_standard_deviation(DC)
    print("Centered Variance:\n", centered_variance)
    print("Centered Standard Deviation:\n", centered_std_dev)
    # Plot histograms and scatter plots
    plot_histograms(DTR, LTR)
    plot_pairwise_scatter(DTR, LTR)
    plot_histograms(DC, LTR, output_dir='Output/Dataset/CenteredHistograms')
    plot_pairwise_scatter(DC, LTR, output_dir='Output/Dataset/CenteredScatterPlots')


