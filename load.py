import os
import sys
import numpy as np
import string
import scipy.special
import itertools
import pandas as pd
from matplotlib import pyplot as plt


def load_data(file_path):
    """
    Load data from a CSV file and transpose the array to match the required format.
    
    Args:
    file_path: Path to the CSV data file.
    
    Returns:
    A tuple containing data as numpy arrays: (Features, Labels)
    """
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            splitted = line.strip().split(',')
            # Assume the last element is the label
            data.append([float(i) for i in splitted[:-1]])
            labels.append(int(splitted[-1]))
    
    # Convert to numpy arrays and transpose so that each column is a sample
    # and each row is a feature.
    D = np.array(data).T
    L = np.array(labels)
    
    return D, L

def loadTrainingAndTestData(train_path, test_path):
    """
    Load training and test data from files.
    
    Args:
    train_path: Path to the training data file.
    test_path: Path to the test data file.
    
    Returns:
    A tuple of tuples: ((Training data, Training labels), (Test data, Test labels))
    """
    DTR, LTR = load_data(train_path)
    DTE, LTE = load_data(test_path)
    
    return (DTR, LTR), (DTE, LTE)

def mcol(v):
    """
    Trasforma un array 1D in una colonna (2D).

    Args:
    v: Array 1D.

    Returns:
    Array 2D con una sola colonna.
    """
    return v.reshape((v.size, 1))

def data_statistics(DTR, LTR):
    # Ensure DTR and LTR are not transposed before being passed here
    columns = [f"Feature {i+1}" for i in range(DTR.shape[1])]
    df = pd.DataFrame(DTR, columns=columns)
    df['Label'] = LTR

    # Print basic statistics
    print("Basic Descriptive Statistics:")
    print(df.describe(include='all'))

    # Print class distribution
    class_counts = df['Label'].value_counts()
    print("\nClass Distribution:")
    print(class_counts)

    # Correlation matrix
    print("\nFeature Correlation Matrix:")
    print(df.corr())

    # Check for missing values
    missing_data = df.isnull().sum()
    print("\nMissing Data in Each Column:")
    print(missing_data)

def data_statistics_light(DTR, LTR):
    """
    Print basic information about the dataset without using Pandas DataFrame.
    
    Args:
    DTR (numpy.ndarray): Data matrix where each row is a sample and each column is a feature.
    LTR (numpy.ndarray): Label vector where each element is the class label for corresponding sample.
    """
    num_samples = DTR.shape[1]
    num_features = DTR.shape[0]
    num_genuine = np.sum(LTR == 1)
    num_fake = np.sum(LTR == 0)
    
    print("Basic Information:")
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print(f"Number of genuine (True, label 1) samples: {num_genuine}")
    print(f"Number of fake (False, label 0) samples: {num_fake}")

def loadFile(filename):
    """
    Carica i dati da un singolo file, trasformandoli in una matrice di attributi e un vettore di etichette.

    Args:
    filename: Percorso al file da caricare.

    Returns:
    Una tupla (Dati, Etichette), dove i dati sono in formato matrice e le etichette sono un array.
    """
    data_list = []
    labels_list = []
    # Mappatura delle etichette testuali a valori numerici
    labels_mapping = {
        'Male': 0,
        'Female': 1,
    }

    with open(filename) as file:
        for line in file:
            try:
                # Estrazione degli attributi e conversione in array numpy
                attributes = line.split(',')[0:4]
                attributes = mcol(np.array([float(i) for i in attributes]))
                # Estrazione dell'etichetta testuale e conversione in valore numerico
                label_name = line.split(',')[-1].strip()
                label = labels_mapping[label_name]
                data_list.append(attributes)
                labels_list.append(label)
            except ValueError:
                # Gestisce linee malformate ignorandole
                pass

    # Concatenazione degli array di attributi e conversione delle etichette in array numpy
    return np.hstack(data_list), np.array(labels_list, dtype=np.int32)


def plot_hist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    output_dir = "Old_Output/Output/Dataset"
    os.makedirs(output_dir, exist_ok=True)

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(f"Feature {dIdx}")
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='Fake')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='Genuine')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        filename = f"Histogram_Feature_{dIdx+1}"
        plt.savefig(os.path.join(output_dir, filename))



def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    output_dir = "Old_Output/Output/Dataset"
    os.makedirs(output_dir, exist_ok=True)

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(f"Feature {dIdx1}")
            plt.ylabel(f"Feature {dIdx2}")
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Fake')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Genuine')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure

            filename = f'Scatter_Features_{dIdx1+1}_{dIdx2+1}.pdf'
            plt.savefig(os.path.join(output_dir, filename))
