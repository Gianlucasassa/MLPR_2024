import scipy.linalg
from Preprocess.PCA import *


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def compute_class_means(D, L):

    classes = np.unique(L)
    means = []
    for cls in classes:
        class_data = D[:, L == cls]
        means.append(class_data.mean(axis=1).reshape(-1, 1))
    return means

def compute_Sb_Sw(D, L):

    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_projection_matrix(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, :m]

def apply_lda(D, P_lda):
    return P_lda.T @ D












def plot_lda_histogram(D, L, output_dir='Output/LDA_Histogram', origin='LDA'):
    W_lda = compute_lda_projection_matrix(D, L, 1)
    D_lda = apply_lda(D, W_lda)

    plt.figure()
    for cls in np.unique(L):
        plt.hist(D_lda[0, L == cls], bins=30, alpha=0.5, label=f'Class {cls}')
    plt.xlabel('LDA Component 1')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Histogram of LDA Component 1 ({origin})')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'LDA_Component_1_Histogram_{origin}.png'))
    plt.close()




def lda_classification(D_train, L_train, D_val, L_val, lda_dim=1):
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
    return (class_0.mean() + class_1.mean()) / 2.0

def classify(D_lda, threshold):
    P_val = np.zeros(D_lda.shape[1])
    P_val[D_lda[0] >= threshold] = 1
    P_val[D_lda[0] < threshold] = 0
    return P_val


def find_best_threshold(D_lda, L):
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
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold

    return best_threshold, best_error_rate
