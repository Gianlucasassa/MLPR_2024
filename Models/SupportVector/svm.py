import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os

import sklearn.datasets

from Models import bayesRisk
from sklearn.model_selection import train_test_split

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)



def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def center_data(DTR, DTE):
    mean_DTR = np.mean(DTR, axis=1, keepdims=True)
    DTR_centered = DTR - mean_DTR
    DTE_centered = DTE - mean_DTR
    return DTR_centered, DTE_centered

class SVMClassifier:
    def __init__(self, C=1.0, K=1.0, kernel='linear', degree=2, coef0=1, gamma=1.0, eps=1.0):
        self.C = C
        self.K = K
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.eps = eps
        self.w = None
        self.b = None
        self.alpha = None
        self.primal_loss = None
        self.dual_loss = None
        self.sv = None
        self.sv_y = None

    def linear_kernel(self, D1, D2):
        return np.dot(D1.T, D2)

    def poly_kernel(self, D1, D2):
        return (np.dot(D1.T, D2) + self.coef0) ** self.degree

    def rbf_kernel(self, D1, D2):
        D1_norms = (D1 ** 2).sum(0)
        D2_norms = (D2 ** 2).sum(0)
        Z = vcol(D1_norms) + vrow(D2_norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-self.gamma * Z)

    def get_kernel_func(self):
        if self.kernel == 'linear':
            return self.linear_kernel
        elif self.kernel == 'poly':
            return self.poly_kernel
        elif self.kernel == 'rbf':
            return self.rbf_kernel
        else:
            raise ValueError("Unsupported kernel")

    def train(self, DTR, LTR):
        ZTR = LTR * 2.0 - 1.0  # Convert labels to +1/-1
        kernel_func = self.get_kernel_func()

        if self.kernel == 'linear':
            DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * self.K])
            H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

            def fOpt(alpha):
                Ha = H @ vcol(alpha)
                loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
                grad = Ha.ravel() - np.ones(alpha.size)
                return loss, grad

            self.alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]),
                                                            bounds=[(0, self.C) for _ in range(DTR.shape[1])], factr=1.0)

            def primalLoss(w_hat):
                S = (vrow(w_hat) @ DTR_EXT).ravel()
                return 0.5 * np.linalg.norm(w_hat) ** 2 + self.C * np.maximum(0, 1 - ZTR * S).sum()

            # Compute primal solution for extended data matrix
            w_hat = (vrow(self.alpha) * vrow(ZTR) * DTR_EXT).sum(1)
            self.w, self.b = w_hat[:-1], w_hat[-1] * self.K  # b must be rescaled in case K != 1

            primal_loss = primalLoss(w_hat).item()
            dual_loss = -fOpt(self.alpha)[0].item()
            print(f'SVM - C {self.C:e} - K {self.K:e} - primal loss {primal_loss:e} - dual loss {dual_loss:e} - duality gap {primal_loss - dual_loss:e}')
        else:
            K = kernel_func(DTR, DTR) + self.eps
            H = vcol(ZTR) * vrow(ZTR) * K

            print(f'Dimensions of K: {K.shape}')
            print(f'Dimensions of H: {H.shape}')

            def fOpt(alpha):
                #print(f'Dimensions of alpha in fOpt: {alpha.shape}')
                Ha = H @ vcol(alpha)
                #print(f'Dimensions of Ha: {Ha.shape}')
                loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
                grad = Ha.ravel() - np.ones(alpha.size)
                return loss, grad

            initial_alpha = np.zeros(DTR.shape[1])
            self.alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, initial_alpha,
                                                            bounds=[(0, self.C) for _ in range(DTR.shape[1])], factr=1.0)

            print("Finished Optimizing")
            self.sv = DTR  # Use all training data as support vectors
            self.sv_y = ZTR
            K_sv = kernel_func(self.sv, self.sv)
            self.b = np.mean(self.sv_y - (self.alpha * self.sv_y) @ K_sv)

            self.primal_loss = self.compute_primal_loss(DTR, LTR, ZTR, H)
            self.dual_loss = -fOpt(self.alpha)[0].item()
            print(f'SVM - C {self.C:e} - Kernel {self.kernel} - dual loss {self.dual_loss:e}')

    def compute_primal_loss(self, DTR, LTR, ZTR, H):
        if self.kernel == 'linear':
            DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * self.K])
            S = (vrow(self.w) @ DTR_EXT).ravel()
            return 0.5 * np.linalg.norm(self.w) ** 2 + self.C * np.maximum(0, 1 - ZTR * S).sum()
        else:
            return self.dual_loss

    def predict(self, DTE):
        if self.kernel == 'linear':
            DTE_EXT = np.vstack([DTE, np.ones((1, DTE.shape[1])) * self.K])
            return (np.dot(vrow(self.w), DTE) + self.b * self.K).ravel()
        else:
            kernel_func = self.get_kernel_func()
            K = kernel_func(self.sv, DTE)
            return (self.alpha * self.sv_y) @ K + self.b


def test_svm_on_iris():
    D, L = load_iris_binary()

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    print("START IRIS - LINEAR KERNEL")
    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            classifier = SVMClassifier(C=C, K=K, kernel='linear')
            classifier.train(DTR, LTR)
            SVAL = classifier.predict(DTE)
            PVAL = (SVAL > 0).astype(int)
            err = (PVAL != LTE).sum() / float(LTE.size)

            print(f'Linear Kernel - K {K} - C {C:.1e}')
            print(f'Error rate: {err * 100:.1f}%')
            print(f'minDCF - pT = 0.5: {bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0):.4f}')
            print(f'actDCF - pT = 0.5: {bayesRisk.compute_actDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0):.4f}')
            print()

    print("START IRIS - POLY & RBF KERNELS")
    kernel_params_list = [
        {'kernel': 'poly', 'degree': 2, 'coef0': 0},
        {'kernel': 'poly', 'degree': 2, 'coef0': 1},
        {'kernel': 'rbf', 'gamma': 1.0},
        {'kernel': 'rbf', 'gamma': 10.0}
    ]

    for params in kernel_params_list:
        for eps in [0.0, 1.0]:
            kernel_type = params['kernel']
            params_copy = params.copy()
            del params_copy['kernel']
            classifier = SVMClassifier(C=1.0, kernel=kernel_type, eps=eps, **params_copy)
            classifier.train(DTR, LTR)
            SVAL = classifier.predict(DTE)
            PVAL = (SVAL > 0).astype(int)
            err = (PVAL != LTE).sum() / float(LTE.size)

            if kernel_type == 'poly':
                kernel_descr = f"Poly (d={params['degree']}, c={params['coef0']})"
            else:
                kernel_descr = f"RBF (gamma={params['gamma']})"

            print(f'{kernel_descr} - eps {eps}')
            print(f'Error rate: {err * 100:.1f}%')
            print(f'minDCF - pT = 0.5: {bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0):.4f}')
            print(f'actDCF - pT = 0.5: {bayesRisk.compute_actDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0):.4f}')
            print()

    print("END IRIS")

def plot_dcf_vs_c(C_values, actual_DCFs, min_DCFs, title, filename):
    plt.figure()
    plt.plot(C_values, actual_DCFs, label='Actual DCF')
    plt.plot(C_values, min_DCFs, label='Min DCF')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def evaluate_SVM(DTE, DTR, LTE, LTR, centered=False):
    output_dir_base = 'Output/SVM/linear'
    os.makedirs(output_dir_base, exist_ok=True)

    # Define C values and effective prior
    C_values = np.logspace(-5, 0, 11)
    effective_prior = 0.1
    Cfp = 1.0
    Cfn = 1.0

    if centered:
        DTR, DTE = center_data(DTR, DTE)

    # Arrays to store results
    actual_DCFs = []
    min_DCFs = []

    for C in C_values:
        classifier = SVMClassifier(C=C, kernel='linear')
        classifier.train(DTR, LTR)

        # Compute scores
        SVAL = classifier.predict(DTE)

        # Compute actual DCF
        actual_DCF = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LTE, effective_prior,
                                                                                         Cfn, Cfp)

        # Compute minDCF
        min_DCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, effective_prior, Cfn, Cfp)

        actual_DCFs.append(actual_DCF)
        min_DCFs.append(min_DCF)

    # Plotting the results
    title = 'DCF vs C (Centered)' if centered else 'DCF vs C'
    filename = os.path.join(output_dir_base, "dcf_vs_C_centered.png" if centered else "dcf_vs_C.png")
    plot_dcf_vs_c(C_values, actual_DCFs, min_DCFs, title, filename)

    return actual_DCFs, min_DCFs


def evaluate_SVM_poly(DTE, DTR, LTE, LTR):
    output_dir_base = 'Output/SVM/poly'
    os.makedirs(output_dir_base, exist_ok=True)

    C_values = np.logspace(-5, 0, 11)
    actual_DCFs = []
    min_DCFs = []

    for C in C_values:
        classifier = SVMClassifier(C=C, kernel='poly', degree=2, coef0=1)
        classifier.train(DTR, LTR)
        SVAL = classifier.predict(DTE)

        actual_DCF = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LTE, 0.5, 1.0, 1.0)
        min_DCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0)

        actual_DCFs.append(actual_DCF)
        min_DCFs.append(min_DCF)

    print("START POLY KERNEL")
    for i, C in enumerate(C_values):
        print(f'Polynomial Kernel - C {C:.4e} - minDCF: {min_DCFs[i]:.4f} - actDCF: {actual_DCFs[i]:.4f}')
    print("END POLY KERNEL")

    filename = os.path.join(output_dir_base, 'dcf_vs_C_poly.png')
    plot_dcf_vs_c(C_values, actual_DCFs, min_DCFs, 'DCF vs C (Polynomial Kernel)', filename)


def evaluate_SVM_rbf(DTE, DTR, LTE, LTR):
    output_dir_base = 'Output/SVM/rbf'
    os.makedirs(output_dir_base, exist_ok=True)

    C_values = np.logspace(-3, 2, 11)
    gamma_values = np.logspace(-4, -1, 4)

    actual_DCFs_all = {gamma: [] for gamma in gamma_values}
    min_DCFs_all = {gamma: [] for gamma in gamma_values}

    for gamma in gamma_values:
        actual_DCFs = []
        min_DCFs = []
        for C in C_values:
            classifier = SVMClassifier(C=C, kernel='rbf', gamma=gamma, eps=1.0)
            classifier.train(DTR, LTR)
            SVAL = classifier.predict(DTE)

            actual_DCF = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LTE, 0.5, 1.0, 1.0)
            min_DCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0)

            actual_DCFs.append(actual_DCF)
            min_DCFs.append(min_DCF)

        actual_DCFs_all[gamma] = actual_DCFs
        min_DCFs_all[gamma] = min_DCFs

        print("START RBF KERNEL")
        for i, C in enumerate(C_values):
            print(f'RBF Kernel - gamma {gamma:.4e} - C {C:.4e} - minDCF: {min_DCFs[i]:.4f} - actDCF: {actual_DCFs[i]:.4f}')
        print("END RBF KERNEL")

    # Plotting the results
    plt.figure()
    for gamma in gamma_values:
        plt.plot(C_values, actual_DCFs_all[gamma], label=f'Actual DCF (gamma={gamma:.4e})')
    plt.xscale('log')
    #plt.ylim(0, 1)
    plt.xlabel('C')
    plt.ylabel('Actual DCF')
    plt.title('Actual DCF vs C (RBF Kernel)')
    plt.legend()
    plt.savefig(os.path.join(output_dir_base, 'actual_dcf_vs_C_rbf.png'))
    plt.close()

    plt.figure()
    for gamma in gamma_values:
        plt.plot(C_values, min_DCFs_all[gamma], label=f'Min DCF (gamma={gamma:.4e})')
    plt.xscale('log')
    #plt.ylim(0, 1) TODO: you can try this
    plt.xlabel('C')
    plt.ylabel('Min DCF')

    plt.title('Min DCF vs C (RBF Kernel)')
    plt.legend()
    plt.savefig(os.path.join(output_dir_base, 'min_dcf_vs_C_rbf.png'))
    plt.close()


def evaluate_SVM_poly_optional(DTE, DTR, LTE, LTR):
    output_dir_base = 'Output/SVM/poly_optional'
    os.makedirs(output_dir_base, exist_ok=True)

    C_values = np.logspace(-5, 0, 11)
    actual_DCFs = []
    min_DCFs = []

    for C in C_values:
        classifier = SVMClassifier(C=C, kernel='poly', degree=4, coef0=1, eps=0.0)
        classifier.train(DTR, LTR)
        SVAL = classifier.predict(DTE)

        actual_DCF = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LTE, 0.5, 1.0, 1.0)
        min_DCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LTE, 0.5, 1.0, 1.0)

        actual_DCFs.append(actual_DCF)
        min_DCFs.append(min_DCF)

    print("START POLY KERNEL OPTIONAL")
    for i, C in enumerate(C_values):
        print(f'Polynomial Kernel (d=4, c=1) - C {C:.4e} - minDCF: {min_DCFs[i]:.4f} - actDCF: {actual_DCFs[i]:.4f}')
    print("END POLY KERNEL OPTIONAL")

    filename = os.path.join(output_dir_base, 'dcf_vs_C_poly_optional.png')
    plot_dcf_vs_c(C_values, actual_DCFs, min_DCFs, 'DCF vs C (Polynomial Kernel d=4, c=1)', filename)

    # Analysis of the features and quadratic separation surfaces
    # Consider only the last two features of each sample
    DTR_2d = DTR[-2:, :]
    DTE_2d = DTE[-2:, :]

    # Transform features using the degree 2 polynomial kernel mapping
    ZTR = np.vstack([DTR_2d[0] * DTR_2d[1], DTR_2d[0] ** 2, DTR_2d[1] ** 2])
    ZTE = np.vstack([DTE_2d[0] * DTE_2d[1], DTE_2d[0] ** 2, DTE_2d[1] ** 2])

    plt.figure()
    for label in np.unique(LTR):
        plt.scatter(ZTR[0, LTR == label], ZTR[1, LTR == label], label=f'Class {label}', alpha=0.7)
    plt.title('Transformed Features using Polynomial Kernel (d=2)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir_base, 'transformed_features_poly_kernel.png'))
    plt.close()
def print_dcf_results(name, actual_DCFs, min_DCFs, C_values):
    print(f"Evaluating {name}")
    for i, C in enumerate(C_values):
        print(f"{name} - C: {C:.4e}")
        print(f"DCF (non-normalized): {actual_DCFs[i]:.3f}")
        print(f"MinDCF (normalized, fast): {min_DCFs[i]:.3f}")
        print()


def reduce_dataset(D, L, fraction):
    np.random.seed(0)
    idx = np.random.permutation(D.shape[1])
    reduced_size = int(D.shape[1] * fraction)
    idx_reduced = idx[:reduced_size]
    return D[:, idx_reduced], L[idx_reduced]


def train_SVM(DTE, DTR, LTE, LTR): #TODO: finished, mainly working, poor results, optional part to be understood
    # Call the function to test SVM on Iris dataset
    test_svm_on_iris()

    # fraction = 0.1
    # DTR, LTR = reduce_dataset(DTR, LTR, fraction)
    # DTE, LTE = reduce_dataset(DTE, LTE, fraction)

    # Evaluate with original data
    actual_DCFs, min_DCFs = evaluate_SVM(DTE, DTR, LTE, LTR, centered=False)
    print_dcf_results("Standard SVM", actual_DCFs, min_DCFs, np.logspace(-5, 0, 11))

    # Evaluate with centered data
    actual_DCFs_centered, min_DCFs_centered = evaluate_SVM(DTE, DTR, LTE, LTR, centered=True)
    print_dcf_results("Centered SVM", actual_DCFs_centered, min_DCFs_centered, np.logspace(-5, 0, 11))

    # Evaluate polynomial kernel
    evaluate_SVM_poly(DTE, DTR, LTE, LTR)

    evaluate_SVM_rbf(DTE, DTR, LTE, LTR)

    evaluate_SVM_poly_optional(DTE, DTR, LTE, LTR)

    return

# import os
#
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.optimize import fmin_l_bfgs_b
#
# from Models.Gaussian.gaussian_models import compute_bayes_risk, compute_normalized_dcf, \
#     compute_confusion_matrix
# from Preprocess.DatasetPlots import center_data
#
# import scipy.optimize
# from sklearn.metrics import roc_curve, auc
#
# # class SVMClassifier:
# #     def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=2, coef0=1.0):
# #         self.kernel = kernel
# #         self.C = C
# #         self.gamma = gamma
# #         self.degree = degree
# #         self.coef0 = coef0
# #         self.alpha = None
# #         self.support_vectors = None
# #         self.support_vector_labels = None
# #         self.b = 0
# #         self.w = None
# #
# #     def linear_kernel(self, x1, x2):
# #         return np.dot(x1.T, x2)
# #
# #     def polynomial_kernel(self, x1, x2):
# #         return (np.dot(x1.T, x2) + self.coef0) ** self.degree
# #
# #     def rbf_kernel(self, x1, x2):
# #         return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
# #
# #     def compute_kernel_matrix(self, X):
# #         n_samples = X.shape[1]
# #         K = np.zeros((n_samples, n_samples))
# #         for i in range(n_samples):
# #             for j in range(n_samples):
# #                 if self.kernel == 'linear':
# #                     K[i, j] = self.linear_kernel(X[:, i], X[:, j])
# #                 elif self.kernel == 'poly':
# #                     K[i, j] = self.polynomial_kernel(X[:, i], X[:, j])
# #                 elif self.kernel == 'rbf':
# #                     K[i, j] = self.rbf_kernel(X[:, i], X[:, j])
# #         return K
# #
# #     def dual_objective(self, alpha, H):
# #         return 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.sum(alpha)
# #
# #     def dual_objective_grad(self, alpha, H):
# #         return np.dot(H, alpha) - np.ones_like(alpha)
# #
# #     def train(self, DTR, LTR):
# #         n_samples = DTR.shape[1]
# #         self.LTR = LTR
# #         K = self.compute_kernel_matrix(DTR)
# #         H = np.outer(LTR, LTR) * K
# #
# #         # Constraints
# #         bounds = [(0, self.C) for _ in range(n_samples)]
# #         constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, LTR)}
# #
# #         # Solve the quadratic programming problem
# #         result = scipy.optimize.minimize(
# #             fun=lambda alpha: self.dual_objective(alpha, H),
# #             x0=np.zeros(n_samples),
# #             jac=lambda alpha: self.dual_objective_grad(alpha, H),
# #             bounds=bounds,
# #             constraints=constraints
# #         )
# #
# #         self.alpha = result.x
# #         sv = self.alpha > 1e-5
# #         self.support_vectors = DTR[:, sv]
# #         self.support_vector_labels = LTR[sv]
# #         self.alpha = self.alpha[sv]
# #
# #         if self.kernel == 'linear':
# #             self.w = np.sum(self.alpha * self.support_vector_labels * self.support_vectors, axis=1)
# #
# #         self.b = np.mean(self.support_vector_labels - np.sum((self.alpha * self.support_vector_labels)[:, None] * K[sv][:, sv], axis=0))
# #
# #     def project(self, D):
# #         if self.kernel == 'linear':
# #             return np.dot(self.w.T, D) + self.b
# #         else:
# #             K = np.array([self.compute_kernel(self.support_vectors, x) for x in D.T])
# #             return np.dot((self.alpha * self.support_vector_labels), K) + self.b
# #
# #     def predict(self, D):
# #         return np.sign(self.project(D))
# #
# #     def compute_predictions(self, scores):
# #         return (scores > 0).astype(int)
# #
# #     def compute_error_rate(self, predictions, true_labels):
# #         return np.mean(predictions != true_labels)
# #
# #     def compute_dcf(self, scores, labels, pi_t):
# #         thresholds = np.sort(scores)
# #         min_dcf = float('inf')
# #         for t in thresholds:
# #             predictions = (scores >= t).astype(int)
# #             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
# #             if dcf < min_dcf:
# #                 min_dcf = dcf
# #         return min_dcf
# #
# #     def compute_min_dcf(self, scores, labels, pi_t):
# #         thresholds = np.sort(scores)
# #         min_dcf = float('inf')
# #         for t in thresholds:
# #             predictions = (scores >= t).astype(int)
# #             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
# #             if dcf < min_dcf:
# #                 min_dcf = dcf
# #         return min_dcf
# #
# #     def compute_dcf_at_threshold(self, predictions, labels, pi_t):
# #         fnr = np.mean(predictions[labels == 1] == 0)
# #         fpr = np.mean(predictions[labels == 0] == 1)
# #         dcf = pi_t * fnr + (1 - pi_t) * fpr
# #         return dcf
# #
# #     def plot_roc_curve(self, llrs, labels, output_file='roc_curve.png'):
# #         fpr, tpr, _ = roc_curve(labels, llrs)
# #         roc_auc = auc(fpr, tpr)
# #         plt.figure()
# #         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# #         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# #         plt.xlim([0.0, 1.0])
# #         plt.ylim([0.0, 1.05])
# #         plt.xlabel('False Positive Rate')
# #         plt.ylabel('True Positive Rate')
# #         plt.title('Receiver Operating Characteristic')
# #         plt.legend(loc="lower right")
# #         plt.savefig(output_file)
# #         plt.close()
# #
# #     def plot_dcf(self, lambdas, dcf_values, min_dcf_values, output_file='dcf_plot.png'):
# #         plt.figure()
# #         plt.plot(lambdas, dcf_values, label='Actual DCF')
# #         plt.plot(lambdas, min_dcf_values, label='Min DCF')
# #         plt.xscale('log')
# #         plt.xlabel('Lambda')
# #         plt.ylabel('DCF')
# #         plt.legend()
# #         plt.grid()
# #         plt.savefig(output_file)
# #         plt.close()
# #
# #     def evaluate_model(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir):
# #         os.makedirs(output_dir, exist_ok=True)
# #         dcf_values = []
# #         min_dcf_values = []
# #
# #         for l in lambdas:
# #             self.C = l
# #             self.train(DTR, LTR)
# #             scores = self.project(DTE)
# #             dcf = self.compute_dcf(scores, LTE, pi_t)
# #             min_dcf = self.compute_min_dcf(scores, LTE, pi_t)
# #             dcf_values.append(dcf)
# #             min_dcf_values.append(min_dcf)
# #             error_rate = self.compute_error_rate(self.compute_predictions(scores), LTE)
# #             print(f"Lambda: {l}, DCF: {dcf}, Min DCF: {min_dcf}, Error Rate: {error_rate}")
# #
# #         self.plot_dcf(lambdas, dcf_values, min_dcf_values, os.path.join(output_dir, 'dcf_plot.png'))
# #
# # def compute_dcf(scores, labels, pi_t):
# #     thresholds = np.sort(scores)
# #     min_dcf = float('inf')
# #     for t in thresholds:
# #         predictions = (scores >= t).astype(int)
# #         fnr = np.mean(predictions[labels == 1] == 0)
# #         fpr = np.mean(predictions[labels == 0] == 1)
# #         dcf = pi_t * fnr + (1 - pi_t) * fpr
# #         if dcf < min_dcf:
# #             min_dcf = dcf
# #     return min_dcf
# #
# # def compute_min_dcf(scores, labels, pi_t):
# #     thresholds = np.sort(scores)
# #     min_dcf = float('inf')
# #     for t in thresholds:
# #         predictions = (scores >= t).astype(int)
# #         fnr = np.mean(predictions[labels == 1] == 0)
# #         fpr = np.mean(predictions[labels == 0] == 1)
# #         dcf = pi_t * fnr + (1 - pi_t) * fpr
# #         if dcf < min_dcf:
# #             min_dcf = dcf
# #     return min_dcf
# #
# #
# #
# #
# #
# # def linear_svm_primal(DTR, LTR, C):
# #     n, m = DTR.shape
# #     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
# #     D_hat = np.vstack([DTR, np.ones(m)])
# #
# #     def primal_objective(w_hat):
# #         w = w_hat[:-1]
# #         b = w_hat[-1]
# #         margin = Z * (np.dot(DTR.T, w) + b)
# #         hinge_loss = np.maximum(0, 1 - margin)
# #         return 0.5 * np.linalg.norm(w) ** 2 + C * np.sum(hinge_loss)
# #
# #     w_hat0 = np.zeros(n + 1)
# #     w_hat, _, _ = fmin_l_bfgs_b(primal_objective, w_hat0, approx_grad=True)
# #
# #     return w_hat[:-1], w_hat[-1]
# #
# #
# # def linear_svm_dual(DTR, LTR, C):
# #     n, m = DTR.shape
# #     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
# #     H = (Z @ Z.T) * (DTR.T @ DTR)
# #
# #     def dual_objective(alpha):
# #         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
# #
# #     def dual_gradient(alpha):
# #         return np.dot(H, alpha) - np.ones(m)
# #
# #     bounds = [(0, C) for _ in range(m)]
# #     constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, Z), 'jac': lambda alpha: Z.flatten()}
# #     alpha0 = np.zeros(m)
# #     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
# #
# #     w_star = np.sum((alpha * Z.flatten()).reshape(-1, 1) * DTR.T, axis=0)
# #     support_vector_indices = np.where(alpha > 1e-6)[0]
# #     b_star = np.mean(Z[support_vector_indices] - np.dot(DTR.T[support_vector_indices], w_star))
# #
# #     return w_star, b_star
# #
# #
# # def svm_evaluate(DTE, w, b):
# #     scores = np.dot(DTE.T, w) + b
# #     return scores
# #
# #
# # def polynomial_kernel_svm(DTR, LTR, DTE, LTE, degree=2, c=1):
# #     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
# #     H = np.dot(Z, Z.T) * (np.dot(DTR.T, DTR) + c) ** degree
# #
# #     def dual_objective(alpha):
# #         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
# #
# #     def dual_gradient(alpha):
# #         return np.dot(H, alpha) - np.ones(len(alpha))
# #
# #     bounds = [(0, c) for _ in range(len(LTR))]
# #     alpha0 = np.zeros(len(LTR))
# #     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
# #
# #     def svm_score(x):
# #         return np.sum(alpha * Z.flatten() * (np.dot(DTR.T, x) + c) ** degree)
# #
# #     scores = np.array([svm_score(x) for x in DTE.T])
# #
# #     return scores
# #
# #
# # def rbf_kernel_svm(DTR, LTR, DTE, LTE, gamma=1.0, C=1.0):
# #     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
# #     K = np.exp(-gamma * np.sum((DTR[:, :, None] - DTR[:, None, :]) ** 2, axis=0))
# #     H = np.dot(Z, Z.T) * K
# #
# #     def dual_objective(alpha):
# #         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
# #
# #     def dual_gradient(alpha):
# #         return np.dot(H, alpha) - np.ones(len(alpha))
# #
# #     bounds = [(0, C) for _ in range(len(LTR))]
# #     alpha0 = np.zeros(len(LTR))
# #     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
# #
# #     def svm_score(x):
# #         k = np.exp(-gamma * np.sum((DTR - x.reshape(-1, 1)) ** 2, axis=0))
# #         return np.sum(alpha * Z.flatten() * k)
# #
# #     scores = np.array([svm_score(x) for x in DTE.T])
# #
# #     return scores
# #
# #
# # def plot_dcf_vs_c(Cs, min_dcf, act_dcf, model_name, filename):
# #     os.makedirs("Output/svm", exist_ok=True)
# #     plt.plot(Cs, min_dcf, label='Min DCF')
# #     plt.plot(Cs, act_dcf, label='Actual DCF')
# #     plt.xscale('log')
# #     plt.xlabel('C')
# #     plt.ylabel('DCF')
# #     plt.title(f'{model_name} DCF vs. C')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.savefig(os.path.join("Output/svm", filename))
# #     plt.close()
# #
# #
# # # Function for Linear SVM
# # def run_linear_svm(DTR, LTR, DTE, LTE, Cs, centered=False):
# #     min_dcf = []
# #     act_dcf = []
# #
# #     if centered:
# #         DTR, mean = center_data(DTR)
# #         DTE, _ = center_data(DTE, mean)
# #
# #     for C in Cs:
# #         w, b = linear_svm_dual(DTR, LTR, C)
# #         scores = svm_evaluate(DTE, w, b)
# #         optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
# #         confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
# #         bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
# #         norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
# #
# #         min_dcf.append(norm_dcf)
# #         act_dcf.append(compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
# #
# #     label = 'Centered Linear SVM' if centered else 'Linear SVM'
# #     filename = 'centered_linear_svm_dcf.png' if centered else 'linear_svm_dcf.png'
# #     plot_dcf_vs_c(Cs, min_dcf, act_dcf, label, filename)
# #
# #
# # # Function for Polynomial SVM
# # def run_polynomial_svm(DTR, LTR, DTE, LTE, Cs, degree=2, c=1):
# #     min_dcf = []
# #     act_dcf = []
# #
# #     for C in Cs:
# #         scores = polynomial_kernel_svm(DTR, LTR, DTE, LTE, degree=degree, c=c)
# #         optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
# #         confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
# #         bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
# #         norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
# #
# #         min_dcf.append(norm_dcf)
# #         act_dcf.append(compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
# #
# #     plot_dcf_vs_c(Cs, min_dcf, act_dcf, 'Polynomial SVM', 'poly_svm_dcf.png')
# #
# #
# # # Function for RBF SVM
# # def run_rbf_svm(DTR, LTR, DTE, LTE, Cs, gammas):
# #     for gamma in gammas:
# #         min_dcf_rbf_svm = []
# #         act_dcf_rbf_svm = []
# #
# #         for C in Cs:
# #             scores = rbf_kernel_svm(DTR, LTR, DTE, LTE, gamma=gamma, C=C)
# #             optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
# #             confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
# #             bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
# #             norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
# #
# #             min_dcf_rbf_svm.append(norm_dcf)
# #             act_dcf_rbf_svm.append(
# #                 compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
# #
# #         plot_dcf_vs_c(Cs, min_dcf_rbf_svm, act_dcf_rbf_svm, f'RBF SVM (γ={gamma})', f'rbf_svm_dcf_{gamma}.png')
#
#
# from scipy.optimize import minimize
#
# from sklearn.metrics import roc_curve, auc
#
# def normalize_data(D):
#     mean = np.mean(D, axis=1, keepdims=True)
#     std = np.std(D, axis=1, keepdims=True)
#     normalized_D = (D - mean) / std
#     print(f"Data normalized: mean={np.mean(normalized_D)}, std={np.std(normalized_D)}")
#     return normalized_D
#
#
# def center_data(D):
#     mean = np.mean(D, axis=1, keepdims=True)
#     centered_D = D - mean
#     print(f"Data centered: mean={np.mean(centered_D)}")
#     return centered_D, mean
#
#
# def vcol(x):
#     return x.reshape((x.size, 1))
#
#
# def vrow(x):
#     return x.reshape((1, x.size))
#
# def polyKernel(degree, c):
#     def polyKernelFunc(D1, D2):
#         return (np.dot(D1.T, D2) + c) ** degree
#     return polyKernelFunc
#
# def rbfKernel(gamma):
#     def rbfKernelFunc(D1, D2):
#         D1Norms = (D1 ** 2).sum(0)
#         D2Norms = (D2 ** 2).sum(0)
#         Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
#         return np.exp(-gamma * Z)
#     return rbfKernelFunc
#
# def train_dual_SVM_linear(DTR, LTR, C, K=1):
#     ZTR = LTR * 2.0 - 1.0
#     DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
#     H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)
#
#     def fOpt(alpha):
#         Ha = H @ vcol(alpha)
#         loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
#         grad = Ha.ravel() - np.ones(alpha.size)
#         return loss, grad
#
#     alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds=[(0, C) for _ in LTR], factr=1.0)
#
#     def primalLoss(w_hat):
#         S = (vrow(w_hat) @ DTR_EXT).ravel()
#         return 0.5 * np.linalg.norm(w_hat) ** 2 + C * np.maximum(0, 1 - ZTR * S).sum()
#
#     w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
#     w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
#
#     primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
#     print('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
#
#     return w, b
#
# def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
#     ZTR = LTR * 2.0 - 1.0
#     K = kernelFunc(DTR, DTR) + eps
#     H = vcol(ZTR) * vrow(ZTR) * K
#
#     def fOpt(alpha):
#         Ha = H @ vcol(alpha)
#         loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
#         grad = Ha.ravel() - np.ones(alpha.size)
#         return loss, grad
#
#     alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds=[(0, C) for _ in LTR], factr=1.0)
#     print('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))
#
#     def fScore(DTE):
#         K = kernelFunc(DTR, DTE) + eps
#         H = vcol(alphaStar) * vcol(ZTR) * K
#         return H.sum(0)
#
#     return fScore
#
#
# def compute_confusion_matrix(predictedLabels, classLabels):
#     nClasses = classLabels.max() + 1
#     M = np.zeros((nClasses, nClasses), dtype=np.int32)
#     for i in range(classLabels.size):
#         M[predictedLabels[i], classLabels[i]] += 1
#     return M
#
#
# def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
#     M = compute_confusion_matrix(predictedLabels, classLabels)  # Confusion matrix
#     Pfn = M[0, 1] / (M[0, 1] + M[1, 1])
#     Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
#     bayesError = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp
#     if normalize:
#         return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
#     return bayesError
#
#
# def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
#     th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
#     return np.int32(llr > th)
#
#
# def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
#     predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
#     return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)
#
#
# def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
#     llrSorter = np.argsort(llr)
#     llrSorted = llr[llrSorter]  # We sort the llrs
#     classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs
#
#     Pfp = []
#     Pfn = []
#
#     nTrue = (classLabelsSorted == 1).sum()
#     nFalse = (classLabelsSorted == 0).sum()
#     nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
#     nFalsePositive = nFalse
#
#     Pfn.append(nFalseNegative / nTrue)
#     Pfp.append(nFalsePositive / nFalse)
#
#     for idx in range(len(llrSorted)):
#         if classLabelsSorted[idx] == 1:
#             nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
#         if classLabelsSorted[idx] == 0:
#             nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
#         Pfn.append(nFalseNegative / nTrue)
#         Pfp.append(nFalsePositive / nFalse)
#
#     llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])
#
#     PfnOut = []
#     PfpOut = []
#     thresholdsOut = []
#     for idx in range(len(llrSorted)):
#         if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
#             idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
#             PfnOut.append(Pfn[idx])
#             PfpOut.append(Pfp[idx])
#             thresholdsOut.append(llrSorted[idx])
#
#     return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)  # we return also the corresponding thresholds
#
#
# def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
#     Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
#     minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (
#                 1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
#     idx = np.argmin(minDCF)
#     if returnThreshold:
#         return minDCF[idx], th[idx]
#     else:
#         return minDCF[idx]
#
#
# compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions
#
# def reduce_dataset(D, L, fraction=0.1, seed=42):
#     np.random.seed(seed)
#     n_samples = int(D.shape[1] * fraction)
#     indices = np.random.choice(D.shape[1], n_samples, replace=False)
#     print(f"Taken {n_samples} samples out of {D.shape[1]}")
#     return D[:, indices], L[indices]
#
#
# class SVMClassifier:
#     def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=2, coef0=1.0):
#         self.kernel = kernel
#         self.C = C
#         self.gamma = gamma
#         self.degree = degree
#         self.coef0 = coef0
#         self.alpha = None
#         self.support_vectors = None
#         self.support_vector_labels = None
#         self.b = 0
#         self.w = None
#
#     def linear_kernel(self, x1, x2):
#         return np.dot(x1.T, x2)
#
#     def polynomial_kernel(self, x1, x2):
#         return (np.dot(x1.T, x2) + self.coef0) ** self.degree
#
#     def rbf_kernel(self, x1, x2):
#         D1Norms = (x1 ** 2).sum(0)
#         D2Norms = (x2 ** 2).sum(0)
#         Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(x1.T, x2)
#         return np.exp(-self.gamma * Z)
#
#     def compute_kernel(self, X1, X2):
#         if self.kernel == 'linear':
#             return self.linear_kernel(X1, X2)
#         elif self.kernel == 'poly':
#             return self.polynomial_kernel(X1, X2)
#         elif self.kernel == 'rbf':
#             return self.rbf_kernel(X1, X2)
#         else:
#             raise ValueError("Unsupported kernel type")
#
#     def train(self, DTR, LTR):
#         if self.kernel == 'linear':
#             self.w, self.b = train_dual_SVM_linear(DTR, LTR, self.C)
#         elif self.kernel == 'poly':
#             kernelFunc = polyKernel(self.degree, self.coef0)
#             self.fScore = train_dual_SVM_kernel(DTR, LTR, self.C, kernelFunc)
#         elif self.kernel == 'rbf':
#             kernelFunc = rbfKernel(self.gamma)
#             self.fScore = train_dual_SVM_kernel(DTR, LTR, self.C, kernelFunc)
#         else:
#             raise ValueError("Unsupported kernel type")
#
#     def project(self, D):
#         if self.kernel == 'linear':
#             return np.dot(self.w.T, D) + self.b
#         else:
#             return self.fScore(D)
#
#     def predict(self, D):
#         return np.sign(self.project(D))
#
#     def compute_predictions(self, scores):
#         return (scores > 0).astype(int)
#
#     def compute_error_rate(self, predictions, true_labels):
#         return np.mean(predictions != true_labels)
#
#     def compute_dcf(self, scores, labels, pi_t):
#         thresholds = np.sort(scores)
#         actual_dcf = float('inf')
#         for t in thresholds:
#             predictions = (scores >= t).astype(int)
#             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
#             if dcf < actual_dcf:
#                 actual_dcf = dcf
#         return actual_dcf
#
#     def compute_min_dcf(self, scores, labels, pi_t):
#         thresholds = np.sort(scores)
#         min_dcf = float('inf')
#         for t in thresholds:
#             predictions = (scores >= t).astype(int)
#             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
#             if dcf < min_dcf:
#                 min_dcf = dcf
#         return min_dcf
#
#     def compute_dcf_at_threshold(self, predictions, labels, pi_t):
#         fnr = np.mean(predictions[labels == 1] == 0)
#         fpr = np.mean(predictions[labels == 0] == 1)
#         dcf = pi_t * fnr + (1 - pi_t) * fpr
#         return dcf
#
#     def plot_roc_curve(self, llrs, labels, output_file='roc_curve.png'):
#         fpr, tpr, _ = roc_curve(labels, llrs)
#         roc_auc = auc(fpr, tpr)
#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic')
#         plt.legend(loc="lower right")
#         plt.savefig(output_file)
#         plt.close()
#
#     def plot_dcf(self, lambdas, dcf_values, min_dcf_values, act_dcf_values, output_path, title):
#         plt.figure()
#         plt.plot(lambdas, dcf_values, label='Actual DCF')
#         plt.plot(lambdas, min_dcf_values, label='Min DCF')
#         plt.plot(lambdas, act_dcf_values, label='Act DCF')
#         plt.xscale('log')
#         plt.xlabel('Lambda')
#         plt.ylabel('DCF')
#         plt.title(title)
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(output_path)
#         plt.close()
#
#     def evaluate_model(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type, gamma=None):
#         os.makedirs(output_dir, exist_ok=True)
#
#         dcf_values = []
#         min_dcf_values = []
#         act_dcf_values = []
#         successful_lambdas = []
#
#         for l in lambdas:
#             self.C = l
#             if gamma is not None:
#                 self.gamma = gamma
#             try:
#                 self.train(DTR, LTR)
#                 #print(f"Training with C={l}, number of support vectors: {len(self.support_vector_labels)}")
#                 scores = self.project(DTE)
#                 dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, pi_t, 1, 1, normalize=True)
#                 min_dcf = compute_minDCF_binary_fast(scores, LTE, pi_t, 1, 1)
#                 act_dcf = compute_actDCF_binary_fast(scores, LTE, pi_t, 1, 1)
#                 dcf_values.append(dcf)
#                 min_dcf_values.append(min_dcf)
#                 act_dcf_values.append(act_dcf)
#                 successful_lambdas.append(l)
#                 error_rate = self.compute_error_rate(self.compute_predictions(scores), LTE)
#                 print(f"Lambda: {l}, DCF: {dcf}, Min DCF: {min_dcf}, Act DCF: {act_dcf}, Error Rate: {error_rate}")
#             except ValueError as e:
#                 print(f"Skipping C={l} due to error: {e}")
#
#         plot_title = f'{kernel_type.capitalize()} SVM DCF vs Lambda'
#         if gamma is not None:
#             plot_title += f' (gamma={gamma})'
#         self.plot_dcf(successful_lambdas, dcf_values, min_dcf_values, act_dcf_values,
#                       os.path.join(output_dir, f'dcf_plot_{kernel_type}{"_gamma_" + str(gamma) if gamma else ""}.png'),
#                       plot_title)
#
#
#     def evaluate_model_polynomial(self, DTR, LTR, DTE, LTE, lambdas, degree, coef0, pi_t, output_dir):
#         self.kernel = 'poly'
#         self.degree = degree
#         self.coef0 = coef0
#         self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type='poly')
#
#
#     def evaluate_model_rbf(self, DTR, LTR, DTE, LTE, lambdas, gammas, pi_t, output_dir):
#         self.kernel = 'rbf'
#         for gamma in gammas:
#             self.gamma = gamma
#             gamma_output_dir = os.path.join(output_dir, f'gamma_{gamma}')
#             self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, gamma_output_dir, kernel_type='rbf', gamma=gamma)
#
#     def evaluate_model_linear(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir):
#         self.kernel = 'linear'
#         self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type='linear')
#
# def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
#     t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
#     return (llrs >= t).astype(int)
#
#
# def train_SVM(DTE, DTR, LTE, LTR):
#
#     #fraction = 0.1  # Use 10% of the data for initial testing
#     #DTR, LTR = reduce_dataset(DTR, LTR, fraction)
#     lambdas = np.logspace(-5, 0, 11)
#     pi_t = 0.1
#     output_dir = 'Output/SVM_Results'
#
#     # Normalize the data
#     DTR_normalized = normalize_data(DTR)
#     DTE_normalized = normalize_data(DTE)
#
#     # Center the data
#     DTR_centered, _ = center_data(DTR)
#     DTE_centered, _ = center_data(DTE)
#
#     # Linear SVM
#     print("Evaluating Linear SVM")
#     svm_classifier = SVMClassifier(kernel='linear')
#     svm_classifier.evaluate_model_linear(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, pi_t, os.path.join(output_dir, 'linear'))
#     print("Evaluating Linear SVM - centered")
#     svm_classifier.evaluate_model_linear(DTR_centered, LTR, DTE_centered, LTE, lambdas, pi_t,
#                                          os.path.join(output_dir, 'linear_centered'))
#     # Polynomial SVM
#     print("Evaluating Polynomial SVM")
#     degree = 2
#     coef0 = 1
#     svm_classifier = SVMClassifier(kernel='poly', degree=degree, coef0=coef0)
#     svm_classifier.evaluate_model_polynomial(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, degree, coef0, pi_t,
#                                              os.path.join(output_dir, 'poly'))
#     # Optional: Polynomial kernel with d = 4, c = 1, ξ = 0
#     print("Evaluating Polynomial SVM with d=4")
#     degree = 4
#     coef0 = 1
#     svm_classifier.evaluate_model_polynomial(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, degree, coef0, pi_t,
#                                              os.path.join(output_dir, 'poly_d4'))
#     # RBF SVM
#     print("Evaluating RBF SVM")
#     gammas = np.logspace(-4, -1, 4)
#     svm_classifier = SVMClassifier(kernel='rbf')
#     svm_classifier.evaluate_model_rbf(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, gammas, pi_t, os.path.join(output_dir, 'rbf'))
