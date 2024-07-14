import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

from Models.Gaussian.gaussian_models import compute_bayes_risk, compute_normalized_dcf, \
    compute_confusion_matrix
from Preprocess.DatasetPlots import center_data

import scipy.optimize
from sklearn.metrics import roc_curve, auc

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
#         return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
#
#     def compute_kernel_matrix(self, X):
#         n_samples = X.shape[1]
#         K = np.zeros((n_samples, n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 if self.kernel == 'linear':
#                     K[i, j] = self.linear_kernel(X[:, i], X[:, j])
#                 elif self.kernel == 'poly':
#                     K[i, j] = self.polynomial_kernel(X[:, i], X[:, j])
#                 elif self.kernel == 'rbf':
#                     K[i, j] = self.rbf_kernel(X[:, i], X[:, j])
#         return K
#
#     def dual_objective(self, alpha, H):
#         return 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.sum(alpha)
#
#     def dual_objective_grad(self, alpha, H):
#         return np.dot(H, alpha) - np.ones_like(alpha)
#
#     def train(self, DTR, LTR):
#         n_samples = DTR.shape[1]
#         self.LTR = LTR
#         K = self.compute_kernel_matrix(DTR)
#         H = np.outer(LTR, LTR) * K
#
#         # Constraints
#         bounds = [(0, self.C) for _ in range(n_samples)]
#         constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, LTR)}
#
#         # Solve the quadratic programming problem
#         result = scipy.optimize.minimize(
#             fun=lambda alpha: self.dual_objective(alpha, H),
#             x0=np.zeros(n_samples),
#             jac=lambda alpha: self.dual_objective_grad(alpha, H),
#             bounds=bounds,
#             constraints=constraints
#         )
#
#         self.alpha = result.x
#         sv = self.alpha > 1e-5
#         self.support_vectors = DTR[:, sv]
#         self.support_vector_labels = LTR[sv]
#         self.alpha = self.alpha[sv]
#
#         if self.kernel == 'linear':
#             self.w = np.sum(self.alpha * self.support_vector_labels * self.support_vectors, axis=1)
#
#         self.b = np.mean(self.support_vector_labels - np.sum((self.alpha * self.support_vector_labels)[:, None] * K[sv][:, sv], axis=0))
#
#     def project(self, D):
#         if self.kernel == 'linear':
#             return np.dot(self.w.T, D) + self.b
#         else:
#             K = np.array([self.compute_kernel(self.support_vectors, x) for x in D.T])
#             return np.dot((self.alpha * self.support_vector_labels), K) + self.b
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
#         min_dcf = float('inf')
#         for t in thresholds:
#             predictions = (scores >= t).astype(int)
#             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
#             if dcf < min_dcf:
#                 min_dcf = dcf
#         return min_dcf
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
#     def plot_dcf(self, lambdas, dcf_values, min_dcf_values, output_file='dcf_plot.png'):
#         plt.figure()
#         plt.plot(lambdas, dcf_values, label='Actual DCF')
#         plt.plot(lambdas, min_dcf_values, label='Min DCF')
#         plt.xscale('log')
#         plt.xlabel('Lambda')
#         plt.ylabel('DCF')
#         plt.legend()
#         plt.grid()
#         plt.savefig(output_file)
#         plt.close()
#
#     def evaluate_model(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         dcf_values = []
#         min_dcf_values = []
#
#         for l in lambdas:
#             self.C = l
#             self.train(DTR, LTR)
#             scores = self.project(DTE)
#             dcf = self.compute_dcf(scores, LTE, pi_t)
#             min_dcf = self.compute_min_dcf(scores, LTE, pi_t)
#             dcf_values.append(dcf)
#             min_dcf_values.append(min_dcf)
#             error_rate = self.compute_error_rate(self.compute_predictions(scores), LTE)
#             print(f"Lambda: {l}, DCF: {dcf}, Min DCF: {min_dcf}, Error Rate: {error_rate}")
#
#         self.plot_dcf(lambdas, dcf_values, min_dcf_values, os.path.join(output_dir, 'dcf_plot.png'))
#
# def compute_dcf(scores, labels, pi_t):
#     thresholds = np.sort(scores)
#     min_dcf = float('inf')
#     for t in thresholds:
#         predictions = (scores >= t).astype(int)
#         fnr = np.mean(predictions[labels == 1] == 0)
#         fpr = np.mean(predictions[labels == 0] == 1)
#         dcf = pi_t * fnr + (1 - pi_t) * fpr
#         if dcf < min_dcf:
#             min_dcf = dcf
#     return min_dcf
#
# def compute_min_dcf(scores, labels, pi_t):
#     thresholds = np.sort(scores)
#     min_dcf = float('inf')
#     for t in thresholds:
#         predictions = (scores >= t).astype(int)
#         fnr = np.mean(predictions[labels == 1] == 0)
#         fpr = np.mean(predictions[labels == 0] == 1)
#         dcf = pi_t * fnr + (1 - pi_t) * fpr
#         if dcf < min_dcf:
#             min_dcf = dcf
#     return min_dcf
#
#
#
#
#
# def linear_svm_primal(DTR, LTR, C):
#     n, m = DTR.shape
#     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
#     D_hat = np.vstack([DTR, np.ones(m)])
#
#     def primal_objective(w_hat):
#         w = w_hat[:-1]
#         b = w_hat[-1]
#         margin = Z * (np.dot(DTR.T, w) + b)
#         hinge_loss = np.maximum(0, 1 - margin)
#         return 0.5 * np.linalg.norm(w) ** 2 + C * np.sum(hinge_loss)
#
#     w_hat0 = np.zeros(n + 1)
#     w_hat, _, _ = fmin_l_bfgs_b(primal_objective, w_hat0, approx_grad=True)
#
#     return w_hat[:-1], w_hat[-1]
#
#
# def linear_svm_dual(DTR, LTR, C):
#     n, m = DTR.shape
#     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
#     H = (Z @ Z.T) * (DTR.T @ DTR)
#
#     def dual_objective(alpha):
#         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
#
#     def dual_gradient(alpha):
#         return np.dot(H, alpha) - np.ones(m)
#
#     bounds = [(0, C) for _ in range(m)]
#     constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, Z), 'jac': lambda alpha: Z.flatten()}
#     alpha0 = np.zeros(m)
#     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
#
#     w_star = np.sum((alpha * Z.flatten()).reshape(-1, 1) * DTR.T, axis=0)
#     support_vector_indices = np.where(alpha > 1e-6)[0]
#     b_star = np.mean(Z[support_vector_indices] - np.dot(DTR.T[support_vector_indices], w_star))
#
#     return w_star, b_star
#
#
# def svm_evaluate(DTE, w, b):
#     scores = np.dot(DTE.T, w) + b
#     return scores
#
#
# def polynomial_kernel_svm(DTR, LTR, DTE, LTE, degree=2, c=1):
#     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
#     H = np.dot(Z, Z.T) * (np.dot(DTR.T, DTR) + c) ** degree
#
#     def dual_objective(alpha):
#         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
#
#     def dual_gradient(alpha):
#         return np.dot(H, alpha) - np.ones(len(alpha))
#
#     bounds = [(0, c) for _ in range(len(LTR))]
#     alpha0 = np.zeros(len(LTR))
#     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
#
#     def svm_score(x):
#         return np.sum(alpha * Z.flatten() * (np.dot(DTR.T, x) + c) ** degree)
#
#     scores = np.array([svm_score(x) for x in DTE.T])
#
#     return scores
#
#
# def rbf_kernel_svm(DTR, LTR, DTE, LTE, gamma=1.0, C=1.0):
#     Z = np.where(LTR == 1, 1, -1).reshape(-1, 1)
#     K = np.exp(-gamma * np.sum((DTR[:, :, None] - DTR[:, None, :]) ** 2, axis=0))
#     H = np.dot(Z, Z.T) * K
#
#     def dual_objective(alpha):
#         return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
#
#     def dual_gradient(alpha):
#         return np.dot(H, alpha) - np.ones(len(alpha))
#
#     bounds = [(0, C) for _ in range(len(LTR))]
#     alpha0 = np.zeros(len(LTR))
#     alpha, _, _ = fmin_l_bfgs_b(dual_objective, alpha0, fprime=dual_gradient, bounds=bounds)
#
#     def svm_score(x):
#         k = np.exp(-gamma * np.sum((DTR - x.reshape(-1, 1)) ** 2, axis=0))
#         return np.sum(alpha * Z.flatten() * k)
#
#     scores = np.array([svm_score(x) for x in DTE.T])
#
#     return scores
#
#
# def plot_dcf_vs_c(Cs, min_dcf, act_dcf, model_name, filename):
#     os.makedirs("Output/svm", exist_ok=True)
#     plt.plot(Cs, min_dcf, label='Min DCF')
#     plt.plot(Cs, act_dcf, label='Actual DCF')
#     plt.xscale('log')
#     plt.xlabel('C')
#     plt.ylabel('DCF')
#     plt.title(f'{model_name} DCF vs. C')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join("Output/svm", filename))
#     plt.close()
#
#
# # Function for Linear SVM
# def run_linear_svm(DTR, LTR, DTE, LTE, Cs, centered=False):
#     min_dcf = []
#     act_dcf = []
#
#     if centered:
#         DTR, mean = center_data(DTR)
#         DTE, _ = center_data(DTE, mean)
#
#     for C in Cs:
#         w, b = linear_svm_dual(DTR, LTR, C)
#         scores = svm_evaluate(DTE, w, b)
#         optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
#         confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
#         bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
#         norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
#
#         min_dcf.append(norm_dcf)
#         act_dcf.append(compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
#
#     label = 'Centered Linear SVM' if centered else 'Linear SVM'
#     filename = 'centered_linear_svm_dcf.png' if centered else 'linear_svm_dcf.png'
#     plot_dcf_vs_c(Cs, min_dcf, act_dcf, label, filename)
#
#
# # Function for Polynomial SVM
# def run_polynomial_svm(DTR, LTR, DTE, LTE, Cs, degree=2, c=1):
#     min_dcf = []
#     act_dcf = []
#
#     for C in Cs:
#         scores = polynomial_kernel_svm(DTR, LTR, DTE, LTE, degree=degree, c=c)
#         optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
#         confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
#         bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
#         norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
#
#         min_dcf.append(norm_dcf)
#         act_dcf.append(compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
#
#     plot_dcf_vs_c(Cs, min_dcf, act_dcf, 'Polynomial SVM', 'poly_svm_dcf.png')
#
#
# # Function for RBF SVM
# def run_rbf_svm(DTR, LTR, DTE, LTE, Cs, gammas):
#     for gamma in gammas:
#         min_dcf_rbf_svm = []
#         act_dcf_rbf_svm = []
#
#         for C in Cs:
#             scores = rbf_kernel_svm(DTR, LTR, DTE, LTE, gamma=gamma, C=C)
#             optimal_decisions = compute_optimal_bayes_decisions(scores, 0.1, 1.0, 1.0)
#             confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
#             bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
#             norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
#
#             min_dcf_rbf_svm.append(norm_dcf)
#             act_dcf_rbf_svm.append(
#                 compute_normalized_dcf(compute_bayes_risk(confusion_matrix, 0.5, 1.0, 1.0), 0.5, 1.0, 1.0))
#
#         plot_dcf_vs_c(Cs, min_dcf_rbf_svm, act_dcf_rbf_svm, f'RBF SVM (γ={gamma})', f'rbf_svm_dcf_{gamma}.png')


from scipy.optimize import minimize

from sklearn.metrics import roc_curve, auc

def normalize_data(D):
    mean = np.mean(D, axis=1, keepdims=True)
    std = np.std(D, axis=1, keepdims=True)
    normalized_D = (D - mean) / std
    print(f"Data normalized: mean={np.mean(normalized_D)}, std={np.std(normalized_D)}")
    return normalized_D


def center_data(D):
    mean = np.mean(D, axis=1, keepdims=True)
    centered_D = D - mean
    print(f"Data centered: mean={np.mean(centered_D)}")
    return centered_D, mean


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def reduce_dataset(D, L, fraction=0.1, seed=42):
    np.random.seed(seed)
    n_samples = int(D.shape[1] * fraction)
    indices = np.random.choice(D.shape[1], n_samples, replace=False)
    print(f"Taken {n_samples} samples out of {D.shape[1]}")
    return D[:, indices], L[indices]


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=2, coef0=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
        self.w = None

    def linear_kernel(self, x1, x2):
        return np.dot(x1.T, x2)

    def polynomial_kernel(self, x1, x2):
        return (np.dot(x1.T, x2) + self.coef0) ** self.degree

    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self.polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError("Unsupported kernel type")

    def train(self, DTR, LTR):
        ZTR = LTR * 2.0 - 1.0  # Convert labels to +1/-1
        DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * 1])  # K=1 for simplicity
        H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

        def fOpt(alpha):
            Ha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
            grad = Ha.ravel() - np.ones(alpha.size)
            return loss, grad

        alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]),
                                                       bounds=[(0, self.C) for _ in LTR], factr=1.0)

        w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)

        self.w = w_hat[0:DTR.shape[0]]
        self.b = w_hat[-1]

        self.alpha = alphaStar
        sv = self.alpha > 1e-5
        self.support_vectors = DTR[:, sv]
        self.support_vector_labels = LTR[sv]
        self.alpha = self.alpha[sv]

        if len(self.support_vector_labels) == 0:
            raise ValueError("No support vectors found for C={}".format(self.C))

        primalLoss, dualLoss = self.primalLoss(w_hat, DTR_EXT, ZTR), -fOpt(alphaStar)[0]
        print('SVM - C %e - primal loss %e - dual loss %e - duality gap %e' % (
            self.C, primalLoss, dualLoss, primalLoss - dualLoss))
        # print(f"Number of support vectors: {len(self.support_vector_labels)}")

    def primalLoss(self, w_hat, DTR_EXT, ZTR):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat) ** 2 + self.C * np.maximum(0, 1 - ZTR * S).sum()

    def project(self, D):
        if self.kernel == 'linear':
            return np.dot(self.w.T, D) + self.b
        else:
            K = np.array([[self.compute_kernel(self.support_vectors[:, i], D[:, j]) for j in range(D.shape[1])] for i in
                          range(self.support_vectors.shape[1])])
            return np.dot((self.alpha * self.support_vector_labels), K) + self.b

    def predict(self, D):
        return np.sign(self.project(D))

    def compute_predictions(self, scores):
        return (scores > 0).astype(int)

    def compute_error_rate(self, predictions, true_labels):
        return np.mean(predictions != true_labels)

    def compute_dcf(self, scores, labels, pi_t):
        thresholds = np.sort(scores)
        min_dcf = float('inf')
        for t in thresholds:
            predictions = (scores >= t).astype(int)
            dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
            if dcf < min_dcf:
                min_dcf = dcf
        return min_dcf

    def compute_min_dcf(self, scores, labels, pi_t):
        thresholds = np.sort(scores)
        min_dcf = float('inf')
        for t in thresholds:
            predictions = (scores >= t).astype(int)
            dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t)
            if dcf < min_dcf:
                min_dcf = dcf
        return min_dcf

    def compute_dcf_at_threshold(self, predictions, labels, pi_t):
        fnr = np.mean(predictions[labels == 1] == 0)
        fpr = np.mean(predictions[labels == 0] == 1)
        dcf = pi_t * fnr + (1 - pi_t) * fpr
        return dcf

    def plot_roc_curve(self, llrs, labels, output_file='roc_curve.png'):
        fpr, tpr, _ = roc_curve(labels, llrs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_file)
        plt.close()

    def evaluate_model(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type, gamma=None):

        os.makedirs(output_dir, exist_ok=True)

        dcf_values = []
        min_dcf_values = []
        successful_lambdas = []

        for l in lambdas:
            self.C = l
            if gamma is not None:
                self.gamma = gamma
            try:
                self.train(DTR, LTR)
                print(f"Training with C={l}, number of support vectors: {len(self.support_vector_labels)}")
                scores = self.project(DTE)
                dcf = self.compute_dcf(scores, LTE, pi_t)
                min_dcf = self.compute_min_dcf(scores, LTE, pi_t)
                dcf_values.append(dcf)
                min_dcf_values.append(min_dcf)
                successful_lambdas.append(l)
                error_rate = self.compute_error_rate(self.compute_predictions(scores), LTE)
                print(f"Lambda: {l}, DCF: {dcf}, Min DCF: {min_dcf}, Error Rate: {error_rate}")
            except ValueError as e:
                print(f"Skipping C={l} due to error: {e}")

        plot_title = f'{kernel_type.capitalize()} SVM - DCF vs Lambda'
        if gamma is not None:
            plot_title += f' (gamma={gamma})'
        self.plot_dcf(successful_lambdas, dcf_values, min_dcf_values, os.path.join(output_dir,
                                                                                   f'dcf_plot_{kernel_type}{"_gamma_" + str(gamma) if gamma else ""}.png'),
                      plot_title)

    def plot_dcf(self, lambdas, dcf_values, min_dcf_values, output_file='dcf_plot.png', plot_title='DCF vs Lambda'):
        plt.figure()
        plt.plot(lambdas, dcf_values, label='Actual DCF')
        plt.plot(lambdas, min_dcf_values, label='Min DCF')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('DCF')
        plt.title(plot_title)
        plt.legend()
        plt.grid()
        plt.savefig(output_file)
        plt.close()

    def evaluate_model_polynomial(self, DTR, LTR, DTE, LTE, lambdas, degree, coef0, pi_t, output_dir):
        self.kernel = 'poly'
        self.degree = degree
        self.coef0 = coef0
        self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type='poly')

    def evaluate_model_rbf(self, DTR, LTR, DTE, LTE, lambdas, gammas, pi_t, output_dir):
        self.kernel = 'rbf'
        for gamma in gammas:
            self.gamma = gamma
            gamma_output_dir = os.path.join(output_dir, f'gamma_{gamma}')
            self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, gamma_output_dir, kernel_type='rbf', gamma=gamma)

    def evaluate_model_linear(self, DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir):
        self.kernel = 'linear'
        self.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir, kernel_type='linear')


def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
    t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    return (llrs >= t).astype(int)


def train_SVM(DTE, DTR, LTE, LTR):
    fraction = 0.1  # Use 10% of the data for initial testing
    DTR, LTR = reduce_dataset(DTR, LTR, fraction)
    lambdas = np.logspace(-5, 0, 11)
    pi_t = 0.1
    output_dir = 'Output/SVM_Results'
    # Normalize the data
    DTR_normalized = normalize_data(DTR)
    DTE_normalized = normalize_data(DTE)
    # Center the data
    DTR_centered, _ = center_data(DTR)
    DTE_centered, _ = center_data(DTE)
    # Linear SVM
    print("Evaluating Linear SVM")
    svm_classifier = SVMClassifier(kernel='linear')
    svm_classifier.evaluate_model_linear(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, pi_t, os.path.join(output_dir, 'linear'))
    print("Evaluating Linear SVM - centered")
    svm_classifier.evaluate_model_linear(DTR_centered, LTR, DTE_centered, LTE, lambdas, pi_t,
                                         os.path.join(output_dir, 'linear_centered'))
    # Polynomial SVM
    print("Evaluating Polynomial SVM")
    degree = 2
    coef0 = 1
    svm_classifier = SVMClassifier(kernel='poly', degree=degree, coef0=coef0)
    svm_classifier.evaluate_model_polynomial(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, degree, coef0, pi_t,
                                             os.path.join(output_dir, 'poly'))
    # Optional: Polynomial kernel with d = 4, c = 1, ξ = 0
    print("Evaluating Polynomial SVM with d=4")
    degree = 4
    coef0 = 1
    svm_classifier.evaluate_model_polynomial(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, degree, coef0, pi_t,
                                             os.path.join(output_dir, 'poly_d4'))
    # RBF SVM
    print("Evaluating RBF SVM")
    gammas = np.logspace(-4, -1, 4)
    svm_classifier = SVMClassifier(kernel='rbf')
    svm_classifier.evaluate_model_rbf(DTR_normalized, LTR, DTE_normalized, LTE, lambdas, gammas, pi_t, os.path.join(output_dir, 'rbf'))
