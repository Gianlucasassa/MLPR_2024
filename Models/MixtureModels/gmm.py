import numpy as np
import scipy.special
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class GMMClass:
    def __init__(self, gmm_init=None, covariance_type='full'):
        self.gmm = gmm_init if gmm_init is not None else []
        self.covariance_type = covariance_type

    def logpdf_GAU_ND(self, X, mu, C):
        D = X.shape[0]
        XC = X - mu[:, None]
        if self.covariance_type == 'diag':
            invC = np.diag(1.0 / (np.diag(C) + 1e-6))  # Add regularization term to avoid division by zero
            log_det_C = np.sum(np.log(np.diag(C) + 1e-6))  # Add regularization term to avoid log(0)
        else:
            invC = np.linalg.inv(C + 1e-6 * np.eye(D))  # Add regularization term
            log_det_C = np.linalg.slogdet(C + 1e-6 * np.eye(D))[1]  # Add regularization term
        const = -0.5 * D * np.log(2 * np.pi) - 0.5 * log_det_C
        return const - 0.5 * np.sum(XC * np.dot(invC, XC), axis=0)

    def logpdf_GMM(self, X):
        S = np.array([self.logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in self.gmm])
        logdens = scipy.special.logsumexp(S, axis=0)
        return logdens

    def EM(self, X, stop_crit=1e-6, max_iter=100):
        D, N = X.shape
        M = len(self.gmm)
        ll_old = None

        for _ in range(max_iter):
            S = np.array([self.logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in self.gmm])
            logden = scipy.special.logsumexp(S, axis=0)
            gamma = np.exp(S - logden)

            Z = gamma.sum(axis=1)
            F = np.dot(gamma, X.T)
            S = np.zeros((M, D, D))

            for g in range(M):
                S[g] = np.dot(gamma[g] * X, X.T)

            for g in range(M):
                w_new = Z[g] / N
                mu_new = F[g] / Z[g]
                C_new = S[g] / Z[g] - np.outer(mu_new, mu_new)
                if self.covariance_type == 'diag':
                    C_new = np.diag(np.diag(C_new))
                self.gmm[g] = (w_new, mu_new, C_new)

            ll_new = logden.mean()
            if ll_old is not None and ll_new - ll_old < stop_crit:
                break
            ll_old = ll_new

        return self.gmm

    def LBG(self, X, alpha=0.1, max_components=16):
        D = X.shape[0]
        self.gmm = [(1.0, X.mean(axis=1), np.cov(X))]
        while len(self.gmm) < max_components:
            new_gmm = []
            for w, mu, C in self.gmm:
                U, s, Vh = np.linalg.svd(C)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                new_gmm.append((w / 2, mu - d.flatten(), C))
                new_gmm.append((w / 2, mu + d.flatten(), C))
            self.gmm = new_gmm
            self.EM(X)
        return self.gmm

    def log_likelihood(self, X):
        return self.logpdf_GMM(X).mean()

    def constrain_eigenvalues(self, psi=0.01):
        for i, (w, mu, C) in enumerate(self.gmm):
            U, s, Vh = np.linalg.svd(C)
            s[s < psi] = psi
            C_new = np.dot(U, np.dot(np.diag(s), Vh))
            self.gmm[i] = (w, mu, C_new)

    def classify(self, X1, X2, max_components=32, stop_crit=1e-6, max_iter=100, alpha=0.1):
        results = []
        for n_components in range(1, max_components + 1):
            self.LBG(X1, alpha=alpha, max_components=n_components)
            self.EM(X1, stop_crit=stop_crit, max_iter=max_iter)
            ll1 = self.log_likelihood(X1)

            self.LBG(X2, alpha=alpha, max_components=n_components)
            self.EM(X2, stop_crit=stop_crit, max_iter=max_iter)
            ll2 = self.log_likelihood(X2)

            results.append((n_components, ll1, ll2))
        return results

def evaluate_classification(DTR, LTR, DTE, LTE, models, n_components_list, output_dir='Output/GMM'):
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        min_dcf_values = []
        actual_dcf_values = []
        log_odds = []

        for n_components in n_components_list:
            sub_output_dir = os.path.join(output_dir, f'{name}_{n_components}')
            os.makedirs(sub_output_dir, exist_ok=True)

            print(f'Training {name} with {n_components} components...')
            # Training the model
            model.LBG(DTR, max_components=n_components)
            model.EM(DTR)

            # Evaluate on the validation set
            scores = model.logpdf_GMM(DTE)
            print(f'Scores for {name} with {n_components} components: {scores[:5]}')  # Print first 5 scores for verification

            min_dcf = compute_min_DCF(scores, LTE)
            actual_dcf = compute_actual_DCF(scores, LTE)

            print(f"EVALUATING {name} with {n_components} components - minDCF={min_dcf}, actualDCF={actual_dcf}")

            min_dcf_values.append(min_dcf)
            actual_dcf_values.append(actual_dcf)
            log_odds.append(np.log(n_components))

            # Save the DCF values
            np.save(os.path.join(sub_output_dir, 'min_dcf.npy'), min_dcf)
            np.save(os.path.join(sub_output_dir, 'actual_dcf.npy'), actual_dcf)

            # ROC Curve
            fpr, tpr, _ = roc_curve(LTE, scores)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC={roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name} with {n_components} components')
            plt.legend(loc="lower right")
            plt.grid()
            plt.savefig(os.path.join(sub_output_dir, 'roc_curve.png'))
            plt.close()

            # Bayes Error Plot
            effPriorLogOdds = np.linspace(-4, 4, 100)
            plot_bayes_error(scores, LTE, effPriorLogOdds, os.path.join(sub_output_dir, 'bayes_error_plot.png'))

        # Save the overall DCF plots
        plt.figure()
        plt.plot(log_odds, min_dcf_values, label='Min DCF')
        plt.plot(log_odds, actual_dcf_values, label='Actual DCF')
        plt.xlabel('Log Number of Components')
        plt.ylabel('DCF')
        plt.legend()
        plt.grid()
        plt.title(f'DCF - {name}')
        plt.savefig(os.path.join(output_dir, f'{name}_dcf_plot.png'))
        plt.close()

def compute_min_DCF(scores, labels, pi_t=0.5, cfn=1, cfp=1):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    dcf_values = []
    for thr in thresholds:
        fnr = np.mean((scores < thr) & (labels == 1))
        fpr = np.mean((scores >= thr) & (labels == 0))
        dcf = pi_t * cfn * fnr + (1 - pi_t) * cfp * fpr
        dcf_values.append(dcf)
    return min(dcf_values)

def compute_actual_DCF(scores, labels, pi_t=0.5, cfn=1, cfp=1):
    threshold = -np.log(pi_t / (1 - pi_t))
    fnr = np.mean((scores < threshold) & (labels == 1))
    fpr = np.mean((scores >= threshold) & (labels == 0))
    return pi_t * cfn * fnr + (1 - pi_t) * cfp * fpr

def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    dcf = []
    mindcf = []
    for p in effPriorLogOdds:
        pi1 = 1 / (1 + np.exp(-p))
        decisions = compute_optimal_bayes_decisions(llrs, pi1, 1, 1)
        confusion_matrix = compute_confusion_matrix(decisions, labels)
        bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
        normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, 1, 1)
        dcf.append(normalized_dcf)

        min_bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
        min_normalized_dcf = compute_normalized_dcf(min_bayes_risk, pi1, 1, 1)
        mindcf.append(min_normalized_dcf)

    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF value')
    plt.legend()
    plt.grid()
    plt.title('Bayes Error Plot')
    plt.savefig(output_file)
    plt.close()

def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
    t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    return (llrs >= t).astype(int)

def compute_confusion_matrix(predictions, labels):
    K = 2  # Since it's a binary classification problem
    confusion_matrix = np.zeros((K, K), dtype=int)

    if predictions.ndim > 1:
        predictions = predictions.flatten()

    for i in range(len(labels)):
        pred = int(predictions[i])
        true = int(labels[i])
        if pred < 0 or pred >= K or true < 0 or true >= K:
            continue  # Skip invalid values
        confusion_matrix[pred, true] += 1

    return confusion_matrix

def compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]
    TP = confusion_matrix[1, 1]
    Pfn = FN / (FN + TP)
    Pfp = FP / (FP + TN)
    DCFu = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
    return DCFu

def compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp):
    Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
    return bayes_risk / Bdummy

def train_GMM(DTE, DTR, LTE, LTR):
    n_components_list = [1, 2, 4, 8, 16, 32]
    models = {
        'GMM_full': GMMClass(covariance_type='full'),
        'GMM_diag': GMMClass(covariance_type='diag')
    }
    evaluate_classification(DTR, LTR, DTE, LTE, models, n_components_list)
