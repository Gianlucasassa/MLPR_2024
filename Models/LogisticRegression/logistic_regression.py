import os
from itertools import groupby

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import roc_curve, auc

from main import compute_dcf, compute_min_dcf



def expand_features(D):
    expanded_features = [D]
    for i in range(D.shape[0]):
        for j in range(i, D.shape[0]):
            expanded_features.append(D[i] * D[j])
    return np.vstack(expanded_features)

def preprocess_data(DTR, DTE, method='none'):
    if method == 'center':
        mean = np.mean(DTR, axis=1, keepdims=True)
        DTR = DTR - mean
        DTE = DTE - mean
    elif method == 'z_norm':
        mean = np.mean(DTR, axis=1, keepdims=True)
        std = np.std(DTR, axis=1, keepdims=True)
        DTR = (DTR - mean) / std
        DTE = (DTE - mean) / std
    elif method == 'whiten':
        mean = np.mean(DTR, axis=1, keepdims=True)
        DTR = DTR - mean
        DTE = DTE - mean
        cov = np.cov(DTR)
        U, S, _ = np.linalg.svd(cov)
        DTR = np.dot(U.T, DTR) / np.sqrt(S[:, np.newaxis])
        DTE = np.dot(U.T, DTE) / np.sqrt(S[:, np.newaxis])
    elif method == 'pca':
        mean = np.mean(DTR, axis=1, keepdims=True)
        DTR = DTR - mean
        DTE = DTE - mean
        cov = np.cov(DTR)
        U, S, _ = np.linalg.svd(cov)
        DTR = np.dot(U.T, DTR)
        DTE = np.dot(U.T, DTE)
    return DTR, DTE

def analyze_dcf_vs_lambda(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1, output_file_prefix='dcf_vs_lambda'):
    actual_dcfs = []
    min_dcfs = []
    for l in lambdas:
        logreg_classifier = LogRegClass(DTR, LTR, l, prior_weighted=False)
        logreg_classifier.train()
        scores = logreg_classifier.predict(DTE)
        actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
        min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
        actual_dcfs.append(actual_dcf)
        min_dcfs.append(min_dcf)

    plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
    plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_file_prefix}.png')
    plt.close()

def analyze_with_fewer_samples(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]
    analyze_dcf_vs_lambda(DTR_reduced, LTR_reduced, DTE, LTE, lambdas, pi_t, output_file_prefix='dcf_vs_lambda_reduced')

def analyze_prior_weighted(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
    actual_dcfs = []
    min_dcfs = []
    for l in lambdas:
        logreg_classifier = LogRegClass(DTR, LTR, l, prior_weighted=True, pi_t=pi_t)
        logreg_classifier.train()
        scores = logreg_classifier.predict(DTE)
        actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
        min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
        actual_dcfs.append(actual_dcf)
        min_dcfs.append(min_dcf)

    plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
    plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Prior-Weighted DCF vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('prior_weighted_dcf_vs_lambda.png')
    plt.close()

def analyze_quadratic(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
    DTR_exp = expand_features(DTR)
    DTE_exp = expand_features(DTE)
    actual_dcfs = []
    min_dcfs = []
    for l in lambdas:
        logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=False)
        logreg_classifier.train()
        scores = logreg_classifier.predict(DTE_exp)
        actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
        min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
        actual_dcfs.append(actual_dcf)
        min_dcfs.append(min_dcf)

    plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
    plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Quadratic DCF vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('quadratic_dcf_vs_lambda.png')
    plt.close()

def analyze_preprocessing(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1, output_file_prefix='dcf_vs_lambda'):
    methods = ['none', 'center', 'z_norm', 'whiten', 'pca']
    for method in methods:
        DTR_prep, DTE_prep = preprocess_data(DTR, DTE, method)
        actual_dcfs = []
        min_dcfs = []
        for l in lambdas:
            logreg_classifier = LogRegClass(DTR_prep, LTR, l, prior_weighted=False)
            logreg_classifier.train()
            scores = logreg_classifier.predict(DTE_prep)
            actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
            min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
            actual_dcfs.append(actual_dcf)
            min_dcfs.append(min_dcf)

        plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
        plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('DCF')
        plt.title(f'DCF vs Lambda ({method})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_file_prefix}_{method}.png')
        plt.close()

def compare_all_models(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
    models = {
        'LogisticRegression': LogRegClass(DTR, LTR, l=1.0),
        'PriorWeightedLogReg': LogRegClass(DTR, LTR, l=1.0, prior_weighted=True, pi_t=pi_t),
        'QuadraticLogReg': LogRegClass(expand_features(DTR), LTR, l=1.0)
    }
    results = {}

    for model_name, model in models.items():
        actual_dcfs = []
        min_dcfs = []
        for l in lambdas:
            model.l = l
            if model_name == 'QuadraticLogReg':
                model.DTR = expand_features(DTR)
                DTE_exp = expand_features(DTE)
                model.train()
                scores = model.predict(DTE_exp)
            else:
                model.train()
                scores = model.predict(DTE)
            actual_dcf = model.compute_dcf(scores, LTE, pi_t)
            min_dcf = model.compute_min_dcf(scores, LTE, pi_t)
            actual_dcfs.append(actual_dcf)
            min_dcfs.append(min_dcf)

        results[model_name] = (actual_dcfs, min_dcfs)

        plt.plot(lambdas, actual_dcfs, marker='o', label=f'{model_name} Actual DCF')
        plt.plot(lambdas, min_dcfs, marker='o', label=f'{model_name} Min DCF')

    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs Lambda for Different Models')
    plt.legend()
    plt.grid(True)
    plt.savefig('dcf_vs_lambda_all_models.png')
    plt.close()

    return results

def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
    dcf = []
    mindcf = []
    for p in effPriorLogOdds:
        pi1 = 1 / (1 + np.exp(-p))
        thresholds = np.sort(llrs)
        fnr, fpr = [], []

        for threshold in thresholds:
            predictions = (llrs >= threshold).astype(int)
            fnr.append(np.mean(predictions[labels == 1] == 0))
            fpr.append(np.mean(predictions[labels == 0] == 1))

        dcf_value = pi1 * 1 * np.mean(predictions[labels == 1] == 0) + (1 - pi1) * 1 * np.mean(
            predictions[labels == 0] == 1)
        min_dcf_value = min([pi1 * 1 * fnr_i + (1 - pi1) * 1 * fpr_i for fnr_i, fpr_i in zip(fnr, fpr)])

        dcf.append(dcf_value)
        mindcf.append(min_dcf_value)

    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF value')
    plt.legend()
    plt.grid()
    plt.title('Bayes Error Plot')
    plt.savefig(output_file)
    plt.close()

class LogRegClass:
    def __init__(self, DTR, LTR, l, prior_weighted=False, pi_t=0.1):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.prior_weighted = prior_weighted
        self.pi_t = pi_t
        self.w = None
        self.b = None

    def logreg_obj(self, v):
        w, b = v[:-1], v[-1]
        Z = 2 * self.LTR - 1
        S = np.dot(w.T, self.DTR) + b
        reg_term = 0.5 * self.l * np.dot(w, w)

        if self.prior_weighted and self.pi_t is not None:
            nT = np.sum(self.LTR)
            nF = len(self.LTR) - nT
            xi = np.where(self.LTR == 1, self.pi_t / nT, (1 - self.pi_t) / nF)
            loss = np.mean(xi * np.logaddexp(0, -Z * S))
        else:
            loss = np.logaddexp(0, -Z * S).mean()

        return reg_term + loss

    def logreg_obj_grad(self, v):
        w, b = v[:-1], v[-1]
        Z = 2 * self.LTR - 1
        S = np.dot(w.T, self.DTR) + b
        G = -Z / (1 + np.exp(Z * S))

        if self.prior_weighted and self.pi_t is not None:
            nT = np.sum(self.LTR)
            nF = len(self.LTR) - nT
            xi = np.where(self.LTR == 1, self.pi_t / nT, (1 - self.pi_t) / nF)
            G = G * xi

        grad_w = self.l * w + (G @ self.DTR.T) / self.DTR.shape[1]
        grad_b = G.mean()
        return np.append(grad_w, grad_b)

    def train(self):
        x0 = np.zeros(self.DTR.shape[0] + 1)
        opt = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj, x0=x0, fprime=self.logreg_obj_grad, approx_grad=False)
        self.w, self.b = opt[0][:-1], opt[0][-1]

    def predict(self, DTE):
        scores = np.dot(self.w.T, DTE) + self.b
        return scores

    def compute_predictions(self, scores):
        return (scores > 0).astype(int)

    def compute_error_rate(self, predictions, true_labels):
        return np.mean(predictions != true_labels)

    @staticmethod
    def plot_error_rates(results, output_file='error_rates.png'):
        from itertools import groupby

        for config_name, group in groupby(results, key=lambda x: x[0]):
            lambdas, error_rates = zip(*[(l, e) for _, l, e in group])
            plt.plot(lambdas, error_rates, marker='o', label=config_name)

        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Lambda for Different Configurations')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()

    def compute_dcf_at_threshold(self, predictions, labels, pi_t, Cfn=1, Cfp=1):
        fnr = np.mean(predictions[labels == 1] == 0)
        fpr = np.mean(predictions[labels == 0] == 1)
        dcf = pi_t * Cfn * fnr + (1 - pi_t) * Cfp * fpr
        return dcf

    def compute_dcf(self, scores, labels, pi_t, Cfn=1, Cfp=1):
        thresholds = np.sort(scores)
        min_dcf = float('inf')
        for t in thresholds:
            predictions = (scores >= t).astype(int)
            dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t, Cfn, Cfp)
            if dcf < min_dcf:
                min_dcf = dcf
        return min_dcf

    def compute_min_dcf(self, scores, labels, pi_t, Cfn=1, Cfp=1):
        thresholds = np.sort(scores)
        min_dcf = float('inf')
        for t in thresholds:
            predictions = (scores >= t).astype(int)
            dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t, Cfn, Cfp)
            if dcf < min_dcf:
                min_dcf = dcf
        return min_dcf



def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]
    z = 2 * LTR - 1  # Convert labels to +/- 1
    S = np.dot(w.T, DTR) + b
    regularizer = (l / 2) * np.linalg.norm(w) ** 2
    loss = np.logaddexp(0, -z * S).mean()
    J = regularizer + loss
    gradient_w = l * w + np.dot(DTR, (-z / (1 + np.exp(z * S)))) / DTR.shape[1]
    gradient_b = (-z / (1 + np.exp(z * S))).mean()
    gradient = np.append(gradient_w, gradient_b)
    return J, gradient


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_logreg(DTR, LTR, l):
    x0 = np.zeros(DTR.shape[0] + 1)
    return fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l), approx_grad=False)[0]


def compute_logreg_scores(D, model):
    w, b = model[:-1], model[-1]
    return np.dot(w.T, D) + b


def compute_logreg_predictions(scores):
    return (scores >= 0).astype(int)


def plot_metrics(lambdas, actual_dcf, min_dcf, title):
    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, actual_dcf, label='Actual DCF')
    plt.plot(lambdas, min_dcf, label='Min DCF')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("Output/LogFigures", title + ".png"))


def train_LR(DTE, DTR, LTE, LTR):
    l_values = np.logspace(-4, 2, 13)
    pi_t_values = [0.1]  # Only used for prior-weighted configuration
    configurations = [
        {'name': 'Normal', 'prior_weighted': False, 'quadratic': False},
        {'name': 'Prior-Weighted', 'prior_weighted': True, 'quadratic': False},
        {'name': 'Quadratic', 'prior_weighted': False, 'quadratic': True},
    ]
    print("LogREg 1")
    results = []
    models_log = {}
    for config in configurations:
        for l in l_values:
            if config['quadratic']:
                DTR_exp = expand_features(DTR)
                DTE_exp = expand_features(DTE)
            else:
                DTR_exp, DTE_exp = DTR, DTE

            if config['prior_weighted']:
                for pi_t in pi_t_values:
                    config_name = f"{config['name']} (pi_t={pi_t})"
                    logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=True, pi_t=pi_t)
                    logreg_classifier.train()
                    scores = logreg_classifier.predict(DTE_exp)
                    predictions = logreg_classifier.compute_predictions(scores)
                    error_rate = logreg_classifier.compute_error_rate(predictions, LTE)
                    print(f"Config: {config_name}, Lambda: {l}, Error Rate: {error_rate}")
                    results.append((config_name, l, error_rate))
                    models_log[config_name] = logreg_classifier
            else:
                config_name = config['name']
                logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=False)
                logreg_classifier.train()
                scores = logreg_classifier.predict(DTE_exp)
                predictions = logreg_classifier.compute_predictions(scores)
                error_rate = logreg_classifier.compute_error_rate(predictions, LTE)
                print(f"Config: {config_name}, Lambda: {l}, Error Rate: {error_rate}")
                results.append((config_name, l, error_rate))
                models_log[config_name] = logreg_classifier
    print("LogREg 2")
    # Use the new plotting function
    LogRegClass.plot_error_rates(results, output_file='error_rates.png')
    # PROJECT PART
    # Change the Output paths to save them in the /Output folder
    output_dir = "Output/LogisticRegression_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lambdas = np.logspace(-4, 2, 13)
    analyze_dcf_vs_lambda(DTR, LTR, DTE, LTE, lambdas)
    analyze_with_fewer_samples(DTR, LTR, DTE, LTE, lambdas)
    analyze_prior_weighted(DTR, LTR, DTE, LTE, lambdas)
    analyze_quadratic(DTR, LTR, DTE, LTE, lambdas)
    analyze_preprocessing(DTR, LTR, DTE, LTE, lambdas, output_file_prefix=os.path.join(output_dir, 'dcf_vs_lambda'))
    results = compare_all_models(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1)
    # Print results for comparison
    for model_name, (actual_dcfs, min_dcfs) in results.items():
        print(f"Results for {model_name}:")
        for l, actual_dcf, min_dcf in zip(lambdas, actual_dcfs, min_dcfs):
            print(f"Lambda: {l}, Actual DCF: {actual_dcf}, Min DCF: {min_dcf}")
    # Plot ROC and Bayes error plots for selected models
    selected_models = ['Normal', 'Prior-Weighted', 'Quadratic']
    for model_name in selected_models:
        model_key = f"{model_name} (pi_t=0.1)" if model_name == 'Prior-Weighted' else model_name
        print("Iterating, now on " + model_key)
        model = models_log[model_key]
        if model_name == 'Quadratic':
            DTE_exp = expand_features(DTE)
            scores = model.predict(DTE_exp)
        else:
            scores = model.predict(DTE)

        # Compute and remove the log-odds of the empirical prior for actual DCF
        if 'Prior-Weighted' in model_key:
            emp_prior = np.mean(LTR)
            scores = scores - np.log(emp_prior / (1 - emp_prior))

        # Compute actual DCF
        actual_dcf = model.compute_dcf(scores, LTE, pi_t=0.1)
        print(f"{model_name} Actual DCF: {actual_dcf}")

        # Compute min DCF
        min_dcf = model.compute_min_dcf(scores, LTE, pi_t=0.1)
        print(f"{model_name} Min DCF: {min_dcf}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(LTE, scores)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, model_name, os.path.join(output_dir, f'ROC_{model_name}.png'))

        # Plot Bayes error plot
        plot_bayes_error(scores, LTE, np.linspace(-3, 3, 21),
                         os.path.join(output_dir, f'Bayes_Error_{model_name}.png'))
