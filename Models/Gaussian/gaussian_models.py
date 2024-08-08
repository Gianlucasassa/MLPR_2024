import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as confMAt

from Models.Gaussian.gaussian_density import *
from Models.bayesRisk import compute_optimal_Bayes_binary_llr, compute_empirical_Bayes_risk_binary, \
    compute_minDCF_binary_slow, compute_minDCF_binary_fast, plot_ROC_curve, plot_bayes_error, compute_confusion_matrix, \
    plot_confusion_matrix
from Preprocess.PCA import *


# from gaussian_density we have logpdf_GAU_ND, compute_mu_C

def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


class Gaussian:
    def __init__(self, model_type='MVG'):
        self.model_type = model_type
        self.params = None

    def Gau_MVG_ML_estimates(self, D, L):
        labelSet = set(L)
        hParams = {}
        for lab in labelSet:
            DX = D[:, L == lab]
            hParams[lab] = compute_mu_C(DX)
        return hParams

    def Gau_Naive_ML_estimates(self, D, L):
        labelSet = set(L)
        hParams = {}
        for lab in labelSet:
            DX = D[:, L == lab]
            mu, C = compute_mu_C(DX)
            hParams[lab] = (mu, C * np.eye(D.shape[0]))
        return hParams

    def Gau_Tied_ML_estimates(self, D, L):
        labelSet = set(L)
        hParams = {}
        hMeans = {}
        CGlobal = 0
        for lab in labelSet:
            DX = D[:, L == lab]
            mu, C_class = compute_mu_C(DX)
            CGlobal += C_class * DX.shape[1]
            hMeans[lab] = mu
        CGlobal = CGlobal / D.shape[1]
        for lab in labelSet:
            hParams[lab] = (hMeans[lab], CGlobal)
        return hParams

    def compute_log_likelihood_Gau(self, D, hParams):
        S = np.zeros((len(hParams), D.shape[1]))
        for lab in range(S.shape[0]):
            S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
        return S

    def compute_logPosterior(self, S_logLikelihood, v_prior):
        SJoint = S_logLikelihood + vcol(np.log(v_prior))
        SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
        SPost = SJoint - SMarginal
        return SPost

    def train(self, DTR, LTR):
        if self.model_type == 'MVG':
            self.params = self.Gau_MVG_ML_estimates(DTR, LTR)
        elif self.model_type == 'Naive':
            self.params = self.Gau_Naive_ML_estimates(DTR, LTR)
        elif self.model_type == 'Tied':
            self.params = self.Gau_Tied_ML_estimates(DTR, LTR)
        else:
            raise ValueError("Unsupported model type")

    def predict(self, DTE, priors):
        if self.params is None:
            raise ValueError("The model has not been trained yet")
        logLikelihood = self.compute_log_likelihood_Gau(DTE, self.params)
        logPosterior = self.compute_logPosterior(logLikelihood, priors)
        return logPosterior

    def compute_llr(self, D, hParams, class1, class2):
        return logpdf_GAU_ND(D, hParams[class2][0], hParams[class2][1]) - logpdf_GAU_ND(D, hParams[class1][0], hParams[class1][1])

    def evaluate_2class(self, DTR, LTR, DTE, LTE, class1, class2):
        priors = np.ones(2) / 2.

        params = None
        if self.model_type == 'MVG':
            params = self.Gau_MVG_ML_estimates(DTR, LTR)
        elif self.model_type == 'Tied':
            params = self.Gau_Tied_ML_estimates(DTR, LTR)
        elif self.model_type == 'Naive':
            params = self.Gau_Naive_ML_estimates(DTR, LTR)

        if params:
            LLR = self.compute_llr(DTE, params, class1, class2)
            PVAL = np.zeros(DTE.shape[1], dtype=np.int32)
            PVAL[LLR >= 0] = class2
            PVAL[LLR < 0] = class1
            error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
            print(f"{self.model_type} - Error rate: %.1f%%" % error_rate)

    def print_covariance_and_correlation_matrices(self, params):
        for label, (mu, C) in params.items():
            print(f"Class {label} covariance matrix:\n{C}\n")
            correlation_matrix = C / (vcol(np.sqrt(C.diagonal())) * vrow(np.sqrt(C.diagonal())))
            print(f"Class {label} correlation matrix:\n{correlation_matrix}\n")



def train_Lab_5(DTR, LTR, DTE, LTE):
    priors = np.ones(2) / 2.  # two priors of 1/2

    print("Full Feature Set")
    models = {
        "MVG": Gaussian(model_type='MVG'),
        "Naive": Gaussian(model_type='Naive'),
        "Tied": Gaussian(model_type='Tied')
    }

    error_rates = {}
    for model_name, model in models.items():
        model.train(DTR, LTR)
        logPosterior = model.predict(DTE, priors)
        PVAL = logPosterior.argmax(0)
        error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
        error_rates[model_name] = error_rate
        print(f"{model_name} - Error rate: {error_rate:.1f}%")

    # Binary task:
    print("Binary task")
    for model_name, model in models.items():
        params = model.Gau_MVG_ML_estimates(DTR, LTR) if model.model_type == 'MVG' else model.Gau_Naive_ML_estimates(DTR, LTR) if model.model_type == 'Naive' else model.Gau_Tied_ML_estimates(DTR, LTR)
        LLR = model.compute_llr(DTE, params, 0, 1)
        PVAL = np.zeros(DTE.shape[1], dtype=np.int32)
        PVAL[LLR >= 0] = 1
        PVAL[LLR < 0] = 0
        print(f"{model_name} - Error rate: {(PVAL != LTE).sum() / float(LTE.size) * 100:.1f}%")

    # Analyze covariance and correlation matrices
    for model_name, model in models.items():
        params = model.Gau_MVG_ML_estimates(DTR, LTR) if model.model_type == 'MVG' else model.Gau_Naive_ML_estimates(DTR, LTR) if model.model_type == 'Naive' else model.Gau_Tied_ML_estimates(DTR, LTR)
        print(f"\nCovariance and Correlation Matrices for {model_name} Model:")
        model.print_covariance_and_correlation_matrices(params)

    # Using features 1 to 4 only
    print("\nUsing Features 1 to 4 Only")
    DTR_4 = DTR[:4, :]
    DTE_4 = DTE[:4, :]

    error_rates_4 = {}
    for model_name, model in models.items():
        model.train(DTR_4, LTR)
        logPosterior = model.predict(DTE_4, priors)
        PVAL = logPosterior.argmax(0)
        error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
        error_rates_4[model_name] = error_rate
        print(f"{model_name} - Error rate: {error_rate:.1f}%")

    print("\nComparing error rates:")
    for model_name in error_rates:
        print(f"{model_name} - Full feature set error rate: {error_rates[model_name]:.1f}%")
        print(f"{model_name} - Features 1 to 4 error rate: {error_rates_4[model_name]:.1f}%")

    # Binary classification task with reduced features
    print("\nBinary task with Features 1 to 4 Only")
    for model_name, model in models.items():
        model.evaluate_2class(DTR_4, LTR, DTE_4, LTE, class1=0, class2=1)

    # Analyze covariance and correlation matrices for reduced features
    for model_name, model in models.items():
        params = model.Gau_MVG_ML_estimates(DTR_4, LTR) if model.model_type == 'MVG' else model.Gau_Naive_ML_estimates(DTR_4, LTR) if model.model_type == 'Naive' else model.Gau_Tied_ML_estimates(DTR_4, LTR)
        print(f"\nCovariance and Correlation Matrices for {model_name} Model (Features 1 to 4):")
        model.print_covariance_and_correlation_matrices(params)

    # Using features 1 and 2 only
    print("\nUsing Features 1 and 2 Only")
    DTR_12 = DTR[:2, :]
    DTE_12 = DTE[:2, :]
    for model_name, model in models.items():
        model.train(DTR_12, LTR)
        logPosterior = model.predict(DTE_12, priors)
        PVAL = logPosterior.argmax(0)
        error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
        print(f"{model_name} - Error rate: {error_rate:.1f}%")

    # Using features 3 and 4 only
    print("\nUsing Features 3 and 4 Only")
    DTR_34 = DTR[2:4, :]
    DTE_34 = DTE[2:4, :]
    for model_name, model in models.items():
        model.train(DTR_34, LTR)
        logPosterior = model.predict(DTE_34, priors)
        PVAL = logPosterior.argmax(0)
        error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
        print(f"{model_name} - Error rate: {error_rate:.1f}%")

    # PCA Preprocessing
    print("\nPCA Preprocessing")
    m = 2
    UPCA = compute_pca(DTR, m=m)  # Estimated only on model training data
    DTR_pca = apply_pca(UPCA, DTR)  # Applied to original model training data
    DTE_pca = apply_pca(UPCA, DTE)  # Applied to original validation data

    for model_name, model in models.items():
        model.train(DTR_pca, LTR)
        logPosterior = model.predict(DTE_pca, priors)
        PVAL = logPosterior.argmax(0)
        error_rate = (PVAL != LTE).sum() / float(LTE.size) * 100
        print(f"{model_name} - Error rate with PCA: {error_rate:.1f}%")

    return



def train_Lab_7(DTR, LTR, DTE, LTE):
    output_dir_base = 'Output/Gaussian'
    applications = [
        (0.5, 1.0, 1.0),  # uniform prior and costs
        (0.9, 1.0, 1.0),  # higher prior probability of genuine samples
        (0.1, 1.0, 1.0),  # higher prior probability of fake samples
        (0.5, 1.0, 9.0),  # higher cost for accepting a fake image
        (0.5, 9.0, 1.0)   # higher cost for rejecting a legit image
    ]
    model = Gaussian(model_type='MVG')
    priors = np.ones(2) / 2.  # uniform priors for two classes

    # Train the model
    model.train(DTR, LTR)

    # Get log-likelihood ratio
    LLR = model.compute_llr(DTE, model.Gau_MVG_ML_estimates(DTR, LTR), 0, 1)

    # Plot ROC curve (same for all applications)
    plot_ROC_curve(LLR, LTE, output_dir=os.path.join(output_dir_base, 'Common'))

    # Bayes Error Plotting (same for all applications)
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        predictions_binary = compute_optimal_Bayes_binary_llr(LLR, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions_binary, LTE, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(LLR, LTE, effPrior, 1.0, 1.0))
    plot_bayes_error(effPriors, actDCF, minDCF, output_dir=os.path.join(output_dir_base, 'Common'))

    # Evaluate each application
    for prior, Cfn, Cfp in applications:
        output_dir = os.path.join(output_dir_base, f'Application_pi_{prior}_Cfn{Cfn}_Cfp{Cfp}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"\nApplication with Prior: {prior}, Cfn: {Cfn}, Cfp: {Cfp}")

        predictions_binary = compute_optimal_Bayes_binary_llr(LLR, prior, Cfn, Cfp)

        # Compute and print confusion matrix
        confusion_matrix = compute_confusion_matrix(predictions_binary, LTE)
        print("Confusion Matrix:")
        print(confusion_matrix)
        plot_confusion_matrix(confusion_matrix, class_labels=["Fake", "Genuine"], output_dir=output_dir)

        # Compute and print DCF
        dcf_non_normalized = compute_empirical_Bayes_risk_binary(
            predictions_binary, LTE, prior, Cfn, Cfp, normalize=False)
        dcf_normalized = compute_empirical_Bayes_risk_binary(
            predictions_binary, LTE, prior, Cfn, Cfp)
        print(f'DCF (non-normalized): {dcf_non_normalized:.3f}')
        print(f'DCF (normalized): {dcf_normalized:.3f}')

        minDCF_fast, _ = compute_minDCF_binary_fast(LLR, LTE, prior, Cfn, Cfp, returnThreshold=True)
        print(f'MinDCF (normalized, fast): {minDCF_fast:.3f}')

    # Evaluate each model with different covariance types
    for model_type in ['MVG', 'Naive', 'Tied']:
        model = Gaussian(model_type=model_type)
        model.train(DTR, LTR)
        LLR = model.compute_llr(DTE, model.Gau_MVG_ML_estimates(DTR, LTR), 0, 1)

        print(f"\nEvaluating {model_type} model")

        for prior, Cfn, Cfp in applications:
            output_dir = os.path.join(output_dir_base, f'{model_type}_Application_pi_{prior}_Cfn{Cfn}_Cfp{Cfp}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"\nApplication with Prior: {prior}, Cfn: {Cfn}, Cfp: {Cfp}")

            predictions_binary = compute_optimal_Bayes_binary_llr(LLR, prior, Cfn, Cfp)

            # Compute and print confusion matrix
            confusion_matrix = compute_confusion_matrix(predictions_binary, LTE)
            print("Confusion Matrix:")
            print(confusion_matrix)
            plot_confusion_matrix(confusion_matrix, class_labels=["Fake", "Genuine"], output_dir=output_dir)

            # Compute and print DCF
            dcf_non_normalized = compute_empirical_Bayes_risk_binary(
                predictions_binary, LTE, prior, Cfn, Cfp, normalize=False)
            dcf_normalized = compute_empirical_Bayes_risk_binary(
                predictions_binary, LTE, prior, Cfn, Cfp)
            print(f'DCF (non-normalized): {dcf_non_normalized:.3f}')
            print(f'DCF (normalized): {dcf_normalized:.3f}')

            minDCF_fast, _ = compute_minDCF_binary_fast(LLR, LTE, prior, Cfn, Cfp, returnThreshold=True)
            print(f'MinDCF (normalized, fast): {minDCF_fast:.3f}')


    #Second part
    effective_priors = [0.1, 0.5, 0.9]
    Cfn, Cfp = 1.0, 1.0  # Costs are equal to 1 for the effective prior evaluation
    m_values = [2, 4, 6]  # Different values of m for PCA
    model = Gaussian(model_type='MVG')

    # Function to evaluate and print DCFs
    def evaluate_and_print_dcf(LLR, true_labels, eff_prior, Cfn, Cfp, model_name, output_dir):
        predictions_binary = compute_optimal_Bayes_binary_llr(LLR, eff_prior, Cfn, Cfp)
        dcf_non_normalized = compute_empirical_Bayes_risk_binary(predictions_binary, true_labels, eff_prior, Cfn, Cfp,
                                                                 normalize=False)
        dcf_normalized = compute_empirical_Bayes_risk_binary(predictions_binary, true_labels, eff_prior, Cfn, Cfp)
        minDCF_fast, _ = compute_minDCF_binary_fast(LLR, true_labels, eff_prior, Cfn, Cfp, returnThreshold=True)

        print(f"{model_name} - Effective Prior: {eff_prior}")
        print(f'DCF (non-normalized): {dcf_non_normalized:.3f}')
        print(f'DCF (normalized): {dcf_normalized:.3f}')
        print(f'MinDCF (normalized, fast): {minDCF_fast:.3f}')

        return dcf_normalized, minDCF_fast

    # Evaluate for each application
    for eff_prior in effective_priors:
        print(f"\nEvaluating for Effective Prior: {eff_prior}")

        # Without PCA
        model.train(DTR, LTR)
        LLR_MVG = model.compute_llr(DTE, model.Gau_MVG_ML_estimates(DTR, LTR), 0, 1)
        dcf_MVG, minDCF_MVG = evaluate_and_print_dcf(LLR_MVG, LTE, eff_prior, Cfn, Cfp, "MVG", output_dir_base)

        model.model_type = 'Tied'
        model.train(DTR, LTR)
        LLR_Tied = model.compute_llr(DTE, model.Gau_Tied_ML_estimates(DTR, LTR), 0, 1)
        dcf_Tied, minDCF_Tied = evaluate_and_print_dcf(LLR_Tied, LTE, eff_prior, Cfn, Cfp, "Tied MVG", output_dir_base)

        model.model_type = 'Naive'
        model.train(DTR, LTR)
        LLR_Naive = model.compute_llr(DTE, model.Gau_Naive_ML_estimates(DTR, LTR), 0, 1)
        dcf_Naive, minDCF_Naive = evaluate_and_print_dcf(LLR_Naive, LTE, eff_prior, Cfn, Cfp, "Naive MVG",
                                                         output_dir_base)

        # With PCA
        for m in m_values:
            UPCA = compute_pca(DTR, m=m)
            DTR_pca = apply_pca(UPCA, DTR)
            DTE_pca = apply_pca(UPCA, DTE)

            model.model_type = 'MVG'
            model.train(DTR_pca, LTR)
            LLR_MVG_pca = model.compute_llr(DTE_pca, model.Gau_MVG_ML_estimates(DTR_pca, LTR), 0, 1)
            dcf_MVG_pca, minDCF_MVG_pca = evaluate_and_print_dcf(LLR_MVG_pca, LTE, eff_prior, Cfn, Cfp, f"MVG_PCA_m{m}",
                                                                 output_dir_base)

            model.model_type = 'Tied'
            model.train(DTR_pca, LTR)
            LLR_Tied_pca = model.compute_llr(DTE_pca, model.Gau_Tied_ML_estimates(DTR_pca, LTR), 0, 1)
            dcf_Tied_pca, minDCF_Tied_pca = evaluate_and_print_dcf(LLR_Tied_pca, LTE, eff_prior, Cfn, Cfp,
                                                                   f"Tied_MVG_PCA_m{m}", output_dir_base)

            model.model_type = 'Naive'
            model.train(DTR_pca, LTR)
            LLR_Naive_pca = model.compute_llr(DTE_pca, model.Gau_Naive_ML_estimates(DTR_pca, LTR), 0, 1)
            dcf_Naive_pca, minDCF_Naive_pca = evaluate_and_print_dcf(LLR_Naive_pca, LTE, eff_prior, Cfn, Cfp,
                                                                     f"Naive_MVG_PCA_m{m}", output_dir_base)

    # Third part - Bayes error plots for the best PCA setup
    best_m = 6  # Based on previous results for effective prior 0.1
    UPCA_best = compute_pca(DTR, m=best_m)
    DTR_pca_best = apply_pca(UPCA_best, DTR)
    DTE_pca_best = apply_pca(UPCA_best, DTE)
    model = Gaussian()

    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    def plot_bayes_error_for_model(model_name, LLR, true_labels, output_dir):
        actDCF = []
        minDCF = []
        for effPrior in effPriors:
            predictions_binary = compute_optimal_Bayes_binary_llr(LLR, effPrior, 1.0, 1.0)
            actDCF.append(compute_empirical_Bayes_risk_binary(predictions_binary, true_labels, effPrior, 1.0, 1.0))
            minDCF.append(compute_minDCF_binary_fast(LLR, true_labels, effPrior, 1.0, 1.0))

        plot_bayes_error(effPriors, actDCF, minDCF, output_dir=output_dir)
        return actDCF, minDCF

    # Evaluate MVG
    model.model_type = 'MVG'
    params_MVG_best = model.Gau_MVG_ML_estimates(DTR_pca_best, LTR)
    LLR_MVG_best = model.compute_llr(DTE_pca_best, params_MVG_best, 0, 1)
    actDCF_MVG, minDCF_MVG = plot_bayes_error_for_model("MVG", LLR_MVG_best, LTE,
                                                        output_dir=os.path.join(output_dir_base, 'best_pi_0.1',
                                                                                'Bayes_Error_MVG'))

    # Evaluate Tied MVG
    model.model_type = 'Tied'
    params_Tied_best = model.Gau_Tied_ML_estimates(DTR_pca_best, LTR)
    LLR_Tied_best = model.compute_llr(DTE_pca_best, params_Tied_best, 0, 1)
    actDCF_Tied, minDCF_Tied = plot_bayes_error_for_model("Tied MVG", LLR_Tied_best, LTE,
                                                          output_dir=os.path.join(output_dir_base, 'best_pi_0.1',
                                                                                  'Bayes_Error_Tied_MVG'))

    # Evaluate Naive MVG
    model.model_type = 'Naive'
    params_Naive_best = model.Gau_Naive_ML_estimates(DTR_pca_best, LTR)
    LLR_Naive_best = model.compute_llr(DTE_pca_best, params_Naive_best, 0, 1)
    actDCF_Naive, minDCF_Naive = plot_bayes_error_for_model("Naive MVG", LLR_Naive_best, LTE,
                                                            output_dir=os.path.join(output_dir_base, 'best_pi_0.1',
                                                                                    'Bayes_Error_Naive_MVG'))

    # Plot all three models together
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF_MVG, label='MVG actDCF', color='r', linestyle='-')
    plt.plot(effPriorLogOdds, minDCF_MVG, label='MVG minDCF', color='r', linestyle='--')
    plt.plot(effPriorLogOdds, actDCF_Tied, label='Tied MVG actDCF', color='g', linestyle='-')
    plt.plot(effPriorLogOdds, minDCF_Tied, label='Tied MVG minDCF', color='g', linestyle='--')
    plt.plot(effPriorLogOdds, actDCF_Naive, label='Naive MVG actDCF', color='b', linestyle='-')
    plt.plot(effPriorLogOdds, minDCF_Naive, label='Naive MVG minDCF', color='b', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel("Prior Log Odds")
    plt.ylabel("DCF Value")
    plt.title("Bayes Error Plot for Best PCA Setup")
    plt.legend()
    plt.savefig(os.path.join(output_dir_base, 'best_pi_0.1', 'Bayes_Error_Combined.png'))
    plt.close()

    return

















#
#
#
# def compute_dcf(predictions, labels, pi1, Cfn, Cfp):
#     confusion_matrix = compute_confusion_matrix(predictions, labels)
#     bayes_risk = compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
#     normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
#     return normalized_dcf
#
#
# def compute_min_dcf(llrs, labels, pi1, Cfn, Cfp):
#     thresholds = np.sort(llrs)
#     min_dcf = float('inf')
#     for t in thresholds:
#         predictions = (llrs >= t).astype(int)
#         confusion_matrix = compute_confusion_matrix(predictions, labels)
#         bayes_risk = compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
#         normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
#         if normalized_dcf < min_dcf:
#             min_dcf = normalized_dcf
#     return min_dcf
#
#
# class GaussianClassifier:
#     def __init__(self, model_type='MVG'):
#         self.means = None
#         self.covariances = None
#         self.model_type = model_type
#
#     def logpdf_GAU_ND(self, X, mu, C):
#         M = X.shape[0]
#         inv_C = np.linalg.inv(C)
#         log_det_C = np.linalg.slogdet(C)[1]
#         diff = X - mu
#         log_density = -0.5 * (M * np.log(2 * np.pi) + log_det_C + np.sum(diff * np.dot(inv_C, diff), axis=0))
#         return log_density
#
#     def compute_ml_estimates(self, D, L):
#         classes = np.unique(L)
#         means = []
#         covariances = []
#         for cls in classes:
#             class_data = D[:, L == cls]
#             mu_ML = class_data.mean(axis=1, keepdims=True)
#             if self.model_type == 'NaiveBayes':
#                 C_ML = np.diag(np.diag(np.cov(class_data)))
#             else:
#                 C_ML = np.cov(class_data)
#             means.append(mu_ML)
#             covariances.append(C_ML)
#         if self.model_type == 'TiedCovariance':
#             covariances = [np.mean(covariances, axis=0)] * len(classes)
#         return means, covariances
#
#     def train(self, DTR, LTR):
#         self.means, self.covariances = self.compute_ml_estimates(DTR, LTR)
#
#     def predict(self, DTE):
#         logS = []
#         for mu, C in zip(self.means, self.covariances):
#             logS.append(self.logpdf_GAU_ND(DTE, mu, C))
#         return np.array(logS)
#
#     def compute_llrs(self, logS):
#         return logS[1] - logS[0]
#
#     def compute_predictions(self, logS, threshold=0):
#         llrs = self.compute_llrs(logS)
#         return (llrs >= threshold).astype(int)
#
#     def compute_error_rate(self, predictions, true_labels):
#         return np.mean(predictions != true_labels)
#
#     def evaluate(self, DTR, LTR, DTE, LTE, output_dir='Output/ClassifierResults'):
#         os.makedirs(output_dir, exist_ok=True)
#         self.train(DTR, LTR)
#         logS = self.predict(DTE)
#         predictions = self.compute_predictions(logS)
#         error_rate = self.compute_error_rate(predictions, LTE)
#         self.plot_classification_results(DTE, LTE, predictions,
#                                          os.path.join(output_dir, f'{self.model_type}_Results.png'))
#         return error_rate
#
#     def plot_classification_results(self, D, L_true, L_pred, output_file):
#         plt.figure()
#         for cls in np.unique(L_true):
#             plt.scatter(D[0, L_true == cls], D[1, L_true == cls], label=f'True Class {cls}', alpha=0.5)
#             plt.scatter(D[0, L_pred == cls], D[1, L_pred == cls], marker='x', label=f'Predicted Class {cls}')
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.legend()
#         plt.title('Classification Results')
#         plt.savefig(output_file)
#         plt.close()
#
#     def ffit_univariate_gaussian_models(self, D, L, output_dir='Output/UnivariateGaussians'):
#         os.makedirs(output_dir, exist_ok=True)
#         classes = np.unique(L)
#         for cls in classes:
#             for i in range(D.shape[0]):
#                 feature_data = D[i, L == cls]
#                 mu_ML = feature_data.mean()
#                 C_ML = feature_data.var()
#                 XPlot = np.linspace(feature_data.min(), feature_data.max(), 1000)
#                 plt.figure()
#                 plt.hist(feature_data, bins=50, density=True, alpha=0.6, color='g')
#                 plt.plot(XPlot,
#                          np.exp(self.logpdf_GAU_ND(XPlot.reshape(1, -1), np.array([[mu_ML]]), np.array([[C_ML]]))))
#                 plt.title(f'Class {cls}, Feature {i + 1}')
#                 plt.xlabel('Value')
#                 plt.ylabel('Density')
#                 plt.savefig(os.path.join(output_dir, f'Class_{cls}_Feature_{i + 1}.png'))
#                 plt.close()
#
#     def logpdf_GAU_1D(self, X, mu, var):
#         log_density = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
#         return log_density
#
#     def vrow(self, col):
#         return col.reshape((1, col.size))
#
#     def vcol(self, row):
#         return row.reshape((row.size, 1))
#
#     def plot_loglikelihood(self, X, mu, C, output_file):
#         ll = np.sum(self.logpdf_GAU_ND(X, mu, C))
#         plt.figure()
#         plt.hist(X.ravel(), bins=50, density=True)
#         XPlot = np.linspace(X.min(), X.max(), 1000)
#         plt.plot(XPlot.ravel(), np.exp(self.logpdf_GAU_ND(self.vrow(XPlot), mu, C)))
#         plt.title(f'Log-Likelihood: {ll}')
#         plt.xlabel('Value')
#         plt.ylabel('Density')
#         plt.savefig(output_file)
#         plt.close()
#         return ll
#
#     def analyze_covariances(self, DTR, LTR, output_dir='Output/Covariances'):
#         os.makedirs(output_dir, exist_ok=True)
#         means, covariances = self.compute_ml_estimates(DTR, LTR)
#         for cls, (mu, C) in enumerate(zip(means, covariances)):
#             print(f"Class {cls} Mean:\n{mu}")
#             print(f"Class {cls} Covariance Matrix:\n{C}")
#             corr = C / (self.vcol(np.diag(C) ** 0.5) * self.vrow(np.diag(C) ** 0.5))
#             print(f"Class {cls} Correlation Matrix:\n{corr}")
#             self.plot_matrix(C, os.path.join(output_dir, f'Class_{cls}_Covariance.png'))
#             self.plot_matrix(corr, os.path.join(output_dir, f'Class_{cls}_Correlation.png'))
#
#     def plot_matrix(self, matrix, output_file):
#         plt.figure()
#         plt.imshow(matrix, interpolation='nearest', cmap='coolwarm')
#         plt.colorbar()
#         plt.title('Matrix Heatmap')
#         plt.savefig(output_file)
#         plt.close()
#
#     # Additional methods for confusion matrix, Bayes decisions, empirical Bayes risk, normalized detection cost, ROC curves, and Bayes error plots
#
#     def compute_confusion_matrix(self, predictions, labels):
#         K = len(np.unique(labels))
#         confusion_matrix = np.zeros((K, K), dtype=int)
#         for i in range(len(labels)):
#             confusion_matrix[predictions[i], labels[i]] += 1
#         return confusion_matrix
#
#     def compute_optimal_bayes_decisions_class(self, llrs, pi1, Cfn, Cfp):
#         t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
#         return (llrs >= t).astype(int)
#
#     def compute_bayes_risk(self, confusion_matrix, pi1, Cfn, Cfp):
#         FN = confusion_matrix[0, 1]
#         FP = confusion_matrix[1, 0]
#         TN = confusion_matrix[0, 0]
#         TP = confusion_matrix[1, 1]
#         Pfn = FN / (FN + TP)
#         Pfp = FP / (FP + TN)
#         DCFu = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
#         return DCFu
#
#     def compute_normalized_dcf(self, bayes_risk, pi1, Cfn, Cfp):
#         Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
#         return bayes_risk / Bdummy
#
#     def plot_roc_curve(self, llrs, labels, output_file='roc_curve.png'):
#         out_dir = "Output/MVG/Roc"
#         os.makedirs(out_dir, exist_ok=True)
#         thresholds = np.sort(llrs)
#         TPR = []
#         FPR = []
#         for t in thresholds:
#             predictions = (llrs >= t).astype(int)
#             confusion_matrix = self.compute_confusion_matrix(predictions, labels)
#             FN = confusion_matrix[0, 1]
#             FP = confusion_matrix[1, 0]
#             TN = confusion_matrix[0, 0]
#             TP = confusion_matrix[1, 1]
#             TPR.append(TP / (TP + FN))
#             FPR.append(FP / (FP + TN))
#         plt.plot(FPR, TPR, marker='.')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC Curve')
#         plt.grid()
#         plt.savefig(os.path.join(out_dir, output_file))
#         plt.close()
#
#     def plot_bayes_error(self, llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
#         out_dir = 'Output/Bayes'
#         os.makedirs(out_dir, exist_ok=True)
#         dcf = []
#         mindcf = []
#         for p in effPriorLogOdds:
#             pi1 = 1 / (1 + np.exp(-p))
#             decisions = self.compute_optimal_bayes_decisions_class(llrs, pi1, 1, 1)
#             confusion_matrix = self.compute_confusion_matrix(decisions, labels)
#             bayes_risk = self.compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#             normalized_dcf = self.compute_normalized_dcf(bayes_risk, pi1, 1, 1)
#             dcf.append(normalized_dcf)
#
#             min_bayes_risk = self.compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#             min_normalized_dcf = self.compute_normalized_dcf(min_bayes_risk, pi1, 1, 1)
#             mindcf.append(min_normalized_dcf)
#
#         plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
#         plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
#         plt.ylim([0, 1.1])
#         plt.xlim([-3, 3])
#         plt.xlabel('Prior Log-Odds')
#         plt.ylabel('DCF value')
#         plt.legend()
#         plt.grid()
#         plt.title('Bayes Error Plot')
#         plt.savefig(os.path.join(out_dir, output_file))
#         plt.close()
#
#
# def compute_confusion_matrix(predictions, labels):
#     K = 2  # Since it's a binary classification problem
#     confusion_matrix = np.zeros((K, K), dtype=int)
#
#     # Flatten predictions if it's a 2D array
#     if predictions.ndim > 1:
#         predictions = predictions.flatten()
#
#     # print(f"Predictions: {predictions}")
#     # print(f"Labels: {labels}")
#
#     for i in range(len(labels)):
#         pred = int(predictions[i])
#         true = int(labels[i])
#         if pred < 0 or pred >= K or true < 0 or true >= K:
#             # print(f"Invalid prediction or label: pred={pred}, true={true}")
#             continue  # Skip invalid values
#         confusion_matrix[pred, true] += 1
#
#     return confusion_matrix
#
#
# def compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
#     FN = confusion_matrix[0, 1]
#     FP = confusion_matrix[1, 0]
#     TN = confusion_matrix[0, 0]
#     TP = confusion_matrix[1, 1]
#     Pfn = FN / (FN + TP)
#     Pfp = FP / (FP + TN)
#     DCFu = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
#     return DCFu
#
#
# def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
#     out_dir = os.path.dirname(output_file)
#     os.makedirs(out_dir, exist_ok=True)
#     dcf = []
#     mindcf = []
#     for p in effPriorLogOdds:
#         pi1 = 1 / (1 + np.exp(-p))
#         decisions = compute_optimal_bayes_decisions_main(llrs, pi1, 1, 1)
#         confusion_matrix = compute_confusion_matrix(decisions, labels)
#         bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#         normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, 1, 1)
#         dcf.append(normalized_dcf)
#
#         min_bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#         min_normalized_dcf = compute_normalized_dcf(min_bayes_risk, pi1, 1, 1)
#         mindcf.append(min_normalized_dcf)
#
#     plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
#     plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
#     plt.ylim([0, 1.1])
#     plt.xlim([-3, 3])
#     plt.xlabel('Prior Log-Odds')
#     plt.ylabel('DCF value')
#     plt.legend()
#     plt.grid()
#     plt.title('Bayes Error Plot')
#     plt.savefig(output_file)
#     plt.close()
#
#
# def compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp):
#     Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
#     return bayes_risk / Bdummy
#
#
# def plot_confusion_matrix(cm, model_name, output_file):
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f'Confusion Matrix - {model_name}')
#     plt.savefig(output_file)
#     plt.close()
#
#
# def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file='roc_curve.png'):
#     out_dir = os.path.dirname(output_file)
#     os.makedirs(out_dir, exist_ok=True)
#
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic - {model_name}')
#     plt.legend(loc="lower right")
#     plt.grid()
#     plt.savefig(output_file)
#     plt.close()
#
#
# def compute_optimal_bayes_decisions_main(llrs, pi1, Cfn, Cfp):
#     t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
#     return (llrs >= t).astype(int)
#
#
# def train_MVG_1(DTE, DTR, LTE, LTR):
#     def evaluate_model(classifier, model_name, DTR, LTR, DTE, LTE, output_dir):
#         classifier.train(DTR, LTR)
#         logS = classifier.predict(DTE)
#         predictions = classifier.compute_predictions(logS)
#         error_rate = classifier.compute_error_rate(predictions, LTE)
#         print(f"{model_name} Error Rate: {error_rate}")
#
#         fpr, tpr, _ = roc_curve(LTE, logS[1])
#         roc_auc = auc(fpr, tpr)
#         plot_roc_curve(fpr, tpr, roc_auc, model_name, os.path.join(output_dir, f'ROC_{model_name}.png'))
#
#         cm = confMAt(LTE, predictions)
#         plot_confusion_matrix(cm, model_name, os.path.join(output_dir, f'CM_{model_name}.png'))
#
#         return error_rate
#
#     output_dir = 'Output/ClassifierResults'
#     # MVG
#     mvg_classifier = GaussianClassifier(model_type='MVG')
#     evaluate_model(mvg_classifier, 'MVG', DTR, LTR, DTE, LTE, output_dir)
#     # Naive Bayes
#     nb_classifier = GaussianClassifier(model_type='NaiveBayes')
#     evaluate_model(nb_classifier, 'NaiveBayes', DTR, LTR, DTE, LTE, output_dir)
#     # Tied Covariance
#     tied_classifier = GaussianClassifier(model_type='TiedCovariance')
#     evaluate_model(tied_classifier, 'TiedCovariance', DTR, LTR, DTE, LTE, output_dir)
#
#     output_dir = 'Output/UnivariateGaussians'
#     mvg_classifier.fit_univariate_gaussian_models(DTR, LTR, output_dir)
#     classes = np.unique(LTR)
#     for cls in classes:
#         for i in range(DTR.shape[0]):
#             feature_data = DTR[i, LTR == cls]
#             mu_ML = feature_data.mean()
#             C_ML = feature_data.var()
#             ll = mvg_classifier.plot_loglikelihood(feature_data, np.array([[mu_ML]]), np.array([[C_ML]]),
#                                                    os.path.join(output_dir,
#                                                                 f'LogLikelihood_Class_{cls}_Feature_{i + 1}.png'))
#             print(f"Class {cls}, Feature {i + 1} Log-Likelihood: {ll}")
#     # Compute covariance and correlation matrices for MVG model
#     mvg_classifier.analyze_covariances(DTR, LTR)
#     # Repeat the analysis using only features 1 to 4 (discarding the last 2 features)
#     DTR_reduced = DTR[:4, :]
#     DTE_reduced = DTE[:4, :]
#     # MVG with reduced features
#     mvg_classifier_reduced = GaussianClassifier(model_type='MVG')
#     evaluate_model(mvg_classifier_reduced, 'MVG_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
#     # Tied Covariance with reduced features
#     tied_classifier_reduced = GaussianClassifier(model_type='TiedCovariance')
#     evaluate_model(tied_classifier_reduced, 'TiedCovariance_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
#     # Naive Bayes with reduced features
#     nb_classifier_reduced = GaussianClassifier(model_type='NaiveBayes')
#     evaluate_model(nb_classifier_reduced, 'NaiveBayes_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
#     # Use PCA to reduce the dimensionality and apply the three classification approaches
#
#     # Reduce to 2 principal components for example
#     pca_dim = 2
#
#     # Apply PCA using the new functions
#     DTR_pca, P_pca = apply_PCA_from_dim(DTR, pca_dim)
#     DTE_pca = apply_pca(DTE, P_pca)
#
#     # MVG with PCA
#     mvg_classifier_pca = GaussianClassifier(model_type='MVG')
#     evaluate_model(mvg_classifier_pca, 'MVG_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)
#
#     # Tied Covariance with PCA
#     tied_classifier_pca = GaussianClassifier(model_type='TiedCovariance')
#     evaluate_model(tied_classifier_pca, 'TiedCovariance_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)
#
#     # Naive Bayes with PCA
#     nb_classifier_pca = GaussianClassifier(model_type='NaiveBayes')
#     evaluate_model(nb_classifier_pca, 'NaiveBayes_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)
#
#
# def train_MVG(DTE, DTR, LTE, LTR):
#     output_dir = "Output/MVG"
#
#     models = {
#         'MVG': GaussianClassifier(model_type='MVG'),
#         'NaiveBayes': GaussianClassifier(model_type='NaiveBayes'),
#         'TiedCovariance': GaussianClassifier(model_type='TiedCovariance')
#     }
#
#     priors = [0.5, 0.9, 0.1]
#     costs = [(1.0, 1.0), (1.0, 9.0), (9.0, 1.0)]
#     pca_dims = [2, 4, 6, None]  # None means no PCA
#
#     for pca_dim in pca_dims:
#         if pca_dim is not None:
#             # Apply PCA
#             DTR_pca, P_pca = apply_PCA_from_dim(DTR, pca_dim)
#             DTE_pca = apply_pca(DTE, P_pca)
#         else:
#             # No PCA
#             DTR_pca, DTE_pca = DTR, DTE
#
#         for model_name, model in models.items():
#             model.train(DTR_pca, LTR)
#
#         for pi1, (Cfn, Cfp) in [(pi1, cost) for pi1 in priors for cost in costs]:
#             print(f"Analyzing for pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}, PCA dim={pca_dim}")
#             for model_name, model in models.items():
#                 logS = model.predict(DTE_pca)
#                 # print(f"{model_name} LLRs: {logS[:5]}")  # Print first 5 LLRs for inspection
#                 predictions = model.compute_predictions(logS)
#                 # print(f"{model_name} Predictions: {predictions[:5]}")  # Print first 5 predictions for inspection
#                 dcf = compute_dcf(predictions, LTE, pi1, Cfn, Cfp)
#                 min_dcf = compute_min_dcf(logS, LTE, pi1, Cfn, Cfp)
#                 print(f"{model_name} Normalized DCF: {dcf}")
#                 print(f"{model_name} Min DCF: {min_dcf}")
#
#         # Plotting functions and other evaluations
#         for model_name, model in models.items():
#             llrs = model.compute_llrs(model.predict(DTE_pca))
#             fpr, tpr, _ = roc_curve(LTE, llrs)
#             roc_auc = auc(fpr, tpr)
#             plot_roc_curve(fpr, tpr, roc_auc, f"{model_name}_PCA_{pca_dim}",
#                            os.path.join(output_dir, f'roc_curve_{model_name}_PCA_{pca_dim}.png'))
#
#             plot_bayes_error(llrs, LTE, np.linspace(-3, 3, 21),
#                              os.path.join(output_dir, f'bayes_error_plot_{model_name}_PCA_{pca_dim}.png'))
#
#
# '''
#
# def plot_loglikelihood(X, mu, C, output_file):
#     ll = loglikelihood(X, mu, C)
#     plt.figure()
#     plt.hist(X.ravel(), bins=50, density=True)
#     XPlot = np.linspace(X.min(), X.max(), 1000)
#     plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
#     plt.title(f'Log-Likelihood: {ll}')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.savefig(output_file)
#     plt.close()
#     return ll
#
#
# def vrow(col):
#     return col.reshape((1, col.size))
#
#
# def vcol(row):
#     return row.reshape((row.size, 1))
#
#
# def compute_ml_estimates(D, L):
#     classes = np.unique(L)
#     means = []
#     covariances = []
#     for cls in classes:
#         class_data = D[:, L == cls]
#         mu_ML = class_data.mean(axis=1, keepdims=True)
#         C_ML = np.cov(class_data)
#         means.append(mu_ML)
#         covariances.append(C_ML)
#     return means, covariances
#
#
# def mvg_classifier(DTR, LTR, DTE):
#     means, covariances = compute_ml_estimates(DTR, LTR)
#     logS = []
#     for mu, C in zip(means, covariances):
#         logS.append(logpdf_GAU_ND(DTE, mu, C))
#     return np.array(logS)
#
#
# def compute_llrs(logS):
#     return logS[1] - logS[0]
#
#
# def tied_covariance_classifier(DTR, LTR, DTE):
#     means, _ = compute_ml_estimates(DTR, LTR)
#     SW = compute_within_class_covariance(DTR, LTR)
#     logS = []
#     for mu in means:
#         logS.append(logpdf_GAU_ND(DTE, mu, SW))
#     return np.array(logS)
#
#
# def naive_bayes_classifier(DTR, LTR, DTE):
#     means, covariances = compute_ml_estimates(DTR, LTR)
#     logS = []
#     for mu, C in zip(means, covariances):
#         C_diag = np.diag(np.diag(C))
#         logS.append(logpdf_GAU_ND(DTE, mu, C_diag))
#     return np.array(logS)
#
#
# def compute_predictions(logS, threshold=0):
#     llrs = compute_llrs(logS)
#     return (llrs >= threshold).astype(int)
#
#
# def compute_error_rate(predictions, true_labels):
#     return np.mean(predictions != true_labels)
#
#
# def evaluate_classifiers(DTR, LTR, DTE, LTE, output_dir='Output/ClassifierResults'):
#     os.makedirs(output_dir, exist_ok=True)
#     logS_mvg = mvg_classifier(DTR, LTR, DTE)
#     predictions_mvg = compute_predictions(logS_mvg)
#     error_mvg = compute_error_rate(predictions_mvg, LTE)
#     plot_classification_results(DTE, LTE, predictions_mvg, os.path.join(output_dir, 'MVG_Results.png'))
#
#     logS_tied = tied_covariance_classifier(DTR, LTR, DTE)
#     predictions_tied = compute_predictions(logS_tied)
#     error_tied = compute_error_rate(predictions_tied, LTE)
#     plot_classification_results(DTE, LTE, predictions_tied, os.path.join(output_dir, 'Tied_Results.png'))
#
#     logS_nb = naive_bayes_classifier(DTR, LTR, DTE)
#     predictions_nb = compute_predictions(logS_nb)
#     error_nb = compute_error_rate(predictions_nb, LTE)
#     plot_classification_results(DTE, LTE, predictions_nb, os.path.join(output_dir, 'NaiveBayes_Results.png'))
#
#     print(f"MVG Error Rate: {error_mvg}")
#     print(f"Tied Covariance Error Rate: {error_tied}")
#     print(f"Naive Bayes Error Rate: {error_nb}")
#
#     return error_mvg, error_tied, error_nb
#
#
# def plot_classification_results(D, L_true, L_pred, output_file):
#     plt.figure()
#     for cls in np.unique(L_true):
#         plt.scatter(D[0, L_true == cls], D[1, L_true == cls], label=f'True Class {cls}', alpha=0.5)
#         plt.scatter(D[0, L_pred == cls], D[1, L_pred == cls], marker='x', label=f'Predicted Class {cls}')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.title('Classification Results')
#     plt.savefig(output_file)
#     plt.close()
#
#
# def analyze_covariances(DTR, LTR, output_dir='Output/Covariances'):
#     os.makedirs(output_dir, exist_ok=True)
#     means, covariances = compute_ml_estimates(DTR, LTR)
#     for cls, (mu, C) in enumerate(zip(means, covariances)):
#         print(f"Class {cls} Mean:\n{mu}")
#         print(f"Class {cls} Covariance Matrix:\n{C}")
#         corr = C / (vcol(np.diag(C) ** 0.5) * vrow(np.diag(C) ** 0.5))
#         print(f"Class {cls} Correlation Matrix:\n{corr}")
#         plot_matrix(C, os.path.join(output_dir, f'Class_{cls}_Covariance.png'))
#         plot_matrix(corr, os.path.join(output_dir, f'Class_{cls}_Correlation.png'))
#
#
# def plot_matrix(matrix, output_file):
#     plt.figure()
#     plt.imshow(matrix, interpolation='nearest', cmap='coolwarm')
#     plt.colorbar()
#     plt.title('Matrix Heatmap')
#     plt.savefig(output_file)
#     plt.close()
#
#
# def compute_confusion_matrix(predictions, labels):
#     K = 2  # Since it's a binary classification problem
#     confusion_matrix = np.zeros((K, K), dtype=int)
#
#     # Flatten predictions if it's a 2D array
#     if predictions.ndim > 1:
#         predictions = predictions.flatten()
#
#     #print(f"Predictions: {predictions}")
#     #print(f"Labels: {labels}")
#
#     for i in range(len(labels)):
#         pred = int(predictions[i])
#         true = int(labels[i])
#         if pred < 0 or pred >= K or true < 0 or true >= K:
#             #print(f"Invalid prediction or label: pred={pred}, true={true}")
#             continue  # Skip invalid values
#         confusion_matrix[pred, true] += 1
#
#     return confusion_matrix
#
#
# def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
#     t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
#     return (llrs >= t).astype(int)
#
#
# def compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
#     FN = confusion_matrix[0, 1]
#     FP = confusion_matrix[1, 0]
#     TN = confusion_matrix[0, 0]
#     TP = confusion_matrix[1, 1]
#     Pfn = FN / (FN + TP)
#     Pfp = FP / (FP + TN)
#     DCFu = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
#     return DCFu
#
#
# def compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp):
#     Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
#     return bayes_risk / Bdummy
#
#
# def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file='roc_curve.png'):
#     out_dir = os.path.dirname(output_file)
#     os.makedirs(out_dir, exist_ok=True)
#
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic - {model_name}')
#     plt.legend(loc="lower right")
#     plt.grid()
#     plt.savefig(output_file)
#     plt.close()
#
#
# def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
#     out_dir = os.path.dirname(output_file)
#     os.makedirs(out_dir, exist_ok=True)
#     dcf = []
#     mindcf = []
#     for p in effPriorLogOdds:
#         pi1 = 1 / (1 + np.exp(-p))
#         decisions = compute_optimal_bayes_decisions(llrs, pi1, 1, 1)
#         confusion_matrix = compute_confusion_matrix(decisions, labels)
#         bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#         normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, 1, 1)
#         dcf.append(normalized_dcf)
#
#         min_bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
#         min_normalized_dcf = compute_normalized_dcf(min_bayes_risk, pi1, 1, 1)
#         mindcf.append(min_normalized_dcf)
#
#     plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
#     plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
#     plt.ylim([0, 1.1])
#     plt.xlim([-3, 3])
#     plt.xlabel('Prior Log-Odds')
#     plt.ylabel('DCF value')
#     plt.legend()
#     plt.grid()
#     plt.title('Bayes Error Plot')
#     plt.savefig(output_file)
#     plt.close()
# '''
