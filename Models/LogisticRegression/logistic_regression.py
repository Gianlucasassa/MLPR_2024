import os

import numpy
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from Models.bayesRisk import compute_empirical_Bayes_risk_binary_llr_optimal_decisions, compute_minDCF_binary_fast
from Preprocess.PCA import apply_pca, compute_pca
import sklearn.datasets

def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]     # We remove setosa from L
    L[L == 2] = 0     # We assign label 0 to virginica (was label 2)
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


# TODO: weighted is NOW working, overall to be reviewed for consistency, quadratic is the worst

class LogRegClass:
    def __init__(self, DTR, LTR, l, weighted=False, pT=0.5):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.weighted = weighted
        self.pT = pT
        self.w = None
        self.b = None

    def logreg_obj(self, v):
        w, b = v[:-1], v[-1]
        ZTR = self.LTR * 2.0 - 1.0
        S = (np.dot(vcol(w).T, self.DTR) + b).ravel()
        loss = np.logaddexp(0, -ZTR * S).mean() + self.l / 2 * np.linalg.norm(w) ** 2
        G = -ZTR / (1.0 + np.exp(ZTR * S))
        GW = (vrow(G) * self.DTR).mean(1) + self.l * w
        Gb = G.mean()
        grad = np.hstack([GW, Gb])
        return loss, grad

    def weighted_logreg_obj(self, v):
        w = v[:-1]
        b = v[-1]
        ZTR = self.LTR * 2.0 - 1.0  # Convert labels to +1/-1
        s = numpy.dot(vcol(w).T, self.DTR).ravel() + b

        # Calculate the weights based on prior and class distribution
        wTrue = self.pT / (ZTR > 0).sum()  # Compute the weights for the two classes
        wFalse = (1 - self.pT) / (ZTR < 0).sum()

        # Compute the weighted loss
        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue  # Apply the weights to the loss computations
        loss[ZTR < 0] *= wFalse
        loss = loss.sum() + self.l / 2 * numpy.linalg.norm(w) ** 2

        # Compute the gradient
        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue  # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        GW = (vrow(G) * self.DTR).sum(1) + self.l * w.ravel()
        Gb = G.sum()

        grad = numpy.hstack([GW, numpy.array(Gb)])
        return loss, grad

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        if self.weighted:
            vf, _, _ = scipy.optimize.fmin_l_bfgs_b(self.weighted_logreg_obj, x0, approx_grad=False)
        else:
            vf, _, _ = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad=False)
        self.w = vf[:-1]
        self.b = vf[-1]

    def predict(self, DTE):
        return np.dot(self.w, DTE) + self.b

    def predict_log_proba(self, DTE):
        scores = self.predict(DTE)
        return scores - np.log(1 / 0.5 - 1)


def expand_features_quadratic(D):
    n_features, n_samples = D.shape
    n_expanded_features = n_features + n_features * (n_features + 1) // 2
    D_expanded = np.zeros((n_expanded_features, n_samples))

    idx = n_features
    D_expanded[:n_features, :] = D
    for i in range(n_features):
        for j in range(i, n_features):
            D_expanded[idx, :] = D[i, :] * D[j, :]
            idx += 1

    return D_expanded


def compute_scores(model, DTE, LTR, pT):
    # Compute scores
    sVal = model.predict(DTE)

    # Compute LLR-like scores for actual DCF
    pEmp = (LTR == 1).sum() / LTR.size
    sValLLR_actual = sVal - np.log(pEmp / (1 - pEmp))

    # Adjust scores for weighted logistic regression
    if model.weighted:
        sValLLR_actual -= np.log(pT / (1 - pT))

    return sVal, sValLLR_actual


def plot_dcf_vs_lambda(lambdas, actual_DCFs, min_DCFs, title, filename):
    plt.figure()
    plt.plot(lambdas, actual_DCFs, label='Actual DCF')
    plt.plot(lambdas, min_DCFs, label='Min DCF')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def train_LR(DTE, DTR, LTE, LTR, lambdas, effective_prior, weighted=False, pT=0.5):
    actual_DCFs = []
    min_DCFs = []

    for l in lambdas:
        model = LogRegClass(DTR, LTR, l, weighted=weighted, pT=pT)
        model.train()

        # Compute scores
        sVal, sValLLR_actual = compute_scores(model, DTE, LTR, pT)

        # Compute actual DCF
        actual_DCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR_actual, LTE, effective_prior,
                                                                               1.0, 1.0)

        # Compute minDCF
        sValLLR_min = sVal - np.log(effective_prior / (1 - effective_prior))
        min_DCF = compute_minDCF_binary_fast(sValLLR_min, LTE, effective_prior, 1.0, 1.0)

        actual_DCFs.append(actual_DCF)
        min_DCFs.append(min_DCF)

    return actual_DCFs, min_DCFs


def center_data(DTR, DTE):
    mean = DTR.mean(axis=1, keepdims=True)
    return DTR - mean, DTE - mean


def z_normalize(DTR, DTE):
    mean = DTR.mean(axis=1, keepdims=True)
    std = DTR.std(axis=1, keepdims=True)
    return (DTR - mean) / std, (DTE - mean) / std


def whiten_data(DTR, DTE):
    mean = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mean
    DTE_centered = DTE - mean
    cov = np.cov(DTR_centered)
    U, S, _ = np.linalg.svd(cov)
    W = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-10)))
    return np.dot(W.T, DTR_centered), np.dot(W.T, DTE_centered)


def print_dcf_results(model_name, actual_DCFs, min_DCFs, lambdas):
    print(f"Evaluating {model_name}")
    for i, l in enumerate(lambdas):
        print(f"{model_name} - Lambda: {l:.4e}")
        print(f"DCF (non-normalized): {actual_DCFs[i]:.3f}")
        print(f"MinDCF (normalized, fast): {min_DCFs[i]:.3f}")
    print()


def evaluate_IRIS_LR(lambdas=[1e-3, 1e-1, 1.0]):
    # Load and split the dataset

    print("NOW EVALUATING IRIS DATASET")
    print()
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Prior for the class 1 (versicolor) in the training set
    pT = LTR.mean()

    for l in lambdas:
        print(f"Evaluating Logistic Regression with lambda={l:.1e}")

        # Initialize and train the logistic regression model
        lr_model = LogRegClass(DTR, LTR, l, weighted=False, pT=pT)
        lr_model.train()

        # Predict the validation set
        scores = lr_model.predict(DVAL)

        # Evaluate the error rate
        predictions = (scores > 0).astype(int)
        error_rate = (predictions != LVAL).mean() * 100

        # Compute minDCF and actDCF
        emp_prior = LTR.mean()
        sllr = scores - np.log(emp_prior / (1 - emp_prior))
        minDCF = compute_minDCF_binary_fast(sllr, LVAL, 0.5, 1.0, 1.0)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sllr, LVAL, 0.5, 1.0, 1.0)

        # Compute the objective function value J(w*, b*)
        obj_value, _ = lr_model.logreg_obj(np.hstack([lr_model.w, lr_model.b]))

        # Print results
        print(f"J(w*, b*): {obj_value:.6e}")
        print(f"Error rate: {error_rate:.1f}%")
        print(f"minDCF: {minDCF:.4f} / actDCF: {actDCF:.4f}\n")

    print("FINISHED EVALUATING IRIS DATASET")
    print()

def evaluate_IRIS_LR_weighted(lambdas=[1e-3, 1e-1, 1.0], pi_T=0.8):
    # Load and split the dataset

    print("NOW EVALUATING IRIS DATASET WITH WEIGHTED LOGISTIC REGRESSION")
    print()
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Number of samples of class 1 and 0 in the training set
    n_T = (LTR == 1).sum()
    n_F = (LTR == 0).sum()

    for l in lambdas:
        print(f"Evaluating Weighted Logistic Regression with lambda={l:.1e}, pi_T={pi_T:.1f}")

        # Initialize and train the weighted logistic regression model
        lr_model = LogRegClass(DTR, LTR, l, weighted=True, pT=pi_T)
        lr_model.train()

        # Predict the validation set
        scores = lr_model.predict(DVAL)

        # Adjust the scores to behave like LLRs by removing prior log-odds
        sllr = scores - np.log(pi_T / (1 - pi_T))

        # Evaluate the error rate
        predictions = (scores > 0).astype(int)
        error_rate = (predictions != LVAL).mean() * 100

        # Compute minDCF and actDCF
        minDCF = compute_minDCF_binary_fast(sllr, LVAL, pi_T, 1.0, 1.0)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sllr, LVAL, pi_T, 1.0, 1.0)

        # Compute the objective function value J(w*, b*)
        obj_value, _ = lr_model.weighted_logreg_obj(np.hstack([lr_model.w, lr_model.b]))

        # Print results
        print(f"J(w*, b*): {obj_value:.6e}")
        print(f"Error rate: {error_rate:.1f}%")
        print(f"minDCF: {minDCF:.4f} / actDCF: {actDCF:.4f}\n")

    print("FINISHED EVALUATING IRIS DATASET WITH WEIGHTED LOGISTIC REGRESSION")
    print()


def Logistic_Regression_train(DTE, DTR, LTE, LTR): #TODO: working but results are not so good. double check the weighted and the Reduced dataset; moreover the minDCF for all of them. (Check 10/10 report)
    evaluate_IRIS_LR()
    evaluate_IRIS_LR_weighted()

    output_dir_base = 'Output/LogisticRegression/'
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    # Define lambda values and effective prior
    lambdas = np.logspace(-4, 2, 13)
    effective_prior = 0.1

    # Standard logistic regression
    actual_DCFs, min_DCFs = train_LR(DTE, DTR, LTE, LTR, lambdas, effective_prior)
    plot_dcf_vs_lambda(lambdas, actual_DCFs, min_DCFs, 'DCF vs Lambda',
                       os.path.join(output_dir_base, "dcf_vs_lambda.png"))
    print_dcf_results("Standard Logistic Regression", actual_DCFs, min_DCFs, lambdas)

    # Reduced training samples
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]
    actual_DCFs_reduced, min_DCFs_reduced = train_LR(DTE, DTR_reduced, LTE, LTR_reduced, lambdas, effective_prior)
    plot_dcf_vs_lambda(lambdas, actual_DCFs_reduced, min_DCFs_reduced, 'DCF vs Lambda (Reduced Training Samples)',
                       os.path.join(output_dir_base, "dcf_vs_lambda_reduced.png"))
    print_dcf_results("Reduced Training Samples Logistic Regression", actual_DCFs_reduced, min_DCFs_reduced, lambdas)

    # Weighted logistic regression
    actual_DCFs_weighted, min_DCFs_weighted = train_LR(DTE, DTR, LTE, LTR, lambdas, effective_prior, weighted=True,
                                                       pT=effective_prior)
    plot_dcf_vs_lambda(lambdas, actual_DCFs_weighted, min_DCFs_weighted, 'DCF vs Lambda (Weighted)',
                       os.path.join(output_dir_base, "dcf_vs_lambda_weighted.png"))
    print_dcf_results("Weighted Logistic Regression", actual_DCFs_weighted, min_DCFs_weighted, lambdas)

    # Quadratic logistic regression
    DTR_expanded = expand_features_quadratic(DTR)  # Expand features to quadratic
    DTE_expanded = expand_features_quadratic(DTE)  # Expand features to quadratic

    # Train and evaluate models with quadratic features
    actual_DCFs_quadratic, min_DCFs_quadratic = train_LR(DTE_expanded, DTR_expanded, LTE, LTR, lambdas, effective_prior)
    plot_dcf_vs_lambda(lambdas, actual_DCFs_quadratic, min_DCFs_quadratic, 'DCF vs Lambda (Quadratic)',
                       os.path.join(output_dir_base, "dcf_vs_lambda_quadratic.png"))
    print_dcf_results("Quadratic Logistic Regression", actual_DCFs_quadratic, min_DCFs_quadratic, lambdas)

    # Preprocessing analysis
    preprocessing_methods = {
        'centered': center_data,
        'z_normalized': z_normalize,
        'whitened': whiten_data,
        'pca_m4': lambda DTR, DTE: (apply_pca(compute_pca(DTR, 4), DTR), apply_pca(compute_pca(DTR, 4), DTE)),
        'pca_m6': lambda DTR, DTE: (apply_pca(compute_pca(DTR, 6), DTR), apply_pca(compute_pca(DTR, 6), DTE))
    }

    for method_name, method in preprocessing_methods.items():
        DTR_proc, DTE_proc = method(DTR, DTE)
        actual_DCFs, min_DCFs = train_LR(DTE_proc, DTR_proc, LTE, LTR, lambdas, effective_prior)
        plot_dcf_vs_lambda(lambdas, actual_DCFs, min_DCFs, f'DCF vs Lambda ({method_name})',
                           os.path.join(output_dir_base, f"dcf_vs_lambda_{method_name}.png"))
        print_dcf_results(f"{method_name.capitalize()} Logistic Regression", actual_DCFs, min_DCFs, lambdas)

    return

    """LAST POINT: As you should have observed, the best models in terms of minimum DCF are not necessarily those that
    provide the best actual DCFs, i.e., they may present significant mis-calibration. We will deal with score
    calibration at the end of the course. For the moment, we focus on selecting the models that optimize the
    minimum DCF on our validation set. Compare all models that you have trained up to now, including
    Gaussian models, in terms of minDCF for the target application  T = 0:1. Which model(s) achieve(s)
    the best results? What kind of separation rules or distribution assumptions characterize this / these
    model(s)? How are the results related to the characteristics of the dataset features? 
    
    For the Gaussian it was:
    Evaluating for Effective Prior: 0.1
    MVG - Effective Prior: 0.1
    DCF (non-normalized): 0.031
    DCF (normalized): 0.305
    MinDCF (normalized, fast): 0.263
    Tied MVG - Effective Prior: 0.1
    DCF (non-normalized): 0.041
    DCF (normalized): 0.406
    MinDCF (normalized, fast): 0.363
    Naive MVG - Effective Prior: 0.1
    DCF (non-normalized): 0.030
    DCF (normalized): 0.302
    MinDCF (normalized, fast): 0.257
    MVG_PCA_m2 - Effective Prior: 0.1
    DCF (non-normalized): 0.039
    DCF (normalized): 0.388
    MinDCF (normalized, fast): 0.353
    Tied_MVG_PCA_m2 - Effective Prior: 0.1
    DCF (non-normalized): 0.040
    DCF (normalized): 0.396
    MinDCF (normalized, fast): 0.363
    Naive_MVG_PCA_m2 - Effective Prior: 0.1
    DCF (non-normalized): 0.039
    DCF (normalized): 0.387
    MinDCF (normalized, fast): 0.356
    MVG_PCA_m4 - Effective Prior: 0.1
    DCF (non-normalized): 0.035
    DCF (normalized): 0.353
    MinDCF (normalized, fast): 0.301
    Tied_MVG_PCA_m4 - Effective Prior: 0.1
    DCF (non-normalized): 0.040
    DCF (normalized): 0.403
    MinDCF (normalized, fast): 0.361
    Naive_MVG_PCA_m4 - Effective Prior: 0.1
    DCF (non-normalized): 0.040
    DCF (normalized): 0.397
    MinDCF (normalized, fast): 0.361
    MVG_PCA_m6 - Effective Prior: 0.1
    DCF (non-normalized): 0.031
    DCF (normalized): 0.305
    MinDCF (normalized, fast): 0.263
    Tied_MVG_PCA_m6 - Effective Prior: 0.1
    DCF (non-normalized): 0.041
    DCF (normalized): 0.406
    MinDCF (normalized, fast): 0.363
    Naive_MVG_PCA_m6 - Effective Prior: 0.1
    DCF (non-normalized): 0.039
    DCF (normalized): 0.392
    MinDCF (normalized, fast): 0.353
    
    Now it's:
    Evaluating Standard Logistic Regression
    Standard Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.402
    DCF (normalized): 0.402
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.405
    DCF (normalized): 0.405
    MinDCF (normalized, fast): 0.365
    Standard Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.413
    DCF (normalized): 0.413
    MinDCF (normalized, fast): 0.365
    Standard Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.430
    DCF (normalized): 0.430
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.457
    DCF (normalized): 0.457
    MinDCF (normalized, fast): 0.361
    Standard Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.581
    DCF (normalized): 0.581
    MinDCF (normalized, fast): 0.362
    Standard Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.852
    DCF (normalized): 0.852
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.995
    DCF (normalized): 0.995
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    Standard Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    Standard Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    Standard Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    Evaluating Reduced Training Samples Logistic Regression
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.978
    DCF (normalized): 0.978
    MinDCF (normalized, fast): 0.447
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.898
    DCF (normalized): 0.898
    MinDCF (normalized, fast): 0.445
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.747
    DCF (normalized): 0.747
    MinDCF (normalized, fast): 0.449
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.573
    DCF (normalized): 0.573
    MinDCF (normalized, fast): 0.450
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.465
    DCF (normalized): 0.465
    MinDCF (normalized, fast): 0.441
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.483
    DCF (normalized): 0.483
    MinDCF (normalized, fast): 0.415
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.716
    DCF (normalized): 0.716
    MinDCF (normalized, fast): 0.399
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.979
    DCF (normalized): 0.979
    MinDCF (normalized, fast): 0.389
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.387
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.380
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.378
    Reduced Training Samples Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.379
    Reduced Training Samples Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.378
    Evaluating Weighted Logistic Regression
    Weighted Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.357
    Weighted Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.365
    Weighted Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.394
    Weighted Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.411
    Weighted Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.368
    Weighted Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.457
    Weighted Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.385
    Weighted Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.397
    Weighted Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.361
    Weighted Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.391
    Weighted Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.367
    Weighted Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.367
    Weighted Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.367
    Evaluating Quadratic Logistic Regression
    Quadratic Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.277
    DCF (normalized): 0.277
    MinDCF (normalized, fast): 0.260
    Quadratic Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.266
    DCF (normalized): 0.266
    MinDCF (normalized, fast): 0.261
    Quadratic Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.277
    DCF (normalized): 0.277
    MinDCF (normalized, fast): 0.259
    Quadratic Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.287
    DCF (normalized): 0.287
    MinDCF (normalized, fast): 0.251
    Quadratic Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.346
    DCF (normalized): 0.346
    MinDCF (normalized, fast): 0.248
    Quadratic Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.495
    DCF (normalized): 0.495
    MinDCF (normalized, fast): 0.244
    Quadratic Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.753
    DCF (normalized): 0.753
    MinDCF (normalized, fast): 0.248
    Quadratic Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.963
    DCF (normalized): 0.963
    MinDCF (normalized, fast): 0.264
    Quadratic Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.287
    Quadratic Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.309
    Quadratic Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.328
    Quadratic Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.331
    Quadratic Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.330
    Evaluating centered Logistic Regression
    centered Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.402
    DCF (normalized): 0.402
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.405
    DCF (normalized): 0.405
    MinDCF (normalized, fast): 0.365
    centered Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.413
    DCF (normalized): 0.413
    MinDCF (normalized, fast): 0.365
    centered Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.430
    DCF (normalized): 0.430
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.457
    DCF (normalized): 0.457
    MinDCF (normalized, fast): 0.361
    centered Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.581
    DCF (normalized): 0.581
    MinDCF (normalized, fast): 0.362
    centered Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.852
    DCF (normalized): 0.852
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.995
    DCF (normalized): 0.995
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    centered Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    centered Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    centered Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    Evaluating z_normalized Logistic Regression
    z_normalized Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.402
    DCF (normalized): 0.402
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.405
    DCF (normalized): 0.405
    MinDCF (normalized, fast): 0.365
    z_normalized Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.413
    DCF (normalized): 0.413
    MinDCF (normalized, fast): 0.365
    z_normalized Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.430
    DCF (normalized): 0.430
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.457
    DCF (normalized): 0.457
    MinDCF (normalized, fast): 0.361
    z_normalized Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.581
    DCF (normalized): 0.581
    MinDCF (normalized, fast): 0.362
    z_normalized Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.852
    DCF (normalized): 0.852
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.995
    DCF (normalized): 0.995
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    z_normalized Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    z_normalized Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    z_normalized Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    Evaluating whitened Logistic Regression
    whitened Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.403
    DCF (normalized): 0.403
    MinDCF (normalized, fast): 0.364
    whitened Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.407
    DCF (normalized): 0.407
    MinDCF (normalized, fast): 0.365
    whitened Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.419
    DCF (normalized): 0.419
    MinDCF (normalized, fast): 0.365
    whitened Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.416
    DCF (normalized): 0.416
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.477
    DCF (normalized): 0.477
    MinDCF (normalized, fast): 0.362
    whitened Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.652
    DCF (normalized): 0.652
    MinDCF (normalized, fast): 0.361
    whitened Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.919
    DCF (normalized): 0.919
    MinDCF (normalized, fast): 0.364
    whitened Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    whitened Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    Evaluating pca_m4 Logistic Regression
    pca_m4 Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.410
    DCF (normalized): 0.410
    MinDCF (normalized, fast): 0.362
    pca_m4 Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.405
    DCF (normalized): 0.405
    MinDCF (normalized, fast): 0.362
    pca_m4 Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.413
    DCF (normalized): 0.413
    MinDCF (normalized, fast): 0.362
    pca_m4 Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.416
    DCF (normalized): 0.416
    MinDCF (normalized, fast): 0.362
    pca_m4 Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.448
    DCF (normalized): 0.448
    MinDCF (normalized, fast): 0.361
    pca_m4 Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.578
    DCF (normalized): 0.578
    MinDCF (normalized, fast): 0.361
    pca_m4 Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.853
    DCF (normalized): 0.853
    MinDCF (normalized, fast): 0.358
    pca_m4 Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.995
    DCF (normalized): 0.995
    MinDCF (normalized, fast): 0.358
    pca_m4 Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    pca_m4 Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    pca_m4 Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.361
    pca_m4 Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.361
    pca_m4 Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.361
    Evaluating pca_m6 Logistic Regression
    pca_m6 Logistic Regression - Lambda: 1.0000e-04
    DCF (non-normalized): 0.402
    DCF (normalized): 0.402
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 3.1623e-04
    DCF (non-normalized): 0.405
    DCF (normalized): 0.405
    MinDCF (normalized, fast): 0.365
    pca_m6 Logistic Regression - Lambda: 1.0000e-03
    DCF (non-normalized): 0.413
    DCF (normalized): 0.413
    MinDCF (normalized, fast): 0.365
    pca_m6 Logistic Regression - Lambda: 3.1623e-03
    DCF (non-normalized): 0.430
    DCF (normalized): 0.430
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 1.0000e-02
    DCF (non-normalized): 0.457
    DCF (normalized): 0.457
    MinDCF (normalized, fast): 0.361
    pca_m6 Logistic Regression - Lambda: 3.1623e-02
    DCF (non-normalized): 0.581
    DCF (normalized): 0.581
    MinDCF (normalized, fast): 0.362
    pca_m6 Logistic Regression - Lambda: 1.0000e-01
    DCF (non-normalized): 0.852
    DCF (normalized): 0.852
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 3.1623e-01
    DCF (non-normalized): 0.995
    DCF (normalized): 0.995
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 1.0000e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 3.1623e+00
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.364
    pca_m6 Logistic Regression - Lambda: 1.0000e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.363
    pca_m6 Logistic Regression - Lambda: 3.1623e+01
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    pca_m6 Logistic Regression - Lambda: 1.0000e+02
    DCF (non-normalized): 1.000
    DCF (normalized): 1.000
    MinDCF (normalized, fast): 0.362
    """

# import os
# from itertools import groupby
#
# import numpy as np
# import scipy
# from matplotlib import pyplot as plt
# from scipy.optimize import fmin_l_bfgs_b
# from sklearn.metrics import roc_curve, auc
#
# from main import compute_dcf, compute_min_dcf
#
#
#
# def expand_features(D):
#     expanded_features = [D]
#     for i in range(D.shape[0]):
#         for j in range(i, D.shape[0]):
#             expanded_features.append(D[i] * D[j])
#     return np.vstack(expanded_features)
#
# def preprocess_data(DTR, DTE, method='none'):
#     if method == 'center':
#         mean = np.mean(DTR, axis=1, keepdims=True)
#         DTR = DTR - mean
#         DTE = DTE - mean
#     elif method == 'z_norm':
#         mean = np.mean(DTR, axis=1, keepdims=True)
#         std = np.std(DTR, axis=1, keepdims=True)
#         DTR = (DTR - mean) / std
#         DTE = (DTE - mean) / std
#     elif method == 'whiten':
#         mean = np.mean(DTR, axis=1, keepdims=True)
#         DTR = DTR - mean
#         DTE = DTE - mean
#         cov = np.cov(DTR)
#         U, S, _ = np.linalg.svd(cov)
#         DTR = np.dot(U.T, DTR) / np.sqrt(S[:, np.newaxis])
#         DTE = np.dot(U.T, DTE) / np.sqrt(S[:, np.newaxis])
#     elif method == 'pca':
#         mean = np.mean(DTR, axis=1, keepdims=True)
#         DTR = DTR - mean
#         DTE = DTE - mean
#         cov = np.cov(DTR)
#         U, S, _ = np.linalg.svd(cov)
#         DTR = np.dot(U.T, DTR)
#         DTE = np.dot(U.T, DTE)
#     return DTR, DTE
#
# def analyze_dcf_vs_lambda(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1, output_file_prefix='dcf_vs_lambda'):
#     actual_dcfs = []
#     min_dcfs = []
#     for l in lambdas:
#         logreg_classifier = LogRegClass(DTR, LTR, l, prior_weighted=False)
#         logreg_classifier.train()
#         scores = logreg_classifier.predict(DTE)
#         actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
#         min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
#         actual_dcfs.append(actual_dcf)
#         min_dcfs.append(min_dcf)
#
#     plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
#     plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
#     plt.xscale('log')
#     plt.xlabel('Lambda')
#     plt.ylabel('DCF')
#     plt.title('DCF vs Lambda')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'{output_file_prefix}.png')
#     plt.close()
#
# def analyze_with_fewer_samples(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
#     DTR_reduced = DTR[:, ::50]
#     LTR_reduced = LTR[::50]
#     analyze_dcf_vs_lambda(DTR_reduced, LTR_reduced, DTE, LTE, lambdas, pi_t, output_file_prefix='dcf_vs_lambda_reduced')
#
# def analyze_prior_weighted(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
#     actual_dcfs = []
#     min_dcfs = []
#     for l in lambdas:
#         logreg_classifier = LogRegClass(DTR, LTR, l, prior_weighted=True, pi_t=pi_t)
#         logreg_classifier.train()
#         scores = logreg_classifier.predict(DTE)
#         actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
#         min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
#         actual_dcfs.append(actual_dcf)
#         min_dcfs.append(min_dcf)
#
#     plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
#     plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
#     plt.xscale('log')
#     plt.xlabel('Lambda')
#     plt.ylabel('DCF')
#     plt.title('Prior-Weighted DCF vs Lambda')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('prior_weighted_dcf_vs_lambda.png')
#     plt.close()
#
# def analyze_quadratic(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
#     DTR_exp = expand_features(DTR)
#     DTE_exp = expand_features(DTE)
#     actual_dcfs = []
#     min_dcfs = []
#     for l in lambdas:
#         logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=False)
#         logreg_classifier.train()
#         scores = logreg_classifier.predict(DTE_exp)
#         actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
#         min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
#         actual_dcfs.append(actual_dcf)
#         min_dcfs.append(min_dcf)
#
#     plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
#     plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
#     plt.xscale('log')
#     plt.xlabel('Lambda')
#     plt.ylabel('DCF')
#     plt.title('Quadratic DCF vs Lambda')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('quadratic_dcf_vs_lambda.png')
#     plt.close()
#
# def analyze_preprocessing(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1, output_file_prefix='dcf_vs_lambda'):
#     methods = ['none', 'center', 'z_norm', 'whiten', 'pca']
#     for method in methods:
#         DTR_prep, DTE_prep = preprocess_data(DTR, DTE, method)
#         actual_dcfs = []
#         min_dcfs = []
#         for l in lambdas:
#             logreg_classifier = LogRegClass(DTR_prep, LTR, l, prior_weighted=False)
#             logreg_classifier.train()
#             scores = logreg_classifier.predict(DTE_prep)
#             actual_dcf = logreg_classifier.compute_dcf(scores, LTE, pi_t)
#             min_dcf = logreg_classifier.compute_min_dcf(scores, LTE, pi_t)
#             actual_dcfs.append(actual_dcf)
#             min_dcfs.append(min_dcf)
#
#         plt.plot(lambdas, actual_dcfs, marker='o', label='Actual DCF')
#         plt.plot(lambdas, min_dcfs, marker='o', label='Min DCF')
#         plt.xscale('log')
#         plt.xlabel('Lambda')
#         plt.ylabel('DCF')
#         plt.title(f'DCF vs Lambda ({method})')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f'{output_file_prefix}_{method}.png')
#         plt.close()
#
# def compare_all_models(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1):
#     models = {
#         'LogisticRegression': LogRegClass(DTR, LTR, l=1.0),
#         'PriorWeightedLogReg': LogRegClass(DTR, LTR, l=1.0, prior_weighted=True, pi_t=pi_t),
#         'QuadraticLogReg': LogRegClass(expand_features(DTR), LTR, l=1.0)
#     }
#     results = {}
#
#     for model_name, model in models.items():
#         actual_dcfs = []
#         min_dcfs = []
#         for l in lambdas:
#             model.l = l
#             if model_name == 'QuadraticLogReg':
#                 model.DTR = expand_features(DTR)
#                 DTE_exp = expand_features(DTE)
#                 model.train()
#                 scores = model.predict(DTE_exp)
#             else:
#                 model.train()
#                 scores = model.predict(DTE)
#             actual_dcf = model.compute_dcf(scores, LTE, pi_t)
#             min_dcf = model.compute_min_dcf(scores, LTE, pi_t)
#             actual_dcfs.append(actual_dcf)
#             min_dcfs.append(min_dcf)
#
#         results[model_name] = (actual_dcfs, min_dcfs)
#
#         plt.plot(lambdas, actual_dcfs, marker='o', label=f'{model_name} Actual DCF')
#         plt.plot(lambdas, min_dcfs, marker='o', label=f'{model_name} Min DCF')
#
#     plt.xscale('log')
#     plt.xlabel('Lambda')
#     plt.ylabel('DCF')
#     plt.title('DCF vs Lambda for Different Models')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('dcf_vs_lambda_all_models.png')
#     plt.close()
#
#     return results
#
# def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file):
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic - {model_name}')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.savefig(output_file)
#     plt.close()
#
#
# def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
#     dcf = []
#     mindcf = []
#     for p in effPriorLogOdds:
#         pi1 = 1 / (1 + np.exp(-p))
#         thresholds = np.sort(llrs)
#         fnr, fpr = [], []
#
#         for threshold in thresholds:
#             predictions = (llrs >= threshold).astype(int)
#             fnr.append(np.mean(predictions[labels == 1] == 0))
#             fpr.append(np.mean(predictions[labels == 0] == 1))
#
#         dcf_value = pi1 * 1 * np.mean(predictions[labels == 1] == 0) + (1 - pi1) * 1 * np.mean(
#             predictions[labels == 0] == 1)
#         min_dcf_value = min([pi1 * 1 * fnr_i + (1 - pi1) * 1 * fpr_i for fnr_i, fpr_i in zip(fnr, fpr)])
#
#         dcf.append(dcf_value)
#         mindcf.append(min_dcf_value)
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
# class LogRegClass:
#     def __init__(self, DTR, LTR, l, prior_weighted=False, pi_t=0.1):
#         self.DTR = DTR
#         self.LTR = LTR
#         self.l = l
#         self.prior_weighted = prior_weighted
#         self.pi_t = pi_t
#         self.w = None
#         self.b = None
#
#     def logreg_obj(self, v):
#         w, b = v[:-1], v[-1]
#         Z = 2 * self.LTR - 1
#         S = np.dot(w.T, self.DTR) + b
#         reg_term = 0.5 * self.l * np.dot(w, w)
#
#         if self.prior_weighted and self.pi_t is not None:
#             nT = np.sum(self.LTR)
#             nF = len(self.LTR) - nT
#             xi = np.where(self.LTR == 1, self.pi_t / nT, (1 - self.pi_t) / nF)
#             loss = np.mean(xi * np.logaddexp(0, -Z * S))
#         else:
#             loss = np.logaddexp(0, -Z * S).mean()
#
#         return reg_term + loss
#
#     def logreg_obj_grad(self, v):
#         w, b = v[:-1], v[-1]
#         Z = 2 * self.LTR - 1
#         S = np.dot(w.T, self.DTR) + b
#         G = -Z / (1 + np.exp(Z * S))
#
#         if self.prior_weighted and self.pi_t is not None:
#             nT = np.sum(self.LTR)
#             nF = len(self.LTR) - nT
#             xi = np.where(self.LTR == 1, self.pi_t / nT, (1 - self.pi_t) / nF)
#             G = G * xi
#
#         grad_w = self.l * w + (G @ self.DTR.T) / self.DTR.shape[1]
#         grad_b = G.mean()
#         return np.append(grad_w, grad_b)
#
#     def train(self):
#         x0 = np.zeros(self.DTR.shape[0] + 1)
#         opt = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj, x0=x0, fprime=self.logreg_obj_grad, approx_grad=False)
#         self.w, self.b = opt[0][:-1], opt[0][-1]
#
#     def predict(self, DTE):
#         scores = np.dot(self.w.T, DTE) + self.b
#         return scores
#
#     def compute_predictions(self, scores):
#         return (scores > 0).astype(int)
#
#     def compute_error_rate(self, predictions, true_labels):
#         return np.mean(predictions != true_labels)
#
#     @staticmethod
#     def plot_error_rates(results, output_file='error_rates.png'):
#         from itertools import groupby
#
#         for config_name, group in groupby(results, key=lambda x: x[0]):
#             lambdas, error_rates = zip(*[(l, e) for _, l, e in group])
#             plt.plot(lambdas, error_rates, marker='o', label=config_name)
#
#         plt.xscale('log')
#         plt.xlabel('Lambda')
#         plt.ylabel('Error Rate')
#         plt.title('Error Rate vs Lambda for Different Configurations')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(output_file)
#         plt.close()
#
#     def compute_dcf_at_threshold(self, predictions, labels, pi_t, Cfn=1, Cfp=1):
#         fnr = np.mean(predictions[labels == 1] == 0)
#         fpr = np.mean(predictions[labels == 0] == 1)
#         dcf = pi_t * Cfn * fnr + (1 - pi_t) * Cfp * fpr
#         return dcf
#
#     def compute_dcf(self, scores, labels, pi_t, Cfn=1, Cfp=1):
#         thresholds = np.sort(scores)
#         min_dcf = float('inf')
#         for t in thresholds:
#             predictions = (scores >= t).astype(int)
#             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t, Cfn, Cfp)
#             if dcf < min_dcf:
#                 min_dcf = dcf
#         return min_dcf
#
#     def compute_min_dcf(self, scores, labels, pi_t, Cfn=1, Cfp=1):
#         thresholds = np.sort(scores)
#         min_dcf = float('inf')
#         for t in thresholds:
#             predictions = (scores >= t).astype(int)
#             dcf = self.compute_dcf_at_threshold(predictions, labels, pi_t, Cfn, Cfp)
#             if dcf < min_dcf:
#                 min_dcf = dcf
#         return min_dcf
#
#         # functions calibration
#
#     def fit(self, X, y):
#         # Ensure the shapes are as expected
#         assert X.shape[0] == y.shape[0], "Mismatch in the number of samples between X and y"
#         print(f"Shapes before transpose: X.shape={X.shape}, y.shape={y.shape}")
#
#         self.DTR = X.T
#         self.LTR = y
#         print(f"Shapes after transpose: self.DTR.shape={self.DTR.shape}, self.LTR.shape={self.LTR.shape}")
#
#         # Ensure dimensions match
#         if self.DTR.shape[1] != self.LTR.shape[0]:
#             raise ValueError(
#                 f"Mismatch in number of samples: DTR.shape[1]={self.DTR.shape[1]} but LTR.shape[0]={self.LTR.shape[0]}")
#
#         x0 = np.zeros(self.DTR.shape[0] + 1)
#         opt = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj, x0=x0, fprime=self.logreg_obj_grad, approx_grad=False)
#         self.w, self.b = opt[0][:-1], opt[0][-1]
#         print(f"Optimization result: w={self.w}, b={self.b}")
#
#     def predict_proba(self, X):
#         scores = np.dot(self.w.T, X.T) + self.b
#         probs = 1 / (1 + np.exp(-scores))
#         return np.vstack([1 - probs, probs]).T
#
#
#
# def logreg_obj(v, DTR, LTR, l):
#     w, b = v[0:-1], v[-1]
#     z = 2 * LTR - 1  # Convert labels to +/- 1
#     S = np.dot(w.T, DTR) + b
#     regularizer = (l / 2) * np.linalg.norm(w) ** 2
#     loss = np.logaddexp(0, -z * S).mean()
#     J = regularizer + loss
#     gradient_w = l * w + np.dot(DTR, (-z / (1 + np.exp(z * S)))) / DTR.shape[1]
#     gradient_b = (-z / (1 + np.exp(z * S))).mean()
#     gradient = np.append(gradient_w, gradient_b)
#     return J, gradient
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def train_logreg(DTR, LTR, l):
#     x0 = np.zeros(DTR.shape[0] + 1)
#     return fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l), approx_grad=False)[0]
#
#
# def compute_logreg_scores(D, model):
#     w, b = model[:-1], model[-1]
#     return np.dot(w.T, D) + b
#
#
# def compute_logreg_predictions(scores):
#     return (scores >= 0).astype(int)
#
#
# def plot_metrics(lambdas, actual_dcf, min_dcf, title):
#     plt.figure()
#     plt.xscale('log')
#     plt.plot(lambdas, actual_dcf, label='Actual DCF')
#     plt.plot(lambdas, min_dcf, label='Min DCF')
#     plt.xlabel('Lambda')
#     plt.ylabel('DCF')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join("Output/LogFigures", title + ".png"))
#
#
# def train_LR(DTE, DTR, LTE, LTR):
#     l_values = np.logspace(-4, 2, 13)
#     pi_t_values = [0.1]  # Only used for prior-weighted configuration
#     configurations = [
#         {'name': 'Normal', 'prior_weighted': False, 'quadratic': False},
#         {'name': 'Prior-Weighted', 'prior_weighted': True, 'quadratic': False},
#         {'name': 'Quadratic', 'prior_weighted': False, 'quadratic': True},
#     ]
#     print("LogREg 1")
#     results = []
#     models_log = {}
#     for config in configurations:
#         for l in l_values:
#             if config['quadratic']:
#                 DTR_exp = expand_features(DTR)
#                 DTE_exp = expand_features(DTE)
#             else:
#                 DTR_exp, DTE_exp = DTR, DTE
#
#             if config['prior_weighted']:
#                 for pi_t in pi_t_values:
#                     config_name = f"{config['name']} (pi_t={pi_t})"
#                     logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=True, pi_t=pi_t)
#                     logreg_classifier.train()
#                     scores = logreg_classifier.predict(DTE_exp)
#                     predictions = logreg_classifier.compute_predictions(scores)
#                     error_rate = logreg_classifier.compute_error_rate(predictions, LTE)
#                     print(f"Config: {config_name}, Lambda: {l}, Error Rate: {error_rate}")
#                     results.append((config_name, l, error_rate))
#                     models_log[config_name] = logreg_classifier
#             else:
#                 config_name = config['name']
#                 logreg_classifier = LogRegClass(DTR_exp, LTR, l, prior_weighted=False)
#                 logreg_classifier.train()
#                 scores = logreg_classifier.predict(DTE_exp)
#                 predictions = logreg_classifier.compute_predictions(scores)
#                 error_rate = logreg_classifier.compute_error_rate(predictions, LTE)
#                 print(f"Config: {config_name}, Lambda: {l}, Error Rate: {error_rate}")
#                 results.append((config_name, l, error_rate))
#                 models_log[config_name] = logreg_classifier
#     print("LogREg 2")
#     # Use the new plotting function
#     LogRegClass.plot_error_rates(results, output_file='error_rates.png')
#     # PROJECT PART
#     # Change the Output paths to save them in the /Output folder
#     output_dir = "Output/LogisticRegression_1"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     lambdas = np.logspace(-4, 2, 13)
#     analyze_dcf_vs_lambda(DTR, LTR, DTE, LTE, lambdas)
#     analyze_with_fewer_samples(DTR, LTR, DTE, LTE, lambdas)
#     analyze_prior_weighted(DTR, LTR, DTE, LTE, lambdas)
#     analyze_quadratic(DTR, LTR, DTE, LTE, lambdas)
#     analyze_preprocessing(DTR, LTR, DTE, LTE, lambdas, output_file_prefix=os.path.join(output_dir, 'dcf_vs_lambda'))
#     results = compare_all_models(DTR, LTR, DTE, LTE, lambdas, pi_t=0.1)
#     # Print results for comparison
#     for model_name, (actual_dcfs, min_dcfs) in results.items():
#         print(f"Results for {model_name}:")
#         for l, actual_dcf, min_dcf in zip(lambdas, actual_dcfs, min_dcfs):
#             print(f"Lambda: {l}, Actual DCF: {actual_dcf}, Min DCF: {min_dcf}")
#     # Plot ROC and Bayes error plots for selected models
#     selected_models = ['Normal', 'Prior-Weighted', 'Quadratic']
#     for model_name in selected_models:
#         model_key = f"{model_name} (pi_t=0.1)" if model_name == 'Prior-Weighted' else model_name
#         print("Iterating, now on " + model_key)
#         model = models_log[model_key]
#         if model_name == 'Quadratic':
#             DTE_exp = expand_features(DTE)
#             scores = model.predict(DTE_exp)
#         else:
#             scores = model.predict(DTE)
#
#         # Compute and remove the log-odds of the empirical prior for actual DCF
#         if 'Prior-Weighted' in model_key:
#             emp_prior = np.mean(LTR)
#             scores = scores - np.log(emp_prior / (1 - emp_prior))
#
#         # Compute actual DCF
#         actual_dcf = model.compute_dcf(scores, LTE, pi_t=0.1)
#         print(f"{model_name} Actual DCF: {actual_dcf}")
#
#         # Compute min DCF
#         min_dcf = model.compute_min_dcf(scores, LTE, pi_t=0.1)
#         print(f"{model_name} Min DCF: {min_dcf}")
#
#         # Plot ROC curve
#         fpr, tpr, _ = roc_curve(LTE, scores)
#         roc_auc = auc(fpr, tpr)
#         plot_roc_curve(fpr, tpr, roc_auc, model_name, os.path.join(output_dir, f'ROC_{model_name}.png'))
#
#         # Plot Bayes error plot
#         plot_bayes_error(scores, LTE, np.linspace(-3, 3, 21),
#                          os.path.join(output_dir, f'Bayes_Error_{model_name}.png'))
