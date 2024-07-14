import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from Models.LogisticRegression.logistic_regression import *
from Models.MixtureModels.gmm import evaluate_classification, GMMClass, train_GMM
from load import *

from Preprocess.DatasetPlots import *
from Preprocess.PCA import *
from Preprocess.LDA import *

from Models.Gaussian.gaussian_models import *

# from Train.train import *
# from Models.Gaussian.new_MVG_model import *
# from Models.Gaussian.class_MVG import *
# from Models.LogisticRegression.logreg import LR_Classifier, QuadraticLR_Classifier
from Models.SupportVector.svm import *
from Models.LogisticRegression.logistic_regression import *

# from Models.SupportVector.svm_kernel import SVMClass
# from Models.MixtureModels.gmm import GMMClass
# from Evaluation.evaluation import *
# from Calibration.calibration import *

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


def train_classifiers(DTR, LTR):
    gmm_model = GMMClass()
    gmm_model.train(DTR, LTR)

    lr_model = LogRegClass(DTR, LTR, l=1.0)
    lr_model.train()

    svm_model = SVMClassifier(kernel='linear')
    svm_model.train(DTR, LTR)

    return gmm_model, lr_model, svm_model


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def compute_llrs(logS):
    return logS[1] - logS[0]


def calibrate_scores(scores, labels):
    # Print the shapes before reshaping
    print(f"Original scores shape: {scores.shape}")
    print(f"Original labels shape: {labels.shape}")

    # Reshape scores to a 2-D array with shape (1, n) if it's not already 2-D
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    labels = labels.reshape(-1)  # Ensure labels are 1-D

    # Print the shapes after reshaping
    print(f"Reshaped scores shape: {scores.shape}")
    print(f"Reshaped labels shape: {labels.shape}")

    # Initialize the LogRegClass with reshaped scores
    model = LogRegClass(scores, labels, l=1.0)
    model.train()

    # Predict calibrated scores
    calibrated_scores = model.predict(scores)
    return calibrated_scores


def compute_min_dcf(scores, labels, pi_t):
    thresholds = np.sort(scores)
    min_dcf = float('inf')
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        fnr = np.mean(predictions[labels == 1] == 0)
        fpr = np.mean(predictions[labels == 0] == 1)
        dcf = pi_t * fnr + (1 - pi_t) * fpr
        if dcf < min_dcf:
            min_dcf = dcf
    return min_dcf


def compute_dcf(scores, labels, pi_t):
    thresholds = np.sort(scores)
    min_dcf = float('inf')
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        dcf = pi_t * np.mean(predictions[labels == 1] == 0) + (1 - pi_t) * np.mean(predictions[labels == 0] == 1)
        if dcf < min_dcf:
            min_dcf = dcf
    return min_dcf


def plot_roc_curve(fpr, tpr, roc_auc, output_file):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_bayes_error(scores, labels, pi_values, output_file):
    fnr = []
    fpr = []
    for pi in pi_values:
        threshold = -np.log(pi / (1 - pi))
        predictions = (scores >= threshold).astype(int)
        fnr.append(np.mean(predictions[labels == 1] == 0))
        fpr.append(np.mean(predictions[labels == 0] == 1))

    plt.plot(pi_values, fnr, label='FNR')
    plt.plot(pi_values, fpr, label='FPR')
    plt.xlim([pi_values[0], pi_values[-1]])
    plt.xlabel('log(π/(1-π))')
    plt.ylabel('Error Rate')
    plt.title('Bayes Error Plot')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def single_fold_calibration(D, L, eval_data, eval_labels, system_1_scores, system_2_scores):
    # Split data
    n_train = int(0.7 * D.shape[1])
    D_train, L_train = D[:, :n_train], L[:n_train]
    D_valid, L_valid = D[:, n_train:], L[n_train:]

    # Calibrate System 1
    system_1_calibrated = calibrate_scores(system_1_scores[:n_train], L_train)

    # Calibrate System 2
    system_2_calibrated = calibrate_scores(system_2_scores[:n_train], L_train)

    # Compute DCF for validation
    system_1_dcf = compute_dcf(system_1_calibrated, L_valid, 0.5)
    system_2_dcf = compute_dcf(system_2_calibrated, L_valid, 0.5)
    fusion_scores = 0.5 * (system_1_calibrated + system_2_calibrated)
    fusion_dcf = compute_dcf(fusion_scores, L_valid, 0.5)

    # Compute DCF for evaluation
    system_1_eval_dcf = compute_dcf(system_1_scores[n_train:], eval_labels, 0.5)
    system_2_eval_dcf = compute_dcf(system_2_scores[n_train:], eval_labels, 0.5)
    fusion_eval_scores = 0.5 * (system_1_scores[n_train:] + system_2_scores[n_train:])
    fusion_eval_dcf = compute_dcf(fusion_eval_scores, eval_labels, 0.5)

    return system_1_dcf, system_2_dcf, fusion_dcf, system_1_eval_dcf, system_2_eval_dcf, fusion_eval_dcf


def k_fold_calibration(D, L, eval_data, eval_labels, system_1_scores, system_2_scores, k=5):
    fold_size = D.shape[1] // k
    system_1_dcfs = []
    system_2_dcfs = []
    fusion_dcfs = []

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        D_train = np.hstack([D[:, :val_start], D[:, val_end:]])
        L_train = np.hstack([L[:val_start], L[val_end:]])
        D_val = D[:, val_start:val_end]
        L_val = L[val_start:val_end]

        # Calibrate System 1
        system_1_calibrated = calibrate_scores(system_1_scores[val_start:val_end], L_train)

        # Calibrate System 2
        system_2_calibrated = calibrate_scores(system_2_scores[val_start:val_end], L_train)

        # Compute DCF
        system_1_dcf = compute_dcf(system_1_calibrated, L_val, 0.5)
        system_2_dcf = compute_dcf(system_2_calibrated, L_val, 0.5)
        fusion_scores = 0.5 * (system_1_calibrated + system_2_calibrated)
        fusion_dcf = compute_dcf(fusion_scores, L_val, 0.5)

        system_1_dcfs.append(system_1_dcf)
        system_2_dcfs.append(system_2_dcf)
        fusion_dcfs.append(fusion_dcf)

    # Compute average DCF for each system and fusion
    avg_system_1_dcf = np.mean(system_1_dcfs)
    avg_system_2_dcf = np.mean(system_2_dcfs)
    avg_fusion_dcf = np.mean(fusion_dcfs)

    # Compute DCF for evaluation
    system_1_eval_dcf = compute_dcf(system_1_scores, eval_labels, 0.5)
    system_2_eval_dcf = compute_dcf(system_2_scores, eval_labels, 0.5)
    fusion_eval_scores = 0.5 * (system_1_scores + system_2_scores)
    fusion_eval_dcf = compute_dcf(fusion_eval_scores, eval_labels, 0.5)

    return avg_system_1_dcf, avg_system_2_dcf, avg_fusion_dcf, system_1_eval_dcf, system_2_eval_dcf, fusion_eval_dcf


def fuse_scores(score_list):
    fused_scores = np.mean(np.array(score_list), axis=0)
    return fused_scores


def evaluate_performance(D, L, scores, pi_values, name):
    min_dcf_list = []
    act_dcf_list = []

    for pi in pi_values:
        pi_t = 1 / (1 + np.exp(-pi))
        min_dcf = compute_min_dcf(scores, L, pi_t)
        act_dcf = compute_dcf(scores, L, pi_t)
        min_dcf_list.append(min_dcf)
        act_dcf_list.append(act_dcf)

    plot_dcf_over_prior(min_dcf_list, act_dcf_list, pi_values, f'{name}_dcf.png')

    return min_dcf_list, act_dcf_list


def evaluate_model(DTR, LTR, DTE, LTE, pi_t=0.1):
    classifier = LogRegClass(DTR, LTR, l=1.0)
    classifier.train()
    scores = classifier.predict(DTE)
    min_dcf = classifier.compute_min_dcf(scores, LTE, pi_t)
    act_dcf = classifier.compute_dcf(scores, LTE, pi_t)
    return min_dcf, act_dcf


def plot_dcf_over_prior(scores, labels, pi_values, output_file):
    dcf = []
    mindcf = []
    for pi in pi_values:
        pi1 = 1 / (1 + np.exp(-pi))
        decisions = compute_optimal_bayes_decisions(llrs=scores, pi1=pi1, Cfn=1, Cfp=1)
        confusion_matrix = compute_confusion_matrix(decisions, labels)
        bayes_risk = compute_bayes_risk(confusion_matrix, pi1, 1, 1)
        normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, 1, 1)
        dcf.append(normalized_dcf)
        min_normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, 1, 1)
        mindcf.append(min_normalized_dcf)
    plt.plot(pi_values, dcf, label='DCF')
    plt.plot(pi_values, mindcf, label='min DCF')
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF value')
    plt.legend()
    plt.grid()
    plt.title('DCF over Prior Log-Odds')
    plt.savefig(output_file)
    plt.close()


# def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file):
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:0.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic - {model_name}')
#     plt.legend(loc="lower right")
#     plt.savefig(output_file)
#     plt.close()


def normalize_data(D):
    mean = np.mean(D, axis=1, keepdims=True)
    std = np.std(D, axis=1, keepdims=True)
    normalized_D = (D - mean) / std
    return normalized_D


def center_data(D):
    mean = np.mean(D, axis=1, keepdims=True)
    centered_D = D - mean
    return centered_D, mean


def main(type, mode):
    (DTR, LTR), (DTE, LTE) = loadTrainingAndTestData('Data/trainData.txt', 'Data/trainData.txt')

    (DTR, LTR), (DTE, LTE) = split_db_2to1(DTR, LTR)
    ciao = False
    print("ciao")
    #if ciao:
    ############################ DATA ANALYSIS - LAB 2 ############################

    data_analysis(DTR, LTR)

    ############################ PCA & LDA - LAB 3  ############################

    #PCA_LDA_analysis(DTE, DTR, LTR, LTE)

    ############################ UNIVARIATE GAUSSIAN MODELS - LAB 4 ############################

    #Univariate_model(DTE, DTR, LTE, LTR)

    ############################ MULTIVARIATE GAUSSIAN MODELS - LAB 5 ############################

    #train_MVG_1(DTE, DTR, LTE, LTR)

    ############################ MULTIVARIATE GAUSSIAN MODELS - PLOTS FOR LAB 7 ############################

    #train_MVG(DTE, DTR, LTE, LTR)

    ############################ LOGISTIC REGRESSION - LAB 8 ############################

    #train_LR(DTE, DTR, LTE, LTR)

    ############################ SVM - LAB 9 ############################

    train_SVM(DTE, DTR, LTE, LTR)

    ############################ GMM - LAB 10 ############################

    train_GMM(DTE, DTR, LTE, LTR)

    ############################ CALIBRATION - LAB 11 ###########################

    D_train, L_train = DTR, LTR
    D_valid, L_valid = DTE, LTE
    D_eval, L_eval = load_data('Data/evalData.txt')

    # Train classifiers
    gmm_model = GMMClass()
    lr_model = LogRegClass(D_train, L_train, l=1.0)
    svm_model = SVMClassifier(kernel='linear')

    # Train GMM
    gmm_model.LBG(D_train, max_components=8)
    gmm_model.EM(D_train)

    # Train Logistic Regression
    lr_model.train()

    # Train SVM
    svm_model.train(D_train, L_train)

    # Get scores for validation data
    gmm_scores_valid = gmm_model.logpdf_GMM(D_valid)
    lr_scores_valid = lr_model.predict(D_valid)
    svm_scores_valid = svm_model.project(D_valid)

    # Calibrate scores
    gmm_calibrated_scores = calibrate_scores(gmm_scores_valid, L_valid)
    lr_calibrated_scores = calibrate_scores(lr_scores_valid, L_valid)
    svm_calibrated_scores = calibrate_scores(svm_scores_valid, L_valid)

    # Fuse scores
    fused_scores_valid = fuse_scores([gmm_calibrated_scores, lr_calibrated_scores, svm_calibrated_scores])

    # Evaluate on validation data
    pi_values = np.linspace(-3, 3, 21)
    gmm_min_dcf, gmm_act_dcf = evaluate_performance(D_valid, L_valid, gmm_calibrated_scores, pi_values, "gmm_valid")
    lr_min_dcf, lr_act_dcf = evaluate_performance(D_valid, L_valid, lr_calibrated_scores, pi_values, "lr_valid")
    svm_min_dcf, svm_act_dcf = evaluate_performance(D_valid, L_valid, svm_calibrated_scores, pi_values, "svm_valid")
    fusion_min_dcf, fusion_act_dcf = evaluate_performance(D_valid, L_valid, fused_scores_valid, pi_values,
                                                          "fusion_valid")

    # Print results for validation data
    print("Validation Results:")
    print(f"GMM: minDCF={gmm_min_dcf}, actDCF={gmm_act_dcf}")
    print(f"LR: minDCF={lr_min_dcf}, actDCF={lr_act_dcf}")
    print(f"SVM: minDCF={svm_min_dcf}, actDCF={svm_act_dcf}")
    print(f"Fusion: minDCF={fusion_min_dcf}, actDCF={fusion_act_dcf}")

    # Get scores for evaluation data
    gmm_scores_eval = gmm_model.logpdf_GMM(D_eval)
    lr_scores_eval = lr_model.predict(D_eval)
    svm_scores_eval = svm_model.project(D_eval)

    # Calibrate scores using evaluation labels
    gmm_calibrated_scores_eval = calibrate_scores(gmm_scores_eval, L_eval)
    lr_calibrated_scores_eval = calibrate_scores(lr_scores_eval, L_eval)
    svm_calibrated_scores_eval = calibrate_scores(svm_scores_eval, L_eval)

    # Fuse scores
    fused_scores_eval = fuse_scores([gmm_calibrated_scores_eval, lr_calibrated_scores_eval, svm_calibrated_scores_eval])

    # Evaluate on evaluation data
    gmm_min_dcf_eval, gmm_act_dcf_eval = evaluate_performance(D_eval, L_eval, gmm_calibrated_scores_eval, pi_values,
                                                              "gmm_eval")
    lr_min_dcf_eval, lr_act_dcf_eval = evaluate_performance(D_eval, L_eval, lr_calibrated_scores_eval, pi_values,
                                                            "lr_eval")
    svm_min_dcf_eval, svm_act_dcf_eval = evaluate_performance(D_eval, L_eval, svm_calibrated_scores_eval, pi_values,
                                                              "svm_eval")
    fusion_min_dcf_eval, fusion_act_dcf_eval = evaluate_performance(D_eval, L_eval, fused_scores_eval, pi_values,
                                                                    "fusion_eval")

    # Print results for evaluation data
    print("Evaluation Results:")
    print(f"GMM: minDCF={gmm_min_dcf_eval}, actDCF={gmm_act_dcf_eval}")
    print(f"LR: minDCF={lr_min_dcf_eval}, actDCF={lr_act_dcf_eval}")
    print(f"SVM: minDCF={svm_min_dcf_eval}, actDCF={svm_act_dcf_eval}")
    print(f"Fusion: minDCF={fusion_min_dcf_eval}, actDCF={fusion_act_dcf_eval}")

    # FINE TEST COMPATTEZZA
    return

    ############################ DATA ANALYSIS ############################

    data_analysis(DTR, LTR)

    ############################ PCA & LDA  ############################

    PCA_LDA_analysis(DTE, DTR, LTR, LTE)

    ############################ UNIVARIATE GAUSSIAN MODELS ############################

    Univariate_model(DTE, DTR, LTE, LTR)

    ############################ MULTIVARIATE GAUSSIAN MODELS ############################

    train_MVG_1(DTE, DTR, LTE, LTR)

    # # MVG
    # mvg_classifier = GaussianClassifier(model_type='MVG')
    # mvg_classifier.train(DTR, LTR)
    # logS = mvg_classifier.predict(DTE)
    # predictions = mvg_classifier.compute_predictions(logS)
    # error_rate = mvg_classifier.compute_error_rate(predictions, LTE)
    # print(f"MVG Error Rate: {error_rate}")
    #
    # # Naive Bayes
    # nb_classifier = GaussianClassifier(model_type='NaiveBayes')
    # nb_classifier.train(DTR, LTR)
    # logS_nb = nb_classifier.predict(DTE)
    # predictions_nb = nb_classifier.compute_predictions(logS_nb)
    # error_rate_nb = nb_classifier.compute_error_rate(predictions_nb, LTE)
    # print(f"Naive Bayes Error Rate: {error_rate_nb}")
    #
    # # Tied Covariance
    # tied_classifier = GaussianClassifier(model_type='TiedCovariance')
    # tied_classifier.train(DTR, LTR)
    # logS_tied = tied_classifier.predict(DTE)
    # predictions_tied = tied_classifier.compute_predictions(logS_tied)
    # error_rate_tied = tied_classifier.compute_error_rate(predictions_tied, LTE)
    # print(f"Tied Covariance Error Rate: {error_rate_tied}")
    #
    # # Project-specific implementation
    # output_dir = 'Output/UnivariateGaussians'
    # mvg_classifier.fit_univariate_gaussian_models(DTR, LTR, output_dir)
    #
    # classes = np.unique(LTR)
    # for cls in classes:
    #     for i in range(DTR.shape[0]):
    #         feature_data = DTR[i, LTR == cls]
    #         mu_ML = feature_data.mean()
    #         C_ML = feature_data.var()
    #         ll = mvg_classifier.plot_loglikelihood(feature_data, np.array([[mu_ML]]), np.array([[C_ML]]),
    #                                                os.path.join(output_dir,
    #                                                             f'LogLikelihood_Class_{cls}_Feature_{i + 1}.png'))
    #         print(f"Class {cls}, Feature {i + 1} Log-Likelihood: {ll}")
    #
    # # Compute covariance and correlation matrices for MVG model
    # mvg_classifier.analyze_covariances(DTR, LTR)
    #
    # # Repeat the analysis using only features 1 to 4 (discarding the last 2 features)
    # DTR_reduced = DTR[:4, :]
    # DTE_reduced = DTE[:4, :]
    #
    # # MVG with reduced features
    # mvg_classifier_reduced = GaussianClassifier(model_type='MVG')
    # mvg_classifier_reduced.train(DTR_reduced, LTR)
    # logS_reduced = mvg_classifier_reduced.predict(DTE_reduced)
    # predictions_reduced = mvg_classifier_reduced.compute_predictions(logS_reduced)
    # error_rate_reduced = mvg_classifier_reduced.compute_error_rate(predictions_reduced, LTE)
    # print(f"MVG Error Rate with reduced features: {error_rate_reduced}")
    #
    # # Tied Covariance with reduced features
    # tied_classifier_reduced = GaussianClassifier(model_type='TiedCovariance')
    # tied_classifier_reduced.train(DTR_reduced, LTR)
    # logS_tied_reduced = tied_classifier_reduced.predict(DTE_reduced)
    # predictions_tied_reduced = tied_classifier_reduced.compute_predictions(logS_tied_reduced)
    # error_rate_tied_reduced = tied_classifier_reduced.compute_error_rate(predictions_tied_reduced, LTE)
    # print(f"Tied Covariance Error Rate with reduced features: {error_rate_tied_reduced}")
    #
    # # Naive Bayes with reduced features
    # nb_classifier_reduced = GaussianClassifier(model_type='NaiveBayes')
    # nb_classifier_reduced.train(DTR_reduced, LTR)
    # logS_nb_reduced = nb_classifier_reduced.predict(DTE_reduced)
    # predictions_nb_reduced = nb_classifier_reduced.compute_predictions(logS_nb_reduced)
    # error_rate_nb_reduced = nb_classifier_reduced.compute_error_rate(predictions_nb_reduced, LTE)
    # print(f"Naive Bayes Error Rate with reduced features: {error_rate_nb_reduced}")
    #
    # # Use PCA to reduce the dimensionality and apply the three classification approaches
    # pca_dim = 2  # For example, reduce to 2 principal components
    # P_pca = estimate_pca(DTR, pca_dim)
    # DTR_pca = apply_pca(DTR, P_pca)
    # DTE_pca = apply_pca(DTE, P_pca)
    #
    # # MVG with PCA
    # mvg_classifier_pca = GaussianClassifier(model_type='MVG')
    # mvg_classifier_pca.train(DTR_pca, LTR)
    # logS_pca = mvg_classifier_pca.predict(DTE_pca)
    # predictions_pca = mvg_classifier_pca.compute_predictions(logS_pca)
    # error_rate_pca = mvg_classifier_pca.compute_error_rate(predictions_pca, LTE)
    # print(f"MVG Error Rate with PCA: {error_rate_pca}")
    #
    # # Tied Covariance with PCA
    # tied_classifier_pca = GaussianClassifier(model_type='TiedCovariance')
    # tied_classifier_pca.train(DTR_pca, LTR)
    # logS_tied_pca = tied_classifier_pca.predict(DTE_pca)
    # predictions_tied_pca = tied_classifier_pca.compute_predictions(logS_tied_pca)
    # error_rate_tied_pca = tied_classifier_pca.compute_error_rate(predictions_tied_pca, LTE)
    # print(f"Tied Covariance Error Rate with PCA: {error_rate_tied_pca}")
    #
    # # Naive Bayes with PCA
    # nb_classifier_pca = GaussianClassifier(model_type='NaiveBayes')
    # nb_classifier_pca.train(DTR_pca, LTR)
    # logS_nb_pca = nb_classifier_pca.predict(DTE_pca)
    # predictions_nb_pca = nb_classifier_pca.compute_predictions(logS_nb_pca)
    # error_rate_nb_pca = nb_classifier_pca.compute_error_rate(predictions_nb_pca, LTE)
    # print(f"Naive Bayes Error Rate with PCA: {error_rate_nb_pca}")

    ############################ MULTIVARIATE GAUSSIAN MODELS - PLOTS FOR LAB 7 ############################

    # # MVG
    # mvg_classifier = GaussianClassifier(model_type='MVG')
    # mvg_classifier.train(DTR, LTR)
    # logS = mvg_classifier.predict(DTE)
    # predictions = mvg_classifier.compute_predictions(logS)
    # error_rate = mvg_classifier.compute_error_rate(predictions, LTE)
    # print(f"MVG Error Rate: {error_rate}")
    #
    # # Naive Bayes
    # nb_classifier = GaussianClassifier(model_type='NaiveBayes')
    # nb_classifier.train(DTR, LTR)
    # logS_nb = nb_classifier.predict(DTE)
    # predictions_nb = nb_classifier.compute_predictions(logS_nb)
    # error_rate_nb = nb_classifier.compute_error_rate(predictions_nb, LTE)
    # print(f"Naive Bayes Error Rate: {error_rate_nb}")
    #
    # # Tied Covariance
    # tied_classifier = GaussianClassifier(model_type='TiedCovariance')
    # tied_classifier.train(DTR, LTR)
    # logS_tied = tied_classifier.predict(DTE)
    # predictions_tied = tied_classifier.compute_predictions(logS_tied)
    # error_rate_tied = tied_classifier.compute_error_rate(predictions_tied, LTE)
    # print(f"Tied Covariance Error Rate: {error_rate_tied}")
    #
    # # Project-specific implementation
    # output_dir = 'Output/UnivariateGaussians'
    # mvg_classifier.fit_univariate_gaussian_models(DTR, LTR, output_dir)
    #
    # classes = np.unique(LTR)
    # for cls in classes:
    #     for i in range(DTR.shape[0]):
    #         feature_data = DTR[i, LTR == cls]
    #         mu_ML = feature_data.mean()
    #         C_ML = feature_data.var()
    #         ll = mvg_classifier.plot_loglikelihood(feature_data, np.array([[mu_ML]]), np.array([[C_ML]]),
    #                                                os.path.join(output_dir,
    #                                                             f'LogLikelihood_Class_{cls}_Feature_{i + 1}.png'))
    #         print(f"Class {cls}, Feature {i + 1} Log-Likelihood: {ll}")
    #
    # # Compute ROC curve and Bayes error plots
    # llrs = mvg_classifier.compute_llrs(logS)
    # mvg_classifier.plot_roc_curve(llrs, LTE, output_file='mvg_roc_curve.png')
    # effPriorLogOdds = np.linspace(-3, 3, 21)
    # mvg_classifier.plot_bayes_error(llrs, LTE, effPriorLogOdds, output_file='mvg_bayes_error_plot.png')
    #
    # # Repeat for Naive Bayes
    # llrs_nb = nb_classifier.compute_llrs(logS_nb)
    # nb_classifier.plot_roc_curve(llrs_nb, LTE, output_file='nb_roc_curve.png')
    # nb_classifier.plot_bayes_error(llrs_nb, LTE, effPriorLogOdds, output_file='nb_bayes_error_plot.png')
    #
    # # Repeat for Tied Covariance
    # llrs_tied = tied_classifier.compute_llrs(logS_tied)
    # tied_classifier.plot_roc_curve(llrs_tied, LTE, output_file='tied_roc_curve.png')
    # tied_classifier.plot_bayes_error(llrs_tied, LTE, effPriorLogOdds, output_file='tied_bayes_error_plot.png')
    #
    # # Analyze and compare DCFs
    # for pi1, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:
    #     print(f"Analyzing for pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    #     for classifier, name in [(mvg_classifier, 'MVG'), (nb_classifier, 'Naive Bayes'), (tied_classifier, 'Tied Covariance')]:
    #         decisions = classifier.compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp)
    #         confusion_matrix = classifier.compute_confusion_matrix(decisions, LTE)
    #         bayes_risk = classifier.compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
    #         normalized_dcf = classifier.compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
    #         print(f"{name} Normalized DCF: {normalized_dcf}")

    train_MVG(DTE, DTR, LTE, LTR, output_dir)

    # Fine for

    ############################ LOGISTIC REGRESSION - LAB 8 ############################

    train_LR(DTE, DTR, LTE, LTR)

    ############################ SVM - LAB 9 ############################

    # lambdas = np.logspace(-4, 2, 13)
    # pi_t = 0.1
    #
    # svm_classifier = SVMClassifier(kernel='linear', C=1.0)
    # svm_classifier.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, 'Output/SVM_Linear')
    #
    # svm_classifier = SVMClassifier(kernel='poly', degree=2, coef0=1.0, C=1.0)
    # svm_classifier.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, 'Output/SVM_Poly')
    #
    # gammas = np.exp(np.array([-4, -3, -2, -1]))
    # for gamma in gammas:
    #     svm_classifier = SVMClassifier(kernel='rbf', gamma=gamma, C=1.0)
    #     svm_classifier.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, f'Output/SVM_RBF_Gamma_{gamma}')
    # print("SVM 1")
    #
    # lambdas = np.logspace(-4, 2, 13)
    # pi_t = 0.1  # Example target prior
    # output_dir = 'Output/SVM_Results'
    #
    # # Initialize SVM classifier
    # svm_classifier = SVMClassifier(kernel='linear')
    #
    # # Evaluate the model with different values of lambda
    # svm_classifier.evaluate_model(DTR, LTR, DTE, LTE, lambdas, pi_t, output_dir)

    # PROJECT

    train_SVM(DTE, DTR, LTE, LTR)

    ############################ GMM - LAB 10 ############################

    train_GMM(DTE, DTR, LTE, LTR)

    '''OLD STUFF 

    # # Fit uni-variate Gaussian models and plot results
    # fit_univariate_gaussian_models(DTR, LTR)
    #
    # # Plot log-likelihood for each feature and class
    # output_dir = 'Output/UnivariateGaussians'
    # classes = np.unique(LTR)
    # for cls in classes:
    #     for i in range(DTR.shape[0]):
    #         feature_data = DTR[i, LTR == cls]
    #         mu_ML = feature_data.mean()
    #         C_ML = feature_data.var()
    #         ll = plot_loglikelihood(feature_data, np.array([[mu_ML]]), np.array([[C_ML]]),
    #                                 os.path.join(output_dir, f'LogLikelihood_Class_{cls}_Feature_{i + 1}.png'))
    #         print(f"Class {cls}, Feature {i + 1} Log-Likelihood: {ll}")

    ############################ OTHER MODELS ############################

    # MVG, Tied, Naive Bayes Evaluation
    evaluate_classifiers(DTR, LTR, DTE, LTE)

    # Analyze Covariances
    analyze_covariances(DTR, LTR)

    # MVG Classifier
    logS_mvg = mvg_classifier(DTR, LTR, DTE)
    predictions_mvg = compute_predictions(logS_mvg)
    error_mvg = compute_error_rate(predictions_mvg, LTE)
    print(f"MVG Error Rate: {error_mvg}")

    # Tied Covariance Classifier
    logS_tied = tied_covariance_classifier(DTR, LTR, DTE)
    predictions_tied = compute_predictions(logS_tied)
    error_tied = compute_error_rate(predictions_tied, LTE)
    print(f"Tied Covariance Error Rate: {error_tied}")

    # Naive Bayes Classifier
    logS_nb = naive_bayes_classifier(DTR, LTR, DTE)
    predictions_nb = compute_predictions(logS_nb)
    error_nb = compute_error_rate(predictions_nb, LTE)
    print(f"Naive Bayes Error Rate: {error_nb}")

    # Analyze Covariances
    analyze_covariances(DTR, LTR)

    # Compute and plot confusion matrices
    confusion_matrix_mvg = compute_confusion_matrix(predictions_mvg, LTE)
    confusion_matrix_tied = compute_confusion_matrix(predictions_tied, LTE)
    confusion_matrix_nb = compute_confusion_matrix(predictions_nb, LTE)
    print(f"MVG Confusion Matrix:\n{confusion_matrix_mvg}")
    print(f"Tied Covariance Confusion Matrix:\n{confusion_matrix_tied}")
    print(f"Naive Bayes Confusion Matrix:\n{confusion_matrix_nb}")

    # Evaluate classifiers with different priors and costs
    priors_costs = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]
    for pi1, Cfn, Cfp in priors_costs:
        llrs_mvg = compute_llrs(logS_mvg)
        optimal_decisions_mvg = compute_optimal_bayes_decisions(llrs_mvg, pi1, Cfn, Cfp)
        confusion_matrix_mvg = compute_confusion_matrix(optimal_decisions_mvg, LTE)
        bayes_risk_mvg = compute_bayes_risk(confusion_matrix_mvg, pi1, Cfn, Cfp)
        normalized_dcf_mvg = compute_normalized_dcf(bayes_risk_mvg, pi1, Cfn, Cfp)
        print(f"MVG Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_mvg}, Normalized DCF: {normalized_dcf_mvg}")

        llrs_tied = compute_llrs(logS_tied)
        optimal_decisions_tied = compute_optimal_bayes_decisions(llrs_tied, pi1, Cfn, Cfp)
        confusion_matrix_tied = compute_confusion_matrix(optimal_decisions_tied, LTE)
        bayes_risk_tied = compute_bayes_risk(confusion_matrix_tied, pi1, Cfn, Cfp)
        normalized_dcf_tied = compute_normalized_dcf(bayes_risk_tied, pi1, Cfn, Cfp)
        print(f"Tied Covariance Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_tied}, Normalized DCF: {normalized_dcf_tied}")

        llrs_nb = compute_llrs(logS_nb)
        optimal_decisions_nb = compute_optimal_bayes_decisions(llrs_nb, pi1, Cfn, Cfp)
        confusion_matrix_nb = compute_confusion_matrix(optimal_decisions_nb, LTE)
        bayes_risk_nb = compute_bayes_risk(confusion_matrix_nb, pi1, Cfn, Cfp)
        normalized_dcf_nb = compute_normalized_dcf(bayes_risk_nb, pi1, Cfn, Cfp)
        print(f"Naive Bayes Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_nb}, Normalized DCF: {normalized_dcf_nb}")

    # PCA Analysis
    pca_dim_values = [2, 4, 6]
    for pca_dim in pca_dim_values:
        P_pca = estimate_pca(DTR, pca_dim)
        DTR_pca = apply_pca(DTR, P_pca)
        DTE_pca = apply_pca(DTE, P_pca)

        # MVG Classifier with PCA
        logS_mvg_pca = mvg_classifier(DTR_pca, LTR, DTE_pca)
        llrs_mvg_pca = compute_llrs(logS_mvg_pca)
        optimal_decisions_mvg_pca = compute_optimal_bayes_decisions(llrs_mvg_pca, pi1, Cfn, Cfp)
        confusion_matrix_mvg_pca = compute_confusion_matrix(optimal_decisions_mvg_pca, LTE)
        bayes_risk_mvg_pca = compute_bayes_risk(confusion_matrix_mvg_pca, pi1, Cfn, Cfp)
        normalized_dcf_mvg_pca = compute_normalized_dcf(bayes_risk_mvg_pca, pi1, Cfn, Cfp)
        print(f"MVG Classifier with PCA (dim={pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_mvg_pca}, Normalized DCF: {normalized_dcf_mvg_pca}")

        # Tied Covariance Classifier with PCA
        logS_tied_pca = tied_covariance_classifier(DTR_pca, LTR, DTE_pca)
        llrs_tied_pca = compute_llrs(logS_tied_pca)
        optimal_decisions_tied_pca = compute_optimal_bayes_decisions(llrs_tied_pca, pi1, Cfn, Cfp)
        confusion_matrix_tied_pca = compute_confusion_matrix(optimal_decisions_tied_pca, LTE)
        bayes_risk_tied_pca = compute_bayes_risk(confusion_matrix_tied_pca, pi1, Cfn, Cfp)
        normalized_dcf_tied_pca = compute_normalized_dcf(bayes_risk_tied_pca, pi1, Cfn, Cfp)
        print(f"Tied Covariance Classifier with PCA (dim={pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_tied_pca}, Normalized DCF: {normalized_dcf_tied_pca}")

        # Naive Bayes Classifier with PCA
        logS_nb_pca = naive_bayes_classifier(DTR_pca, LTR, DTE_pca)
        llrs_nb_pca = compute_llrs(logS_nb_pca)
        optimal_decisions_nb_pca = compute_optimal_bayes_decisions(llrs_nb_pca, pi1, Cfn, Cfp)
        confusion_matrix_nb_pca = compute_confusion_matrix(optimal_decisions_nb_pca, LTE)
        bayes_risk_nb_pca = compute_bayes_risk(confusion_matrix_nb_pca, pi1, Cfn, Cfp)
        normalized_dcf_nb_pca = compute_normalized_dcf(bayes_risk_nb_pca, pi1, Cfn, Cfp)
        print(f"Naive Bayes Classifier with PCA (dim={pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_nb_pca}, Normalized DCF: {normalized_dcf_nb_pca}")

    # Bayes Error Plots
    effPriorLogOdds = np.linspace(-3, 3, 21)
    plot_bayes_error(compute_llrs(logS_mvg), LTE, effPriorLogOdds, 'bayes_error_plot_mvg.png')
    plot_bayes_error(compute_llrs(logS_tied), LTE, effPriorLogOdds, 'bayes_error_plot_tied.png')
    plot_bayes_error(compute_llrs(logS_nb), LTE, effPriorLogOdds, 'bayes_error_plot_nb.png')

    # ROC Curve
    plot_roc_curve(compute_llrs(logS_mvg), LTE, 'roc_curve_mvg.png')
    plot_roc_curve(compute_llrs(logS_tied), LTE, 'roc_curve_tied.png')
    plot_roc_curve(compute_llrs(logS_nb), LTE, 'roc_curve_nb.png')

    # Final analysis for best PCA setup
    best_pca_dim = 4  # Assuming 4 is the best from earlier analysis
    P_best_pca = estimate_pca(DTR, best_pca_dim)
    DTR_best_pca = apply_pca(DTR, P_best_pca)
    DTE_best_pca = apply_pca(DTE, P_best_pca)

    # MVG Classifier with best PCA setup
    logS_mvg_best_pca = mvg_classifier(DTR_best_pca, LTR, DTE_best_pca)
    llrs_mvg_best_pca = compute_llrs(logS_mvg_best_pca)
    optimal_decisions_mvg_best_pca = compute_optimal_bayes_decisions(llrs_mvg_best_pca, pi1, Cfn, Cfp)
    confusion_matrix_mvg_best_pca = compute_confusion_matrix(optimal_decisions_mvg_best_pca, LTE)
    bayes_risk_mvg_best_pca = compute_bayes_risk(confusion_matrix_mvg_best_pca, pi1, Cfn, Cfp)
    normalized_dcf_mvg_best_pca = compute_normalized_dcf(bayes_risk_mvg_best_pca, pi1, Cfn, Cfp)
    print(f"MVG Classifier with Best PCA (dim={best_pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
    print(f"Bayes Risk: {bayes_risk_mvg_best_pca}, Normalized DCF: {normalized_dcf_mvg_best_pca}")

    # Tied Covariance Classifier with best PCA setup
    logS_tied_best_pca = tied_covariance_classifier(DTR_best_pca, LTR, DTE_best_pca)
    llrs_tied_best_pca = compute_llrs(logS_tied_best_pca)
    optimal_decisions_tied_best_pca = compute_optimal_bayes_decisions(llrs_tied_best_pca, pi1, Cfn, Cfp)
    confusion_matrix_tied_best_pca = compute_confusion_matrix(optimal_decisions_tied_best_pca, LTE)
    bayes_risk_tied_best_pca = compute_bayes_risk(confusion_matrix_tied_best_pca, pi1, Cfn, Cfp)
    normalized_dcf_tied_best_pca = compute_normalized_dcf(bayes_risk_tied_best_pca, pi1, Cfn, Cfp)
    print(f"Tied Covariance Classifier with Best PCA (dim={best_pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
    print(f"Bayes Risk: {bayes_risk_tied_best_pca}, Normalized DCF: {normalized_dcf_tied_best_pca}")

    # Naive Bayes Classifier with best PCA setup
    logS_nb_best_pca = naive_bayes_classifier(DTR_best_pca, LTR, DTE_best_pca)
    llrs_nb_best_pca = compute_llrs(logS_nb_best_pca)
    optimal_decisions_nb_best_pca = compute_optimal_bayes_decisions(llrs_nb_best_pca, pi1, Cfn, Cfp)
    confusion_matrix_nb_best_pca = compute_confusion_matrix(optimal_decisions_nb_best_pca, LTE)
    bayes_risk_nb_best_pca = compute_bayes_risk(confusion_matrix_nb_best_pca, pi1, Cfn, Cfp)
    normalized_dcf_nb_best_pca = compute_normalized_dcf(bayes_risk_nb_best_pca, pi1, Cfn, Cfp)
    print(f"Naive Bayes Classifier with Best PCA (dim={best_pca_dim}) - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
    print(f"Bayes Risk: {bayes_risk_nb_best_pca}, Normalized DCF: {normalized_dcf_nb_best_pca}")

    # Bayes Error Plots for the best PCA setup
    plot_bayes_error(compute_llrs(logS_mvg_best_pca), LTE, effPriorLogOdds, 'bayes_error_plot_mvg_best_pca.png')
    plot_bayes_error(compute_llrs(logS_tied_best_pca), LTE, effPriorLogOdds, 'bayes_error_plot_tied_best_pca.png')
    plot_bayes_error(compute_llrs(logS_nb_best_pca), LTE, effPriorLogOdds, 'bayes_error_plot_nb_best_pca.png')

    print("Lab 7")

    # Evaluate classifiers with different priors and costs
    priors_costs = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]
    for pi1, Cfn, Cfp in priors_costs:
        llrs_mvg = compute_llrs(logS_mvg)
        optimal_decisions_mvg = compute_optimal_bayes_decisions(llrs_mvg, pi1, Cfn, Cfp)
        confusion_matrix_mvg = compute_confusion_matrix(optimal_decisions_mvg, LTE)
        bayes_risk_mvg = compute_bayes_risk(confusion_matrix_mvg, pi1, Cfn, Cfp)
        normalized_dcf_mvg = compute_normalized_dcf(bayes_risk_mvg, pi1, Cfn, Cfp)
        print(f"MVG Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_mvg}, Normalized DCF: {normalized_dcf_mvg}")

        llrs_tied = compute_llrs(logS_tied)
        optimal_decisions_tied = compute_optimal_bayes_decisions(llrs_tied, pi1, Cfn, Cfp)
        confusion_matrix_tied = compute_confusion_matrix(optimal_decisions_tied, LTE)
        bayes_risk_tied = compute_bayes_risk(confusion_matrix_tied, pi1, Cfn, Cfp)
        normalized_dcf_tied = compute_normalized_dcf(bayes_risk_tied, pi1, Cfn, Cfp)
        print(f"Tied Covariance Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_tied}, Normalized DCF: {normalized_dcf_tied}")

        llrs_nb = compute_llrs(logS_nb)
        optimal_decisions_nb = compute_optimal_bayes_decisions(llrs_nb, pi1, Cfn, Cfp)
        confusion_matrix_nb = compute_confusion_matrix(optimal_decisions_nb, LTE)
        bayes_risk_nb = compute_bayes_risk(confusion_matrix_nb, pi1, Cfn, Cfp)
        normalized_dcf_nb = compute_normalized_dcf(bayes_risk_nb, pi1, Cfn, Cfp)
        print(f"Naive Bayes Classifier - Prior: {pi1}, Cfn: {Cfn}, Cfp: {Cfp}")
        print(f"Bayes Risk: {bayes_risk_nb}, Normalized DCF: {normalized_dcf_nb}")

    # PCA Analysis
    pca_dim_values = [2, 4, 6]
    for pca_dim in pca_dim_values:
        P_pca = estimate_pca(DTR, pca_dim)
        DTR_pca = apply_pca(DTR, P_pca)
        DTE_pca = apply_pca(DTE, P_pca)

        # MVG Classifier with PCA
        logS_mvg_pca = mvg_classifier(DTR_pca, LTR, DTE_pca)
        llrs_mvg_pca = compute_llrs(logS_mvg_pca)
        optimal_decisions_mvg_pca = compute_optimal_bayes_decisions(llrs_mvg_pca, 0.5, 1.0, 1.0)
        confusion_matrix_mvg_pca = compute_confusion_matrix(optimal_decisions_mvg_pca, LTE)
        bayes_risk_mvg_pca = compute_bayes_risk(confusion_matrix_mvg_pca, 0.5, 1.0, 1.0)
        normalized_dcf_mvg_pca = compute_normalized_dcf(bayes_risk_mvg_pca, 0.5, 1.0, 1.0)
        print(f"MVG Classifier with PCA (dim={pca_dim}) - Prior: 0.5, Cfn: 1.0, Cfp: 1.0")
        print(f"Bayes Risk: {bayes_risk_mvg_pca}, Normalized DCF: {normalized_dcf_mvg_pca}")

        # Tied Covariance Classifier with PCA
        logS_tied_pca = tied_covariance_classifier(DTR_pca, LTR, DTE_pca)
        llrs_tied_pca = compute_llrs(logS_tied_pca)
        optimal_decisions_tied_pca = compute_optimal_bayes_decisions(llrs_tied_pca, 0.5, 1.0, 1.0)
        confusion_matrix_tied_pca = compute_confusion_matrix(optimal_decisions_tied_pca, LTE)
        bayes_risk_tied_pca = compute_bayes_risk(confusion_matrix_tied_pca, 0.5, 1.0, 1.0)
        normalized_dcf_tied_pca = compute_normalized_dcf(bayes_risk_tied_pca, 0.5, 1.0, 1.0)
        print(f"Tied Covariance Classifier with PCA (dim={pca_dim}) - Prior: 0.5, Cfn: 1.0, Cfp: 1.0")
        print(f"Bayes Risk: {bayes_risk_tied_pca}, Normalized DCF: {normalized_dcf_tied_pca}")

        # Naive Bayes Classifier with PCA
        logS_nb_pca = naive_bayes_classifier(DTR_pca, LTR, DTE_pca)
        llrs_nb_pca = compute_llrs(logS_nb_pca)
        optimal_decisions_nb_pca = compute_optimal_bayes_decisions(llrs_nb_pca, 0.5, 1.0, 1.0)
        confusion_matrix_nb_pca = compute_confusion_matrix(optimal_decisions_nb_pca, LTE)
        bayes_risk_nb_pca = compute_bayes_risk(confusion_matrix_nb_pca, 0.5, 1.0, 1.0)
        normalized_dcf_nb_pca = compute_normalized_dcf(bayes_risk_nb_pca, 0.5, 1.0, 1.0)
        print(f"Naive Bayes Classifier with PCA (dim={pca_dim}) - Prior: 0.5, Cfn: 1.0, Cfp: 1.0")
        print(f"Bayes Risk: {bayes_risk_nb_pca}, Normalized DCF: {normalized_dcf_nb_pca}")

    # Best PCA Setup (dim=4)
    best_pca_dim = 4
    P_pca = estimate_pca(DTR, best_pca_dim)
    DTR_pca = apply_pca(DTR, P_pca)
    DTE_pca = apply_pca(DTE, P_pca)

    # MVG Classifier with Best PCA
    logS_mvg_best_pca = mvg_classifier(DTR_pca, LTR, DTE_pca)
    llrs_mvg_best_pca = compute_llrs(logS_mvg_best_pca)
    plot_bayes_error(llrs_mvg_best_pca, LTE, np.linspace(-3, 3, 21), 'bayes_error_plot_mvg_best_pca.png')

    # Tied Covariance Classifier with Best PCA
    logS_tied_best_pca = tied_covariance_classifier(DTR_pca, LTR, DTE_pca)
    llrs_tied_best_pca = compute_llrs(logS_tied_best_pca)
    plot_bayes_error(llrs_tied_best_pca, LTE, np.linspace(-3, 3, 21), 'bayes_error_plot_tied_best_pca.png')

    # Naive Bayes Classifier with Best PCA
    logS_nb_best_pca = naive_bayes_classifier(DTR_pca, LTR, DTE_pca)
    llrs_nb_best_pca = compute_llrs(logS_nb_best_pca)
    plot_bayes_error(llrs_nb_best_pca, LTE, np.linspace(-3, 3, 21), 'bayes_error_plot_nb_best_pca.png')

    # ROC Curves
    plot_roc_curve(compute_llrs(logS_mvg), LTE, 'roc_curve_mvg.png')
    plot_roc_curve(compute_llrs(logS_tied), LTE, 'roc_curve_tied.png')
    plot_roc_curve(compute_llrs(logS_nb), LTE, 'roc_curve_nb.png')

    print("Lab 8")

    # 1. Compute actual DCF and minimum DCF for the primary application (πT = 0.1):
    # Logistic Regression without preprocessing
    lambdas = np.logspace(-4, 2, 13)
    actual_dcf = []
    min_dcf = []

    for l in lambdas:
        model = train_logreg(DTR, LTR, l)
        scores = compute_logreg_scores(DTE, model)
        predictions = compute_logreg_predictions(scores)
        error = compute_error_rate(predictions, LTE)
        print(f"Logistic Regression Error Rate (λ={l}): {error}")

        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))

        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf.append(norm_dcf)

        actual_decisions = (adjusted_scores >= 0).astype(int)
        actual_confusion_matrix = compute_confusion_matrix(actual_decisions, LTE)
        actual_bayes_risk = compute_bayes_risk(actual_confusion_matrix, 0.1, 1.0, 1.0)
        actual_norm_dcf = compute_normalized_dcf(actual_bayes_risk, 0.1, 1.0, 1.0)
        actual_dcf.append(actual_norm_dcf)

    plot_metrics(lambdas, actual_dcf, min_dcf, "Logistic Regression DCFs")

    # 2. Repeat the analysis with reduced dataset:
    # Logistic Regression with reduced dataset
    actual_dcf_reduced = []
    min_dcf_reduced = []

    for l in lambdas:
        DTR_reduced = DTR[:, ::50]
        LTR_reduced = LTR[::50]
        model = train_logreg(DTR_reduced, LTR_reduced, l)
        scores = compute_logreg_scores(DTE, model)
        predictions = compute_logreg_predictions(scores)
        error = compute_error_rate(predictions, LTE)
        print(f"Logistic Regression Error Rate with Reduced Data (λ={l}): {error}")

        empirical_prior = LTR_reduced.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))

        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_reduced.append(norm_dcf)

        actual_decisions = (adjusted_scores >= 0).astype(int)
        actual_confusion_matrix = compute_confusion_matrix(actual_decisions, LTE)
        actual_bayes_risk = compute_bayes_risk(actual_confusion_matrix, 0.1, 1.0, 1.0)
        actual_norm_dcf = compute_normalized_dcf(actual_bayes_risk, 0.1, 1.0, 1.0)
        actual_dcf_reduced.append(actual_norm_dcf)

    plot_metrics(lambdas, actual_dcf_reduced, min_dcf_reduced, "Logistic Regression DCFs with Reduced Data")

    # 3. Repeat the analysis with prior-weighted logistic regression model:
    # Prior-Weighted Logistic Regression
    actual_dcf_prior_weighted = []
    min_dcf_prior_weighted = []

    for l in lambdas:
        model = train_logreg(DTR, LTR, l)
        scores = compute_logreg_scores(DTE, model)
        predictions = compute_logreg_predictions(scores)
        error = compute_error_rate(predictions, LTE)
        print(f"Prior-Weighted Logistic Regression Error Rate (λ={l}): {error}")

        adjusted_scores = scores - np.log(0.1 / (1 - 0.1))

        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_prior_weighted.append(norm_dcf)

        actual_decisions = (adjusted_scores >= 0).astype(int)
        actual_confusion_matrix = compute_confusion_matrix(actual_decisions, LTE)
        actual_bayes_risk = compute_bayes_risk(actual_confusion_matrix, 0.1, 1.0, 1.0)
        actual_norm_dcf = compute_normalized_dcf(actual_bayes_risk, 0.1, 1.0, 1.0)
        actual_dcf_prior_weighted.append(actual_norm_dcf)

    plot_metrics(lambdas, actual_dcf_prior_weighted, min_dcf_prior_weighted, "Prior-Weighted Logistic Regression DCFs")

    # 4. Repeat the analysis with quadratic logistic regression model:
    # Quadratic Logistic Regression
    def expand_quadratic_features(D):
        num_features = D.shape[0]
        num_samples = D.shape[1]
        expanded_D = np.zeros((num_features + num_features * (num_features + 1) // 2, num_samples))
        expanded_D[:num_features, :] = D

        idx = num_features
        for i in range(num_features):
            for j in range(i, num_features):
                expanded_D[idx, :] = D[i, :] * D[j, :]
                idx += 1

        return expanded_D

    DTR_quad = expand_quadratic_features(DTR)
    DTE_quad = expand_quadratic_features(DTE)

    actual_dcf_quad = []
    min_dcf_quad = []

    for l in lambdas:
        model = train_logreg(DTR_quad, LTR, l)
        scores = compute_logreg_scores(DTE_quad, model)
        predictions = compute_logreg_predictions(scores)
        error = compute_error_rate(predictions, LTE)
        print(f"Quadratic Logistic Regression Error Rate (λ={l}): {error}")

        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))

        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_quad.append(norm_dcf)

        actual_decisions = (adjusted_scores >= 0).astype(int)
        actual_confusion_matrix = compute_confusion_matrix(actual_decisions, LTE)
        actual_bayes_risk = compute_bayes_risk(actual_confusion_matrix, 0.1, 1.0, 1.0)
        actual_norm_dcf = compute_normalized_dcf(actual_bayes_risk, 0.1, 1.0, 1.0)
        actual_dcf_quad.append(actual_norm_dcf)

    plot_metrics(lambdas, actual_dcf_quad, min_dcf_quad, "Quadratic Logistic Regression DCFs")

    # 5. Analyze the effects of centering on the model results:
    # Analyze the effects of centering on the model results

    # Center the data
    DTR_centered, mean = center_data(DTR)
    DTE_centered, _ = center_data(DTE, mean)

    actual_dcf_centered = []
    min_dcf_centered = []

    for l in lambdas:
        model = train_logreg(DTR_centered, LTR, l)
        scores = compute_logreg_scores(DTE_centered, model)
        predictions = compute_logreg_predictions(scores)
        error = compute_error_rate(predictions, LTE)
        print(f"Centered Logistic Regression Error Rate (λ={l}): {error}")

        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))

        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_centered.append(norm_dcf)

        actual_decisions = (adjusted_scores >= 0).astype(int)
        actual_confusion_matrix = compute_confusion_matrix(actual_decisions, LTE)
        actual_bayes_risk = compute_bayes_risk(actual_confusion_matrix, 0.1, 1.0, 1.0)
        actual_norm_dcf = compute_normalized_dcf(actual_bayes_risk, 0.1, 1.0, 1.0)
        actual_dcf_centered.append(actual_norm_dcf)

    plot_metrics(lambdas, actual_dcf_centered, min_dcf_centered, "Centered Logistic Regression DCFs")

    # 6. Compare all models in terms of minDCF for the target application (πT = 0.1):
    # Compare all models in terms of minDCF for the target application (πT = 0.1)
    min_dcf_all_models = {
        'MVG': [],
        'Tied Covariance': [],
        'Naive Bayes': [],
        'Logistic Regression': [],
        'Prior-Weighted Logistic Regression': [],
        'Quadratic Logistic Regression': [],
        'Centered Logistic Regression': []
    }

    # Add MVG results
    logS_mvg = mvg_classifier(DTR, LTR, DTE)
    llrs_mvg = compute_llrs(logS_mvg)
    optimal_decisions_mvg = compute_optimal_bayes_decisions(llrs_mvg, 0.1, 1.0, 1.0)
    confusion_matrix_mvg = compute_confusion_matrix(optimal_decisions_mvg, LTE)
    bayes_risk_mvg = compute_bayes_risk(confusion_matrix_mvg, 0.1, 1.0, 1.0)
    norm_dcf_mvg = compute_normalized_dcf(bayes_risk_mvg, 0.1, 1.0, 1.0)
    min_dcf_all_models['MVG'].append(norm_dcf_mvg)

    # Add Tied Covariance results
    logS_tied = tied_covariance_classifier(DTR, LTR, DTE)
    llrs_tied = compute_llrs(logS_tied)
    optimal_decisions_tied = compute_optimal_bayes_decisions(llrs_tied, 0.1, 1.0, 1.0)
    confusion_matrix_tied = compute_confusion_matrix(optimal_decisions_tied, LTE)
    bayes_risk_tied = compute_bayes_risk(confusion_matrix_tied, 0.1, 1.0, 1.0)
    norm_dcf_tied = compute_normalized_dcf(bayes_risk_tied, 0.1, 1.0, 1.0)
    min_dcf_all_models['Tied Covariance'].append(norm_dcf_tied)

    # Add Naive Bayes results
    logS_nb = naive_bayes_classifier(DTR, LTR, DTE)
    llrs_nb = compute_llrs(logS_nb)
    optimal_decisions_nb = compute_optimal_bayes_decisions(llrs_nb, 0.1, 1.0, 1.0)
    confusion_matrix_nb = compute_confusion_matrix(optimal_decisions_nb, LTE)
    bayes_risk_nb = compute_bayes_risk(confusion_matrix_nb, 0.1, 1.0, 1.0)
    norm_dcf_nb = compute_normalized_dcf(bayes_risk_nb, 0.1, 1.0, 1.0)
    min_dcf_all_models['Naive Bayes'].append(norm_dcf_nb)

    # Add Logistic Regression results
    for l in lambdas:
        model = train_logreg(DTR, LTR, l)
        scores = compute_logreg_scores(DTE, model)
        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))
        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_all_models['Logistic Regression'].append(norm_dcf)

    # Add Prior-Weighted Logistic Regression results
    for l in lambdas:
        model = train_logreg(DTR, LTR, l)
        scores = compute_logreg_scores(DTE, model)
        adjusted_scores = scores - np.log(0.1 / (1 - 0.1))
        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_all_models['Prior-Weighted Logistic Regression'].append(norm_dcf)

    # Add Quadratic Logistic Regression results
    DTR_quad = expand_quadratic_features(DTR)
    DTE_quad = expand_quadratic_features(DTE)
    for l in lambdas:
        model = train_logreg(DTR_quad, LTR, l)
        scores = compute_logreg_scores(DTE_quad, model)
        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))
        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_all_models['Quadratic Logistic Regression'].append(norm_dcf)

    # Add Centered Logistic Regression results
    DTR_centered, mean = center_data(DTR)
    DTE_centered, _ = center_data(DTE, mean)
    for l in lambdas:
        model = train_logreg(DTR_centered, LTR, l)
        scores = compute_logreg_scores(DTE_centered, model)
        empirical_prior = LTR.mean()
        adjusted_scores = scores - np.log(empirical_prior / (1 - empirical_prior))
        optimal_decisions = compute_optimal_bayes_decisions(adjusted_scores, 0.1, 1.0, 1.0)
        confusion_matrix = compute_confusion_matrix(optimal_decisions, LTE)
        bayes_risk = compute_bayes_risk(confusion_matrix, 0.1, 1.0, 1.0)
        norm_dcf = compute_normalized_dcf(bayes_risk, 0.1, 1.0, 1.0)
        min_dcf_all_models['Centered Logistic Regression'].append(norm_dcf)

    # Plot comparison of all models
    for model_name, dcfs in min_dcf_all_models.items():
        if len(dcfs) == len(lambdas):
            plt.plot(lambdas, dcfs, label=model_name)

    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Min DCF')
    plt.title('Comparison of all models (Min DCF)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_min_dcf.png')
    plt.close()

    print("Lab 9")

    # Define the range of hyperparameters
    Cs = np.logspace(-5, 0, 11)
    gammas = np.logspace(-4, -1, 4)

    # Running the SVM models
    run_linear_svm(DTR, LTR, DTE, LTE, Cs, centered=False)
    run_linear_svm(DTR, LTR, DTE, LTE, Cs, centered=True)
    run_polynomial_svm(DTR, LTR, DTE, LTE, Cs, degree=2, c=1)
    run_rbf_svm(DTR, LTR, DTE, LTE, Cs, gammas)
'''

def Univariate_model(DTE, DTR, LTE, LTR):
    mvg_classifier = GaussianClassifier()
    mvg_classifier.fit_univariate_gaussian_models(DTR, LTR)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classificazione.")
    parser.add_argument('--type', type=str, default='MVG', help='Percorso del file di input')
    parser.add_argument('--mode', type=str, default='train', help='Percorso del file di input')

    args = parser.parse_args()

    main(args.type, args.mode)

"""
 plotSingle(DTR, LTR, 10)
    plotCross(DTR, LTR, 10)
    DC = centerData(DTR)  # delete the mean component from the data

    # plot the features
    plotSingle(DC, LTR, 10)
    plotCross(DC, LTR, 10)

    mp = 6
    # PCA implementation
    DP, P = PCA_impl(DTR, mp)
    DTEP = np.dot(P.T, DTE)

    plotCross(DP, LTR, m)  # plotting the data on a 2-D cartesian graph
    plotSingle(DP, LTR, m)

    ml = 1
    # LDA implementation
    DW, W = LDA_impl(DTR, LTR, ml)
    DTEW = np.dot(W.T, DTE)
    plotCross(DW, LTR, m)
    plotSingle(DW, LTR, ml)

    # LDA + PCA implementation
    DPW, W = LDA_impl(DP, LTR, ml)
    DTEPW = np.dot(W.T, DTEP)

    # Pearson correlation

    plot_features(DTR, LTR)

    # PCA and variance plot
    PCA_plot(DTR)
"""
