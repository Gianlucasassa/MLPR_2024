import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from Models.Gaussian.gaussian_density import fit_univariate_gaussian_models
from Models.LogisticRegression.logistic_regression import *
from Models.MixtureModels.gmm import GMMClass, train_GMM
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


# def train_classifiers(DTR, LTR):
#     gmm_model = GMMClass()
#     gmm_model.train(DTR, LTR)
#
#     lr_model = LogRegClass(DTR, LTR, l=1.0)
#     lr_model.train()
#
#     svm_model = SVMClassifier(kernel='linear')
#     svm_model.train(DTR, LTR)
#
#     return gmm_model, lr_model, svm_model

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


# # def compute_llrs(logS):
# #     return logS[1] - logS[0]
#
#
# def calibrate_scores(scores, labels):
#     # Print the shapes before reshaping
#     print(f"Original scores shape: {scores.shape}")
#     print(f"Original labels shape: {labels.shape}")
#
#     # Reshape scores to a 2-D array with shape (1, n) if it's not already 2-D
#     if scores.ndim == 1:
#         scores = scores.reshape(1, -1)
#     labels = labels.reshape(-1)  # Ensure labels are 1-D
#
#     # Print the shapes after reshaping
#     print(f"Reshaped scores shape: {scores.shape}")
#     print(f"Reshaped labels shape: {labels.shape}")
#
#     # Initialize the LogRegClass with reshaped scores
#     model = LogRegClass(scores, labels, l=1.0)
#     model.train()
#
#     # Predict calibrated scores
#     calibrated_scores = model.predict(scores)
#     return calibrated_scores
#
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
# def compute_dcf(scores, labels, pi_t):
#     thresholds = np.sort(scores)
#     min_dcf = float('inf')
#     for t in thresholds:
#         predictions = (scores >= t).astype(int)
#         dcf = pi_t * np.mean(predictions[labels == 1] == 0) + (1 - pi_t) * np.mean(predictions[labels == 0] == 1)
#         if dcf < min_dcf:
#             min_dcf = dcf
#     return min_dcf
#
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
#     def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
#         thresholds = np.log(pi1 / (1 - pi1))
#         return (llrs >= thresholds).astype(int)
#
#     def compute_confusion_matrix(predictions, labels):
#         TP = np.sum((predictions == 1) & (labels == 1))
#         TN = np.sum((predictions == 0) & (labels == 0))
#         FP = np.sum((predictions == 1) & (labels == 0))
#         FN = np.sum((predictions == 0) & (labels == 1))
#         return np.array([[TN, FP], [FN, TP]])
#
#     def compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
#         FN, FP = confusion_matrix[1, 0], confusion_matrix[0, 1]
#         N0, N1 = confusion_matrix[0, 0] + confusion_matrix[0, 1], confusion_matrix[1, 0] + confusion_matrix[1, 1]
#         Pfn = FN / N1
#         Pfp = FP / N0
#         return pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
#
#     def compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp):
#         dummy_risk = min(pi1 * Cfn, (1 - pi1) * Cfp)
#         return bayes_risk / dummy_risk
#
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
#         min_dcf_value = compute_min_dcf(llrs, labels, pi1, 1, 1)
#         mindcf.append(min_dcf_value)
#
#     plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
#     plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
#     plt.ylim([0, 1.1])
#     plt.xlim([-4, 4])
#     plt.xlabel('Prior Log-Odds')
#     plt.ylabel('DCF value')
#     plt.legend()
#     plt.grid()
#     plt.title('Bayes Error Plot')
#     plt.savefig(output_file)
#     plt.close()
#
#
# def compute_min_dcf(llrs, labels, pi1, Cfn, Cfp):
#     thresholds = np.sort(llrs)
#     min_dcf = float('inf')
#     for threshold in thresholds:
#         predictions = (llrs >= threshold).astype(int)
#         confusion_matrix = compute_confusion_matrix(predictions, labels)
#         bayes_risk = compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
#         normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
#         if normalized_dcf < min_dcf:
#             min_dcf = normalized_dcf
#     return min_dcf
#
#
# """
# CALIB
# """
#
#
# def single_fold_calibration(DTR, LTR, DTE, LTE, model, prior=0.2):
#     # Split the training data into calibration training and validation sets
#     calibration_train_idx = np.arange(0, DTR.shape[1], 3)
#     calibration_val_idx = np.setdiff1d(np.arange(DTR.shape[1]), calibration_train_idx)
#
#     DTR_cal_train = DTR[:, calibration_train_idx]
#     LTR_cal_train = LTR[calibration_train_idx]
#     DTR_cal_val = DTR[:, calibration_val_idx]
#     LTR_cal_val = LTR[calibration_val_idx]
#
#     # Train the model on the calibration training set
#     if isinstance(model, LogRegClass):
#         model.DTR = DTR_cal_train
#         model.LTR = LTR_cal_train
#         model.train()
#     elif isinstance(model, SVMClassifier):
#         model.train(DTR_cal_train, LTR_cal_train)
#     elif isinstance(model, GMMClass):
#         model.train(DTR_cal_train, numComponents=8, psiEig=0.01)
#
#     scores_cal_val = model.predict(DTR_cal_val)
#     print(f"SHAPE OF SCORE CAL VAL = {scores_cal_val.shape}")
#     scores_cal_val = scores_cal_val.flatten()
#     print(f"SHAPE OF SCORE CAL VAL FLATTENED = {scores_cal_val.shape}")
#
#     # Fit a logistic regression model for calibration
#     lr = LogRegClass(None, None, 0)
#     lr.train(scores_cal_val.reshape(-1, 1), LTR_cal_val)
#
#     # Apply the calibration model to the test set
#     scores_test = model.predict(DTE)
#     scores_test = scores_test.flatten()
#     calibrated_scores = lr.predict_proba(scores_test.reshape(-1, 1))[:, 1]
#
#     minDCF = compute_minDCF_binary_fast(calibrated_scores, LTE, prior, 1, 1)
#     actDCF = compute_actDCF_binary_fast(calibrated_scores, LTE, prior, 1, 1)
#
#     return minDCF, actDCF, calibrated_scores
#
#
# def generate_kfold_splits(DTR, LTR, K=5):
#     indices = np.arange(DTR.shape[1])
#     np.random.shuffle(indices)
#     folds = np.array_split(indices, K)
#     return folds
#
#
# def kfold_calibration(DTR, LTR, model, splits, prior=0.2):
#     K = len(splits)
#     calibrated_scores_list = []
#     all_labels = []
#
#     for i in range(K):
#         train_idx = np.hstack([splits[j] for j in range(K) if j != i])
#         val_idx = splits[i]
#
#         DTR_fold = DTR[:, train_idx]
#         LTR_fold = LTR[train_idx]
#         DTE_fold = DTR[:, val_idx]
#         LTE_fold = LTR[val_idx]
#
#         if isinstance(model, LogRegClass):
#             model.DTR = DTR_fold
#             model.LTR = LTR_fold
#             model.train()
#         elif isinstance(model, SVMClassifier):
#             model.train(DTR_fold, LTR_fold)
#         elif isinstance(model, GMMClass):
#             model.train(DTR_fold, numComponents=8, psiEig=0.01)
#
#         fold_scores = model.predict(DTE_fold)
#         fold_scores = fold_scores.flatten()
#
#         # Fit a logistic regression model for calibration
#         lr = LogRegClass(None, None, 0)
#         lr.train(fold_scores.reshape(-1, 1), LTE_fold)
#         calibrated_fold_scores = lr.predict_proba(fold_scores.reshape(-1, 1))[:, 1]
#         calibrated_scores_list.append(calibrated_fold_scores)
#         all_labels.append(LTE_fold)
#
#         # Debug print statements
#         print(f"Fold {i} - Model: {type(model).__name__}")
#         print(f"Train indices: {train_idx[:10]}... Val indices: {val_idx[:10]}...")
#         print(f"LTR_fold: {LTR_fold[:10]}... LTE_fold: {LTE_fold[:10]}")
#         print(f"Fold scores shape: {fold_scores.shape} - Calibrated fold scores shape: {calibrated_fold_scores.shape}")
#         print(f"Fold labels shape: {LTE_fold.shape}")
#
#     # Combine the calibrated scores and labels from all folds
#     calibrated_scores = np.hstack(calibrated_scores_list)
#     all_labels = np.hstack(all_labels)
#
#     # Debug prints
#     print(f"All calibrated scores shape: {calibrated_scores.shape}")
#     print(f"All labels shape: {all_labels.shape}")
#
#     minDCF = compute_minDCF_binary_fast(calibrated_scores, all_labels, prior, 1, 1)
#     actDCF = compute_actDCF_binary_fast(calibrated_scores, all_labels, prior, 1, 1)
#
#     return minDCF, actDCF, calibrated_scores, all_labels
#
#
# def score_level_fusion(scores1, scores2, L, prior=0.2):
#     fused_scores = np.vstack((scores1, scores2)).T
#
#     lr = LogRegClass(None, None, 0)
#     assert fused_scores.shape[0] == L.shape[0], "Mismatch in the number of samples between fused scores and labels"
#     lr.train(fused_scores, L)
#     calibrated_fusion_scores = lr.predict_proba(fused_scores)[:, 1]
#
#     minDCF = compute_minDCF_binary_fast(calibrated_fusion_scores, L, prior, 1, 1)
#     actDCF = compute_actDCF_binary_fast(calibrated_fusion_scores, L, prior, 1, 1)
#
#     return minDCF, actDCF, calibrated_fusion_scores
#
#
# def plot_bayes_error_with_calibration(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
#     dcf = []
#     mindcf = []
#     for p in effPriorLogOdds:
#         pi1 = 1 / (1 + np.exp(-p))
#         decisions = compute_optimal_Bayes_binary_llr(llrs, pi1, 1, 1)
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
# # Define functions for computing DCF
# def compute_confusion_matrix(predictedLabels, classLabels):
#     nClasses = classLabels.max() + 1
#     M = np.zeros((nClasses, nClasses), dtype=np.int32)
#     for i in range(classLabels.size):
#         M[predictedLabels[i], classLabels[i]] += 1
#     return M
#
#
# def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
#     th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
#     return np.int32(llr > th)
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
# # Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
# def compute_actDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, normalize=True):
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
#         if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[idx]:
#             PfnOut.append(Pfn[idx])
#             PfpOut.append(Pfp[idx])
#             thresholdsOut.append(llrSorted[idx])
#
#     return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)
#
#
# def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
#     Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
#     minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1 - prior) * Cfp)
#     idx = np.argmin(minDCF)
#     if returnThreshold:
#         return minDCF[idx], th[idx]
#     else:
#         return minDCF[idx]
#
#
# def plot_dcf_over_prior(min_dcf_list, act_dcf_list, pi_values, filename):
#     plt.figure()
#     plt.plot(pi_values, min_dcf_list, label='minDCF', color='b')
#     plt.plot(pi_values, act_dcf_list, label='actDCF', color='r')
#     plt.xlabel('Prior Log Odds')
#     plt.ylabel('DCF')
#     plt.legend()
#     plt.savefig(filename)
#     plt.close()
#
#
# def evaluate_performance(D, L, scores, pi_values, name):
#     min_dcf_list = []
#     act_dcf_list = []
#
#     for pi in pi_values:
#         min_dcf, act_dcf = compute_minDCF_binary_fast(scores, L, pi, 1.0, 1.0), compute_actDCF_binary_fast(scores, L,
#                                                                                                            pi, 1.0, 1.0)
#         min_dcf_list.append(min_dcf)
#         act_dcf_list.append(act_dcf)
#
#     plot_dcf_over_prior(min_dcf_list, act_dcf_list, pi_values, f'{name}_dcf.png')
#     return min(min_dcf_list), min(act_dcf_list)
#
#
# def evaluate_system(DTR, LTR, DTE, LTE, D_eval, L_eval, svm_model, lr_model, gmm_model):
#     output_dir = "Old_Output/Output/Evaluation"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Step 1: Compute minimum and actual DCF, and Bayes error plots for the delivered system
#     effPriorLogOdds = np.linspace(-4, 4, 21)
#
#     minDCF_svm_eval, actDCF_svm_eval, svm_scores_eval = single_fold_calibration(DTR, LTR, D_eval, L_eval, svm_model)
#     minDCF_lr_eval, actDCF_lr_eval, lr_scores_eval = single_fold_calibration(DTR, LTR, D_eval, L_eval, lr_model)
#     minDCF_gmm_eval, actDCF_gmm_eval, gmm_scores_eval = single_fold_calibration(DTR, LTR, D_eval, L_eval, gmm_model)
#
#     plot_bayes_error_with_calibration(svm_scores_eval, L_eval, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_SVM_Eval.png'))
#     plot_bayes_error_with_calibration(lr_scores_eval, L_eval, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_LR_Eval.png'))
#     plot_bayes_error_with_calibration(gmm_scores_eval, L_eval, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_GMM_Eval.png'))
#
#     print("Evaluation Results for Individual Models:")
#     print(f'SVM Eval minDCF: {minDCF_svm_eval}, actDCF: {actDCF_svm_eval}')
#     print(f'LR Eval minDCF: {minDCF_lr_eval}, actDCF: {actDCF_lr_eval}')
#     print(f'GMM Eval minDCF: {minDCF_gmm_eval}, actDCF: {actDCF_gmm_eval}')
#
#     # Step 2: Consider the three best performing systems, and their fusion. Evaluate the corresponding actual DCF
#     minDCF_fusion_eval, actDCF_fusion_eval, fused_scores_eval = score_level_fusion(svm_scores_eval, lr_scores_eval,
#                                                                                    L_eval)
#     minDCF_fusion_eval, actDCF_fusion_eval, fused_scores_eval = score_level_fusion(fused_scores_eval, gmm_scores_eval,
#                                                                                    L_eval)
#
#     plot_bayes_error_with_calibration(fused_scores_eval, L_eval, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_Fusion_Eval.png'))
#
#     print(f'Fusion Eval minDCF: {minDCF_fusion_eval}, actDCF: {actDCF_fusion_eval}')
#
#     # Step 3: Evaluate minimum and actual DCF for the target application, and analyze the corresponding Bayes error plots
#     plot_bayes_error_with_calibration(fused_scores_eval, L_eval, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_Target_Application_Eval.png'))
#
#     # Step 4: Analyze whether your training strategy was effective for the selected approach (SVM in this case)
#     C_values = [0.001, 0.01, 0.1, 1, 10]
#     kernel_types = ['linear', 'poly', 'rbf']
#
#     best_min_dcf = minDCF_svm_eval
#     for C in C_values:
#         for kernel in kernel_types:
#             svm_model = SVMClassifier(kernel=kernel, C=C)
#             svm_model.train(DTR, LTR)
#             svm_scores_eval = svm_model.project(D_eval)
#             svm_min_dcf_eval, _ = evaluate_performance(D_eval, L_eval, svm_scores_eval, effPriorLogOdds,
#                                                        f'svm_eval_C{C}_kernel{kernel}')
#             print(f'SVM (C={C}, kernel={kernel}) Eval minDCF: {svm_min_dcf_eval}')
#             if svm_min_dcf_eval < best_min_dcf:
#                 best_min_dcf = svm_min_dcf_eval
#                 print(f'New best SVM model found with C={C} and kernel={kernel} with minDCF: {svm_min_dcf_eval}')
#
#     return
#
#
# def run_project_analysis(DTR, LTR, DTE, LTE):
#     output_dir = "Old_Output/Output/CalibrationFusion"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Define the models without pre-training
#     svm_model = SVMClassifier(kernel='poly', degree=4, C=0.00316)
#     lr_model = LogRegClass(None, None, 0.03162277660168379)
#     gmm_model = GMMClass(covariance_type='full')
#
#     # Generate consistent K-fold splits
#     splits = generate_kfold_splits(DTR, LTR, K=5)
#
#     # Calibration and evaluation using single-fold approach
#     minDCF_svm_single, actDCF_svm_single, calibrated_svm_scores_single = single_fold_calibration(DTR, LTR, DTE, LTE,
#                                                                                                  svm_model)
#     print("SVM Done Single")
#     minDCF_lr_single, actDCF_lr_single, calibrated_lr_scores_single = single_fold_calibration(DTR, LTR, DTE, LTE,
#                                                                                               lr_model)
#     print("LR Done Single")
#     minDCF_gmm_single, actDCF_gmm_single, calibrated_gmm_scores_single = single_fold_calibration(DTR, LTR, DTE, LTE,
#                                                                                                  gmm_model)
#     print("GMM Done Single")
#
#     print("Single done")
#
#     # Calibration and evaluation using K-fold approach
#     minDCF_svm_kfold, actDCF_svm_kfold, calibrated_svm_scores_kfold, all_labels_svm = kfold_calibration(DTR, LTR,
#                                                                                                         svm_model,
#                                                                                                         splits)
#     minDCF_lr_kfold, actDCF_lr_kfold, calibrated_lr_scores_kfold, all_labels_lr = kfold_calibration(DTR, LTR, lr_model,
#                                                                                                     splits)
#     minDCF_gmm_kfold, actDCF_gmm_kfold, calibrated_gmm_scores_kfold, all_labels_gmm = kfold_calibration(DTR, LTR,
#                                                                                                         gmm_model,
#                                                                                                         splits)
#
#     print("Kfold done")
#
#     # Debug prints for label consistency
#     print(f"Labels SVM: {all_labels_svm[:10]} - Labels LR: {all_labels_lr[:10]} - Labels GMM: {all_labels_gmm[:10]}")
#     print(
#         f"Labels SVM shape: {all_labels_svm.shape} - Labels LR shape: {all_labels_lr.shape} - Labels GMM shape: {all_labels_gmm.shape}")
#
#     # Ensure labels are consistent
#     assert np.array_equal(all_labels_svm, all_labels_lr) and np.array_equal(all_labels_lr,
#                                                                             all_labels_gmm), "Mismatch in labels between models"
#     all_labels_kfold = all_labels_svm
#
#     # Plot the results
#     effPriorLogOdds = np.linspace(-4, 4, 21)
#     plot_bayes_error_with_calibration(calibrated_svm_scores_single, LTE, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_SVM_Single.png'))
#     plot_bayes_error_with_calibration(calibrated_lr_scores_single, LTE, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_LR_Single.png'))
#     plot_bayes_error_with_calibration(calibrated_gmm_scores_single, LTE, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_GMM_Single.png'))
#     plot_bayes_error_with_calibration(calibrated_svm_scores_kfold, all_labels_kfold, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_SVM_Kfold.png'))
#     plot_bayes_error_with_calibration(calibrated_lr_scores_kfold, all_labels_kfold, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_LR_Kfold.png'))
#     plot_bayes_error_with_calibration(calibrated_gmm_scores_kfold, all_labels_kfold, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_GMM_Kfold.png'))
#
#     # Score-level fusion
#     minDCF_fusion_single, actDCF_fusion_single, fused_scores_single = score_level_fusion(calibrated_svm_scores_single,
#                                                                                          calibrated_lr_scores_single,
#                                                                                          LTE)
#     minDCF_fusion_kfold, actDCF_fusion_kfold, fused_scores_kfold = score_level_fusion(calibrated_svm_scores_kfold,
#                                                                                       calibrated_lr_scores_kfold,
#                                                                                       all_labels_kfold)
#
#     print("Fusion done")
#
#     # Plot the fusion results
#     plot_bayes_error_with_calibration(fused_scores_single, LTE, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_Fusion_Single.png'))
#     plot_bayes_error_with_calibration(fused_scores_kfold, all_labels_kfold, effPriorLogOdds,
#                                       output_file=os.path.join(output_dir, 'Bayes_Error_Fusion_Kfold.png'))
#
#     # Print results
#     print(f'SVM Single-Fold minDCF: {minDCF_svm_single}, actDCF: {actDCF_svm_single}')
#     print(f'LR Single-Fold minDCF: {minDCF_lr_single}, actDCF: {actDCF_lr_single}')
#     print(f'GMM Single-Fold minDCF: {minDCF_gmm_single}, actDCF: {actDCF_gmm_single}')
#     print(f'SVM K-Fold minDCF: {minDCF_svm_kfold}, actDCF: {actDCF_svm_kfold}')
#     print(f'LR K-Fold minDCF: {minDCF_lr_kfold}, actDCF: {actDCF_lr_kfold}')
#     print(f'GMM K-Fold minDCF: {minDCF_gmm_kfold}, actDCF: {actDCF_gmm_kfold}')
#     print(f'Fusion Single-Fold minDCF: {minDCF_fusion_single}, actDCF: {actDCF_fusion_single}')
#     print(f'Fusion K-Fold minDCF: {minDCF_fusion_kfold}, actDCF: {actDCF_fusion_kfold}')
#
#     # Load evaluation data
#     D_eval, L_eval = load_data('Data/evalData.txt')
#
#     # Evaluate system
#     evaluate_system(DTR, LTR, DTE, LTE, D_eval, L_eval, svm_model, lr_model, gmm_model)


# def single_fold_calibration(D, L, eval_data, eval_labels, system_1_scores, system_2_scores):
#     # Split data
#     n_train = int(0.7 * D.shape[1])
#     D_train, L_train = D[:, :n_train], L[:n_train]
#     D_valid, L_valid = D[:, n_train:], L[n_train:]
#
#     # Calibrate System 1
#     system_1_calibrated = calibrate_scores(system_1_scores[:n_train], L_train)
#
#     # Calibrate System 2
#     system_2_calibrated = calibrate_scores(system_2_scores[:n_train], L_train)
#
#     # Compute DCF for validation
#     system_1_dcf = compute_dcf(system_1_calibrated, L_valid, 0.5)
#     system_2_dcf = compute_dcf(system_2_calibrated, L_valid, 0.5)
#     fusion_scores = 0.5 * (system_1_calibrated + system_2_calibrated)
#     fusion_dcf = compute_dcf(fusion_scores, L_valid, 0.5)
#
#     # Compute DCF for evaluation
#     system_1_eval_dcf = compute_dcf(system_1_scores[n_train:], eval_labels, 0.5)
#     system_2_eval_dcf = compute_dcf(system_2_scores[n_train:], eval_labels, 0.5)
#     fusion_eval_scores = 0.5 * (system_1_scores[n_train:] + system_2_scores[n_train:])
#     fusion_eval_dcf = compute_dcf(fusion_eval_scores, eval_labels, 0.5)
#
#     return system_1_dcf, system_2_dcf, fusion_dcf, system_1_eval_dcf, system_2_eval_dcf, fusion_eval_dcf


# def k_fold_calibration(D, L, eval_data, eval_labels, system_1_scores, system_2_scores, k=5):
#     fold_size = D.shape[1] // k
#     system_1_dcfs = []
#     system_2_dcfs = []
#     fusion_dcfs = []
#
#     for i in range(k):
#         val_start = i * fold_size
#         val_end = (i + 1) * fold_size
#         D_train = np.hstack([D[:, :val_start], D[:, val_end:]])
#         L_train = np.hstack([L[:val_start], L[val_end:]])
#         D_val = D[:, val_start:val_end]
#         L_val = L[val_start:val_end]
#
#         # Calibrate System 1
#         system_1_calibrated = calibrate_scores(system_1_scores[val_start:val_end], L_train)
#
#         # Calibrate System 2
#         system_2_calibrated = calibrate_scores(system_2_scores[val_start:val_end], L_train)
#
#         # Compute DCF
#         system_1_dcf = compute_dcf(system_1_calibrated, L_val, 0.5)
#         system_2_dcf = compute_dcf(system_2_calibrated, L_val, 0.5)
#         fusion_scores = 0.5 * (system_1_calibrated + system_2_calibrated)
#         fusion_dcf = compute_dcf(fusion_scores, L_val, 0.5)
#
#         system_1_dcfs.append(system_1_dcf)
#         system_2_dcfs.append(system_2_dcf)
#         fusion_dcfs.append(fusion_dcf)
#
#     # Compute average DCF for each system and fusion
#     avg_system_1_dcf = np.mean(system_1_dcfs)
#     avg_system_2_dcf = np.mean(system_2_dcfs)
#     avg_fusion_dcf = np.mean(fusion_dcfs)
#
#     # Compute DCF for evaluation
#     system_1_eval_dcf = compute_dcf(system_1_scores, eval_labels, 0.5)
#     system_2_eval_dcf = compute_dcf(system_2_scores, eval_labels, 0.5)
#     fusion_eval_scores = 0.5 * (system_1_scores + system_2_scores)
#     fusion_eval_dcf = compute_dcf(fusion_eval_scores, eval_labels, 0.5)
#
#     return avg_system_1_dcf, avg_system_2_dcf, avg_fusion_dcf, system_1_eval_dcf, system_2_eval_dcf, fusion_eval_dcf


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

    plot_hist(DTR, LTR)

    (DTR, LTR), (DTE, LTE) = split_db_2to1(DTR, LTR)
    ciao = False
    print("ciao")

    # if ciao:
    ############################ DATA ANALYSIS - LAB 2 ############################

    # data_analysis(DTR, LTR)

    ############################ PCA & LDA - LAB 3  ############################

    # PCA_LDA_analysis(DTR, LTR, DTE, LTE)

    ############################ UNIVARIATE GAUSSIAN MODELS - LAB 4 ############################

    fit_univariate_gaussian_models(DTR, LTR)

    ############################ MULTIVARIATE GAUSSIAN MODELS - LAB 5 ############################

    train_Lab_5(DTR, LTR, DTE, LTE)
    # train_MVG_1(DTE, DTR, LTE, LTR)

    ############################ MULTIVARIATE GAUSSIAN MODELS - PLOTS FOR LAB 7 ############################

    # train_MVG(DTE, DTR, LTE, LTR)
    train_Lab_7(DTR, LTR, DTE, LTE)

    ############################ LOGISTIC REGRESSION - LAB 8 ############################

    Logistic_Regression_train(DTE, DTR, LTE, LTR)

    ############################ SVM - LAB 9 ############################

    train_SVM(DTE, DTR, LTE, LTR)

    ############################ GMM - LAB 10 ############################

    train_GMM(DTE, DTR, LTE, LTR)

    # 2nd part
    best_models = evaluate_classifiers(DTR, LTR, DTE, LTE)

    ############################ CALIB - FUSION - EVAL - LAB 11 ############################

    # Evaluate using single-fold approach
    single_fold_results = evaluate_classifiers_single_fold(DTR, LTR, DTE, LTE, best_models)

    # Print single-fold results
    print("Single-Fold Evaluation Results:")
    for result in single_fold_results:
        model_name, params, act_dcf, min_dcf = result
        print(f"Model: {model_name}, Params: {params}")
        print(f"Actual DCF: {act_dcf:.4f}, Minimum DCF: {min_dcf:.4f}\n")

    evaluate_classifiers_kfold(DTR, LTR, DTE, LTE, best_models)


    return


def evaluate_classifiers(DTR, LTR, DTE, LTE):
    print("Evaluating classifiers...")

    # Hyperparameters to evaluate

    # Logistic Regression (including Quadratic)
    logreg_params = [
        {'l': 1e-4, 'quadratic': False},
        {'l': 1e-3, 'quadratic': False},
        {'l': 1e-2, 'quadratic': False},
        {'l': 1e-1, 'quadratic': False},
        {'l': 1e-4, 'quadratic': True},
        {'l': 1e-3, 'quadratic': True},
        {'l': 1e-2, 'quadratic': True},
        {'l': 1e-1, 'quadratic': True},
    ]

    # SVM (linear, polynomial, and RBF kernels)
    svm_params = [
        {'C': 0.1, 'kernel': 'linear'},
        {'C': 1.0, 'kernel': 'linear'},
        {'C': 10.0, 'kernel': 'linear'},
        {'C': 0.1, 'kernel': 'poly', 'degree': 2, 'coef0': 1},
        {'C': 1.0, 'kernel': 'poly', 'degree': 2, 'coef0': 1},
        {'C': 10.0, 'kernel': 'poly', 'degree': 2, 'coef0': 1},
        {'C': 0.1, 'kernel': 'rbf', 'gamma': 1e-2},
        {'C': 1.0, 'kernel': 'rbf', 'gamma': 1e-2},
        {'C': 10.0, 'kernel': 'rbf', 'gamma': 1e-2},
        {'C': 0.1, 'kernel': 'rbf', 'gamma': 1e-3},
        {'C': 1.0, 'kernel': 'rbf', 'gamma': 1e-3},
        {'C': 10.0, 'kernel': 'rbf', 'gamma': 1e-3},
    ]

    # GMM (various number of components and covariance types)
    gmm_params = [
        {'num_components': 1, 'cov_type': 'full'},
        {'num_components': 2, 'cov_type': 'full'},
        {'num_components': 4, 'cov_type': 'full'},
        {'num_components': 8, 'cov_type': 'full'},
        {'num_components': 16, 'cov_type': 'full'},
        {'num_components': 32, 'cov_type': 'full'},
        {'num_components': 1, 'cov_type': 'diagonal'},
        {'num_components': 2, 'cov_type': 'diagonal'},
        {'num_components': 4, 'cov_type': 'diagonal'},
        {'num_components': 8, 'cov_type': 'diagonal'},
        {'num_components': 16, 'cov_type': 'diagonal'},
        {'num_components': 32, 'cov_type': 'diagonal'},
        {'num_components': 1, 'cov_type': 'tied'},
        {'num_components': 2, 'cov_type': 'tied'},
        {'num_components': 4, 'cov_type': 'tied'},
        {'num_components': 8, 'cov_type': 'tied'},
        {'num_components': 16, 'cov_type': 'tied'},
        {'num_components': 32, 'cov_type': 'tied'},
    ]

    best_models = {}
    all_dcf_results = []

    # Quadratic logistic regression
    DTR_expanded = expand_features_quadratic(DTR)  # Expand features to quadratic
    DTE_expanded = expand_features_quadratic(DTE)  # Expand features to quadratic

    # Logistic Regression Evaluation
    min_dcf_lr = float('inf')
    best_logreg_model = None
    for params in logreg_params:
        if params.get('quadratic', False):
            log_reg_model = LogRegClass(DTR_expanded, LTR, params['l'])
        else:
            log_reg_model = LogRegClass(DTR, LTR, params['l'])

        log_reg_model.train()

        if params.get('quadratic', False):
            scores = log_reg_model.predict(DTE_expanded)
        else:
            scores = log_reg_model.predict(DTE)

        act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, 0.5, 1, 1)
        min_dcf = compute_minDCF_binary_fast(scores, LTE, 0.5, 1, 1)  # Do not unpack, use directly

        all_dcf_results.append(('Logistic Regression', params, act_dcf, min_dcf))
        if min_dcf < min_dcf_lr:
            min_dcf_lr = min_dcf
            best_logreg_model = (log_reg_model, params)

    best_models['Logistic Regression'] = best_logreg_model

    # SVM Evaluation
    min_dcf_svm = float('inf')
    best_svm_model = None
    for params in svm_params:
        svm_model = SVMClassifier(**params)
        svm_model.train(DTR, LTR)
        scores = svm_model.predict(DTE)
        act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, 0.5, 1, 1)
        min_dcf = compute_minDCF_binary_fast(scores, LTE, 0.5, 1, 1)
        all_dcf_results.append(('SVM', params, act_dcf, min_dcf))
        if min_dcf < min_dcf_svm:
            min_dcf_svm = min_dcf
            best_svm_model = (svm_model, params)
    best_models['SVM'] = best_svm_model

    # GMM Evaluation
    min_dcf_gmm = float('inf')
    best_gmm_model = None
    for params in gmm_params:
        gmm_model = GMMClass(**params)
        gmm_model.train(DTR)
        scores = gmm_model.predict(DTE)
        act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, 0.5, 1, 1)
        min_dcf = compute_minDCF_binary_fast(scores, LTE, 0.5, 1, 1)
        all_dcf_results.append(('GMM', params, act_dcf, min_dcf))
        if min_dcf < min_dcf_gmm:
            min_dcf_gmm = min_dcf
            best_gmm_model = (gmm_model, params)
    best_models['GMM'] = best_gmm_model

    # Print out the DCFs for all models
    for model_name, params, act_dcf, min_dcf in all_dcf_results:
        print(f"Model: {model_name}, Params: {params}")
        print(f"Actual DCF: {act_dcf:.4f}, Minimum DCF: {min_dcf:.4f}\n")

    # Save DCF plots
    output_dir = 'Output/Comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Bayes error plots
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Best models
    print("Best three models:")

    # Plot for each model separately
    for model_name, (model, params) in best_models.items():
        print(f"Model: {model_name}, Params: {params}")
        actDCFs = []
        minDCFs = []
        if 'quadratic' in params and params['quadratic']:
            scores = model.predict(DTE_expanded)
        else:
            scores = model.predict(DTE)
        for effPrior in effPriors:
            act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, effPrior, 1.0, 1.0)
            min_dcf = compute_minDCF_binary_fast(scores, LTE, effPrior, 1.0, 1.0)
            actDCFs.append(act_dcf)
            minDCFs.append(min_dcf)
        plot_bayes_error(effPriors, actDCFs, minDCFs, os.path.join(output_dir, f"{model_name}_bayes_error_plot"))

    # Plot for all best models together
    plt.figure(figsize=(10, 6))
    for model_name, (model, params) in best_models.items():
        actDCFs = []
        minDCFs = []
        if 'quadratic' in params and params['quadratic']:
            scores = model.predict(DTE_expanded)
        else:
            scores = model.predict(DTE)
        for effPrior in effPriors:
            act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, effPrior, 1.0, 1.0)
            min_dcf = compute_minDCF_binary_fast(scores, LTE, effPrior, 1.0, 1.0)
            actDCFs.append(act_dcf)
            minDCFs.append(min_dcf)
        plt.plot(effPriorLogOdds, actDCFs, label=f'{model_name} actDCF')
        plt.plot(effPriorLogOdds, minDCFs, '--', label=f'{model_name} minDCF')

    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title('Comparison of Best Models - Bayes Error Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "best_models_comparison_plot.png"))
    plt.close()

    return best_models


# Utility Functions
def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def evaluate_classifiers_kfold(DTR, LTR, DTE, LTE, best_models, pT=0.5, K=5):
    DTE_expanded = expand_features_quadratic(DTE)  # Expand features for quadratic models if not already done

    output_dir = "Output/Calibration"
    os.makedirs(output_dir, exist_ok=True)

    print("K-Fold Evaluation Results:")

    # Apply K-fold calibration and fusion
    calibrated_scores_all = {}
    kfold_results = []

    for model_name, (model, params) in best_models.items():
        if 'quadratic' in params and params['quadratic']:
            scores = model.predict(DTE_expanded)
        else:
            scores = model.predict(DTE)

        calibrated_scores, all_labels = k_fold_calibration_logreg(scores, LTE, pT, K)
        calibrated_scores_all[model_name] = calibrated_scores

        # Calculate DCFs after calibration
        act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(calibrated_scores, all_labels, pT, 1, 1)
        min_dcf = compute_minDCF_binary_fast(calibrated_scores, all_labels, pT, 1, 1)

        # Store results for printing
        kfold_results.append((model_name, params, act_dcf, min_dcf))

    # Print K-fold results
    for result in kfold_results:
        model_name, params, act_dcf, min_dcf = result
        print(f"Model: {model_name}, Params: {params}")
        print(
            f"Actual DCF after K-fold calibration: {act_dcf:.4f}, Minimum DCF after K-fold calibration: {min_dcf:.4f}\n")

    # Fusion of calibrated scores
    fused_scores = score_fusion_logreg(list(calibrated_scores_all.values()), LTE, pT)

    # Post-fusion Bayes error plot
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCFs_fusion, minDCFs_fusion = bayesPlot(fused_scores, LTE, effPriorLogOdds, effPriors)
    plt.plot(effPriorLogOdds, actDCFs_fusion, label='Fusion - actDCF (cal)')
    plt.plot(effPriorLogOdds, minDCFs_fusion, '--', label='Fusion - minDCF')
    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title('Fusion Bayes Error Plot')
    plt.legend()
    plt.savefig(f"{output_dir}/fusion_bayes_error_plot.png")


def calibrate_scores_logreg(scores, labels, pT):
    logreg = LogRegClass(vrow(scores), labels, l=0.0, weighted=True, pT=pT)
    logreg.train()
    calibrated_scores = logreg.predict(vrow(scores)) - np.log(pT / (1 - pT))
    return calibrated_scores


def k_fold_calibration_logreg(scores, labels, pT, K=5):
    fold_size = len(scores) // K
    calibrated_scores = []
    all_labels = []

    for i in range(K):
        val_indices = range(i * fold_size, (i + 1) * fold_size)
        train_indices = list(set(range(len(scores))) - set(val_indices))

        SCAL = scores[train_indices]
        LCAL = labels[train_indices]
        SVAL = scores[val_indices]
        LVAL = labels[val_indices]

        logreg = LogRegClass(vrow(SCAL), LCAL, l=0.0, weighted=True, pT=pT)
        logreg.train()
        calibrated_SVAL = logreg.predict(vrow(SVAL))

        calibrated_scores.append(calibrated_SVAL)
        all_labels.append(LVAL)

    calibrated_scores = np.hstack(calibrated_scores)
    all_labels = np.hstack(all_labels)

    return calibrated_scores, all_labels


# Score Fusion with Logistic Regression
def score_fusion_logreg(scores_list, labels, pT):
    SMatrix = np.vstack(scores_list)
    logreg = LogRegClass(SMatrix, labels, l=0.0, weighted=True, pT=pT)
    logreg.train()
    fused_scores = logreg.predict(SMatrix)
    return fused_scores


def bayesPlot(scores, labels, effPriorLogOdds, effPriors):
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, labels, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(scores, labels, effPrior, 1.0, 1.0))
    return actDCF, minDCF


def plot_bayes_error_plots(models, DTE, LTE, pT, DTE_expanded=None, output_dir="Output/Calibration"):
    os.makedirs(output_dir, exist_ok=True)
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    for model_name, (model, params) in models.items():
        print(f"Model: {model_name}, Params: {params}")

        if 'quadratic' in params and params['quadratic']:
            scores = model.predict(DTE_expanded)
        else:
            scores = model.predict(DTE)

        # Before Calibration
        actDCFs, minDCFs = bayesPlot(scores, LTE, effPriorLogOdds, effPriors)
        plt.plot(effPriorLogOdds, actDCFs, label=f'{model_name} - actDCF')
        plt.plot(effPriorLogOdds, minDCFs, '--', label=f'{model_name} - minDCF')

        # After Calibration
        calibrated_scores = calibrate_scores_logreg(scores, LTE, pT)
        actDCFs_cal, _ = bayesPlot(calibrated_scores, LTE, effPriorLogOdds, effPriors)
        plt.plot(effPriorLogOdds, actDCFs_cal, ':', label=f'{model_name} - actDCF (cal)')

    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title('Bayes Error Plot - Pre and Post Calibration')
    plt.legend()
    plt.savefig(f"{output_dir}/bayes_error_plot.png")


def evaluate_single_fold(DTR, LTR, DTE, LTE, model, params, pT=0.5):
    if 'quadratic' in params and params['quadratic']:
        DTR_expanded = expand_features_quadratic(DTR)
        DTE_expanded = expand_features_quadratic(DTE)
        scores = model.predict(DTE_expanded)
    else:
        scores = model.predict(DTE)

    act_dcf = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores, LTE, pT, 1, 1)
    min_dcf = compute_minDCF_binary_fast(scores, LTE, pT, 1, 1)

    print(f"Model: {type(model).__name__}, Params: {params}")
    print(f"Actual DCF: {act_dcf:.4f}, Minimum DCF: {min_dcf:.4f}\n")

    return scores, act_dcf, min_dcf


def evaluate_classifiers_single_fold(DTR, LTR, DTE, LTE, best_models, pT=0.5):
    results = []
    for model_name, (model, params) in best_models.items():
        scores, act_dcf, min_dcf = evaluate_single_fold(DTR, LTR, DTE, LTE, model, params, pT)
        results.append((model_name, params, act_dcf, min_dcf))
    return results

# 2 Logistic Regression:
#     Model: Quadratic Logistic Regression
#     Configuration: Lambda of 0.03162277660168379
#     Performance: minDCF of 0.024363159242191502, actualDCF of 0.024363159242191502
# 3 GMM:
#     Model: GMM
#     Configuration: Full covariance with 8 components
#     Performance: minDCF of 0.9979038658474142, actualDCF of 1.0
# '''
#
# ############################ CALIBRATION - LAB 11 ###########################
#
# run_project_analysis(DTR, LTR, DTE, LTE)
#
# # eval(DTE, DTR, LTE, LTR)
#
# return


# def eval(DTE, DTR, LTE, LTR):
#     D_train, L_train = DTR, LTR
#     D_valid, L_valid = DTE, LTE
#     D_eval, L_eval = load_data('Data/evalData.txt')
#     # Train classifiers
#     gmm_model = GMMClass()
#     lr_model = LogRegClass(D_train, L_train, l=1.0)
#     svm_model = SVMClassifier(kernel='linear')
#     # Train GMM TODO: adapt to the changes in the GMM of this morning
#     # gmm_model.LBG(D_train, max_components=8)
#     # gmm_model.EM(D_train)
#     # Train Logistic Regression
#     lr_model.train()
#     # Train SVM
#     svm_model.train(D_train, L_train)
#     # Get scores for validation data
#     gmm_scores_valid = gmm_model.logpdf_GMM(D_valid)
#     lr_scores_valid = lr_model.predict(D_valid)
#     svm_scores_valid = svm_model.project(D_valid)
#     # Calibrate scores
#     gmm_calibrated_scores = calibrate_scores(gmm_scores_valid, L_valid)
#     lr_calibrated_scores = calibrate_scores(lr_scores_valid, L_valid)
#     svm_calibrated_scores = calibrate_scores(svm_scores_valid, L_valid)
#     # Fuse scores
#     fused_scores_valid = fuse_scores([gmm_calibrated_scores, lr_calibrated_scores, svm_calibrated_scores])
#     # Evaluate on validation data
#     pi_values = np.linspace(-3, 3, 21)
#     gmm_min_dcf, gmm_act_dcf = evaluate_performance(D_valid, L_valid, gmm_calibrated_scores, pi_values, "gmm_valid")
#     lr_min_dcf, lr_act_dcf = evaluate_performance(D_valid, L_valid, lr_calibrated_scores, pi_values, "lr_valid")
#     svm_min_dcf, svm_act_dcf = evaluate_performance(D_valid, L_valid, svm_calibrated_scores, pi_values, "svm_valid")
#     fusion_min_dcf, fusion_act_dcf = evaluate_performance(D_valid, L_valid, fused_scores_valid, pi_values,
#                                                           "fusion_valid")
#     # Print results for validation data
#     print("Validation Results:")
#     print(f"GMM: minDCF={gmm_min_dcf}, actDCF={gmm_act_dcf}")
#     print(f"LR: minDCF={lr_min_dcf}, actDCF={lr_act_dcf}")
#     print(f"SVM: minDCF={svm_min_dcf}, actDCF={svm_act_dcf}")
#     print(f"Fusion: minDCF={fusion_min_dcf}, actDCF={fusion_act_dcf}")
#     # Get scores for evaluation data
#     gmm_scores_eval = gmm_model.logpdf_GMM(D_eval)
#     lr_scores_eval = lr_model.predict(D_eval)
#     svm_scores_eval = svm_model.project(D_eval)
#     # Calibrate scores using evaluation labels
#     gmm_calibrated_scores_eval = calibrate_scores(gmm_scores_eval, L_eval)
#     lr_calibrated_scores_eval = calibrate_scores(lr_scores_eval, L_eval)
#     svm_calibrated_scores_eval = calibrate_scores(svm_scores_eval, L_eval)
#     # Fuse scores
#     fused_scores_eval = fuse_scores([gmm_calibrated_scores_eval, lr_calibrated_scores_eval, svm_calibrated_scores_eval])
#     # Evaluate on evaluation data
#     gmm_min_dcf_eval, gmm_act_dcf_eval = evaluate_performance(D_eval, L_eval, gmm_calibrated_scores_eval, pi_values,
#                                                               "gmm_eval")
#     lr_min_dcf_eval, lr_act_dcf_eval = evaluate_performance(D_eval, L_eval, lr_calibrated_scores_eval, pi_values,
#                                                             "lr_eval")
#     svm_min_dcf_eval, svm_act_dcf_eval = evaluate_performance(D_eval, L_eval, svm_calibrated_scores_eval, pi_values,
#                                                               "svm_eval")
#     fusion_min_dcf_eval, fusion_act_dcf_eval = evaluate_performance(D_eval, L_eval, fused_scores_eval, pi_values,
#                                                                     "fusion_eval")
#     # Print results for evaluation data
#     print("Evaluation Results:")
#     print(f"GMM: minDCF={gmm_min_dcf_eval}, actDCF={gmm_act_dcf_eval}")
#     print(f"LR: minDCF={lr_min_dcf_eval}, actDCF={lr_act_dcf_eval}")
#     print(f"SVM: minDCF={svm_min_dcf_eval}, actDCF={svm_act_dcf_eval}")
#     print(f"Fusion: minDCF={fusion_min_dcf_eval}, actDCF={fusion_act_dcf_eval}")
#
#
# def Univariate_model(DTE, DTR, LTE, LTR):
#     mvg_classifier = GaussianClassifier()
#     mvg_classifier.fit_univariate_gaussian_models(DTR, LTR)


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
#
