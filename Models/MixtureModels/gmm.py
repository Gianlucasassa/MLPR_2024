import os

import numpy
import numpy as np
import scipy
import scipy.special

import sklearn.datasets
from matplotlib import pyplot as plt

from Models.bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast, \
    compute_empirical_Bayes_risk_binary_llr_optimal_decisions


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


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


def logpdf_GAU_ND(x, mu, C):  # Fast version from Lab 4
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


class GMMClass:
    def __init__(self, num_components=1, cov_type='full', psi=0.01, alpha=0.1, epsilon=1e-6):
        self.num_components = num_components
        self.cov_type = cov_type
        self.psi = psi
        self.alpha = alpha
        self.epsilon = epsilon
        self.gmm = None

    def logpdf_GMM(self, X, gmm):
        S = []
        for w, mu, C in gmm:
            logpdf_conditional = logpdf_GAU_ND(X, mu, C)
            logpdf_joint = logpdf_conditional + np.log(w)
            S.append(logpdf_joint)

        S = np.vstack(S)
        logdens = scipy.special.logsumexp(S, axis=0)
        return logdens

    def smooth_covariance_matrix(self, C, psi):
        U, s, Vh = np.linalg.svd(C)
        s[s < psi] = psi
        return U @ (vcol(s) * U.T)

    def train_GMM_EM_Iteration(self, X, gmm):
        S = []
        for w, mu, C in gmm:
            logpdf_conditional = logpdf_GAU_ND(X, mu, C)
            logpdf_joint = logpdf_conditional + np.log(w)
            S.append(logpdf_joint)

        S = np.vstack(S)
        logdens = scipy.special.logsumexp(S, axis=0)
        gammaAllComponents = np.exp(S - logdens)

        gmmUpd = []
        for gIdx in range(len(gmm)):
            gamma = gammaAllComponents[gIdx]
            Z = gamma.sum()
            F = vcol((vrow(gamma) * X).sum(1))
            S = (vrow(gamma) * X) @ X.T
            muUpd = F / Z
            CUpd = S / Z - muUpd @ muUpd.T
            wUpd = Z / X.shape[1]
            if self.cov_type == 'diagonal':
                CUpd = CUpd * np.eye(X.shape[0])
            gmmUpd.append((wUpd, muUpd, CUpd))

        if self.cov_type == 'tied':
            CTied = sum(w * C for w, mu, C in gmmUpd)
            gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

        if self.psi is not None:
            gmmUpd = [(w, mu, self.smooth_covariance_matrix(C, self.psi)) for w, mu, C in gmmUpd]

        return gmmUpd

    def train_GMM_EM(self, X, gmm):
        llOld = self.logpdf_GMM(X, gmm).mean()
        llDelta = None
        it = 1
        while (llDelta is None or llDelta > self.epsilon):
            gmmUpd = self.train_GMM_EM_Iteration(X, gmm)
            llUpd = self.logpdf_GMM(X, gmmUpd).mean()
            llDelta = llUpd - llOld
            gmm = gmmUpd
            llOld = llUpd
            it += 1
        return gmm

    def split_GMM_LBG(self, gmm):
        gmmOut = []
        for (w, mu, C) in gmm:
            U, s, Vh = np.linalg.svd(C)
            d = U[:, 0:1] * s[0] ** 0.5 * self.alpha
            gmmOut.append((0.5 * w, mu - d, C))
            gmmOut.append((0.5 * w, mu + d, C))
        return gmmOut

    def train(self, X):
        mu, C = compute_mu_C(X)
        if self.cov_type == 'diagonal':
            C = C * np.eye(X.shape[0])

        if self.psi is not None:
            gmm = [(1.0, mu, self.smooth_covariance_matrix(C, self.psi))]
        else:
            gmm = [(1.0, mu, C)]

        while len(gmm) < self.num_components:
            gmm = self.split_GMM_LBG(gmm)
            gmm = self.train_GMM_EM(X, gmm)
        self.gmm = gmm

    def predict(self, X):
        return self.logpdf_GMM(X, self.gmm)


def evaluateIRIS_GMM():
    print('IRIS')
    D, L = load_iris()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for cov_type in ['full', 'diagonal', 'tied']:
        for num_components in [1, 2, 4, 8, 16]:
            gmm_0 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)
            gmm_1 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)
            gmm_2 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)

            gmm_0.train(DTR[:, LTR == 0])
            gmm_1.train(DTR[:, LTR == 1])
            gmm_2.train(DTR[:, LTR == 2])

            SVAL = np.vstack([
                gmm_0.predict(DVAL),
                gmm_1.predict(DVAL),
                gmm_2.predict(DVAL)
            ])

            SVAL += vcol(np.log(np.ones(3) / 3))
            PVAL = SVAL.argmax(0)

            print(
                f'Cov Type: {cov_type.ljust(10)} - {num_components} Gau - Error rate: {(LVAL != PVAL).sum() / LVAL.size * 100:.1f}%')

    # Evaluate on binary dataset
    print()
    print('Binary task')
    data_path = os.path.join(os.path.dirname(__file__), 'Data', 'ext_data_binary.npy')
    labels_path = os.path.join(os.path.dirname(__file__), 'Data', 'ext_data_binary_labels.npy')
    D, L = np.load(data_path), np.load(labels_path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for cov_type in ['full', 'diagonal', 'tied']:
        print(cov_type)
        for num_components in [1, 2, 4, 8, 16]:
            gmm_0 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)
            gmm_1 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)

            gmm_0.train(DTR[:, LTR == 0])
            gmm_1.train(DTR[:, LTR == 1])

            SLLR = gmm_1.predict(DVAL) - gmm_0.predict(DVAL)
            minDCF = compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)
            actDCF = compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)

            print(f'\tnumC = {num_components}: minDCF: {minDCF:.4f} / actDCF: {actDCF:.4f}')
        print()


def evaluateGMM(DVAL, DTR, LVAL, LTR):
    output_dir = 'Output/GMM'
    os.makedirs(output_dir, exist_ok=True)

    num_components_list = [1, 2, 4, 8, 16, 32]
    results = {'full': {'minDCF': [], 'actDCF': []},
               'diagonal': {'minDCF': [], 'actDCF': []},
               'tied': {'minDCF': [], 'actDCF': []}}

    for cov_type in ['full', 'diagonal', 'tied']:
        print(f'Covariance Type: {cov_type}')
        minDCF_list = []
        actDCF_list = []
        for num_components in num_components_list:
            gmm_0 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)
            gmm_1 = GMMClass(num_components=num_components, cov_type=cov_type, psi=0.01)

            gmm_0.train(DTR[:, LTR == 0])
            gmm_1.train(DTR[:, LTR == 1])

            SLLR = gmm_1.predict(DVAL) - gmm_0.predict(DVAL)

            minDCF = compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)
            actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SLLR, LVAL, 0.5, 1.0, 1.0)

            minDCF_list.append(minDCF)
            actDCF_list.append(actDCF)

            print(f'\tnumC = {num_components}: minDCF: {minDCF:.4f} / actDCF: {actDCF:.4f}')

        results[cov_type]['minDCF'] = minDCF_list
        results[cov_type]['actDCF'] = actDCF_list
        print()

    # Plotting for each covariance type separately
    for cov_type in ['full', 'diagonal', 'tied']:
        plt.figure()
        plt.plot(num_components_list, results[cov_type]['minDCF'], label='minDCF', marker='o')
        plt.plot(num_components_list, results[cov_type]['actDCF'], label='actDCF', marker='o')
        plt.xscale('log')
        plt.xlabel('Number of Components')
        plt.ylabel('DCF')
        plt.title(f'DCF vs Number of Components ({cov_type.capitalize()} Covariance)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'dcf_vs_num_components_{cov_type}.png'))
        plt.close()

    # Plotting all together
    plt.figure()
    for cov_type in ['full', 'diagonal', 'tied']:
        plt.plot(num_components_list, results[cov_type]['minDCF'], label=f'{cov_type.capitalize()} minDCF', marker='o')
        plt.plot(num_components_list, results[cov_type]['actDCF'], label=f'{cov_type.capitalize()} actDCF', marker='o')

    plt.xscale('log')
    plt.xlabel('Number of Components')
    plt.ylabel('DCF')
    plt.title('DCF vs Number of Components (All Covariance Types)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'dcf_vs_num_components_all.png'))
    plt.close()


def train_GMM(DTE, DTR, LTE, LTR):
    #Check correctness on iris dataset
    evaluateIRIS_GMM()

    #Actual training
    evaluateGMM(DTE, DTR, LTE, LTR)


    return

#
# def vcol(x):
#     return x.reshape((x.size, 1))
#
# def vrow(x):
#     return x.reshape((1, x.size))
#
# def logpdf_GAU_ND(x, mu, C):
#     P = np.linalg.inv(C)
#     mu = vcol(mu)  # Ensure mu is a column vector for broadcasting
#     x_centered = x - mu
#     return -0.5 * x.shape[0] * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * np.sum(x_centered * (P @ x_centered), axis=0)
#
# class GMMClass:
#     def __init__(self, gmm_init=None, covariance_type='full'):
#         self.gmm = gmm_init if gmm_init is not None else []
#         self.covariance_type = covariance_type
#
#     def logpdf_GAU_ND(self, X, mu, C):
#         return logpdf_GAU_ND(X, mu, C)
#
#     def logpdf_GMM(self, X):
#         S = np.array([self.logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in self.gmm])
#         logdens = scipy.special.logsumexp(S, axis=0)
#         return logdens
#
#     def smooth_covariance_matrix(self, C, psi):
#         U, s, Vh = np.linalg.svd(C)
#         s[s < psi] = psi
#         C_upd = U @ np.diag(s) @ Vh
#         return C_upd
#
#     def train_GMM_EM_Iteration(self, X, psiEig=None):
#         S = []
#         for w, mu, C in self.gmm:
#             logpdf_conditional = self.logpdf_GAU_ND(X, mu, C)
#             logpdf_joint = logpdf_conditional + np.log(w)
#             S.append(logpdf_joint)
#
#         S = np.vstack(S)
#         logdens = scipy.special.logsumexp(S, axis=0)
#         gammaAllComponents = np.exp(S - logdens)
#
#         gmmUpd = []
#         for gIdx in range(len(self.gmm)):
#             gamma = gammaAllComponents[gIdx]
#             Z = gamma.sum()
#             F = vcol((vrow(gamma) * X).sum(1))
#             S = (vrow(gamma) * X) @ X.T
#             muUpd = F / Z
#             CUpd = S / Z - muUpd @ muUpd.T
#             wUpd = Z / X.shape[1]
#             if self.covariance_type == 'diag':
#                 CUpd = np.diag(np.diag(CUpd))
#             gmmUpd.append((wUpd, muUpd, CUpd))
#
#         if self.covariance_type == 'tied':
#             CTied = sum(w * C for w, mu, C in gmmUpd)
#             gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]
#
#         if psiEig is not None:
#             gmmUpd = [(w, mu, self.smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
#
#         return gmmUpd
#
#     def train_GMM_EM(self, X, psiEig=None, epsLLAverage=1e-6, verbose=True):
#         llOld = self.logpdf_GMM(X).mean()
#         llDelta = None
#         if verbose:
#             print(f'GMM - it {0:3d} - average ll {llOld:.8e}')
#         it = 1
#         while llDelta is None or llDelta > epsLLAverage:
#             gmmUpd = self.train_GMM_EM_Iteration(X, psiEig=psiEig)
#             llUpd = self.logpdf_GMM(X).mean()
#             llDelta = llUpd - llOld
#             if verbose:
#                 print(f'GMM - it {it:3d} - average ll {llUpd:.8e}')
#             self.gmm = gmmUpd
#             llOld = llUpd
#             it += 1
#         if verbose:
#             print(f'GMM - it {it:3d} - average ll {llUpd:.8e} (eps = {epsLLAverage:.1e})')
#         return self.gmm
#
#     def split_GMM_LBG(self, alpha=0.1, verbose=True):
#         gmmOut = []
#         if verbose:
#             print(f'LBG - going from {len(self.gmm)} to {len(self.gmm) * 2} components')
#         for w, mu, C in self.gmm:
#             U, s, Vh = np.linalg.svd(C)
#             d = U[:, 0:1] * np.sqrt(s[0]) * alpha
#             gmmOut.append((0.5 * w, vcol(mu - d), C))
#             gmmOut.append((0.5 * w, vcol(mu + d), C))
#         return gmmOut
#
#     def train_GMM_LBG_EM(self, X, numComponents, psiEig=None, epsLLAverage=1e-6, alpha=0.1, verbose=True):
#         mu, C = np.mean(X, axis=1), np.cov(X)
#         mu = vcol(mu)
#         if self.covariance_type == 'diag':
#             C = np.diag(np.diag(C))  # Initial diagonal GMM to train a diagonal GMM
#
#         if psiEig is not None:
#             self.gmm = [(1.0, mu, self.smooth_covariance_matrix(C, psiEig))]
#         else:
#             self.gmm = [(1.0, mu, C)]
#
#         while len(self.gmm) < numComponents:
#             if verbose:
#                 print(f'Average ll before LBG: {self.logpdf_GMM(X).mean():.8e}')
#             self.gmm = self.split_GMM_LBG(alpha, verbose=verbose)
#             if verbose:
#                 print(f'Average ll after LBG: {self.logpdf_GMM(X).mean():.8e}')
#             self.gmm = self.train_GMM_EM(X, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
#         return self.gmm
#
#     def predict(self, X):
#         return self.logpdf_GMM(X)
#
# def train_GMM(DTE, DTR, LTE, LTR):
#     n_components_list = [1, 2, 4, 8, 16, 32]
#     covariance_types = ['full', 'diagonal', 'tied']
#     psiEig = 0.01
#     alpha = 0.1
#     epsLLAverage = 1e-6
#     prior = 0.5
#     Cfn = 1
#     Cfp = 1
#
#     results = {}
#
#     for cov_type in covariance_types:
#         results[cov_type] = {}
#         for n_components in n_components_list:
#             model = GMMClass(covariance_type=cov_type)
#             model.train_GMM_LBG_EM(DTR, numComponents=n_components, psiEig=psiEig, alpha=alpha,
#                                    epsLLAverage=epsLLAverage, verbose=True)
#             scores = model.logpdf_GMM(DTE)
#             min_dcf = compute_minDCF_binary_fast(scores, LTE, prior, Cfn, Cfp)
#             actual_dcf = compute_actDCF_binary_fast(scores, LTE, prior, Cfn, Cfp)
#             results[cov_type][n_components] = (min_dcf, actual_dcf)
#             print(f'{cov_type} covariance with {n_components} components: minDCF={min_dcf}, actualDCF={actual_dcf}')
#     return results
#
#
#
#
#
#
#
# # Functions for computing DCF
# def compute_confusion_matrix(predictedLabels, classLabels):
#     nClasses = classLabels.max() + 1
#     M = np.zeros((nClasses, nClasses), dtype=np.int32)
#     for i in range(classLabels.size):
#         M[predictedLabels[i], classLabels[i]] += 1
#     return M
#
# def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
#     th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
#     return np.int32(llr > th)
#
# def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
#     M = compute_confusion_matrix(predictedLabels, classLabels)  # Confusion matrix
#     Pfn = M[0, 1] / (M[0, 1] + M[1, 1])
#     Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
#     bayesError = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp
#     if normalize:
#         return bayesError / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
#     return bayesError
#
# # Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
# def compute_actDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp,
#                                                               normalize=True):
#     predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
#     return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp,
#                                                normalize=normalize)
#
#
# def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
#     llrSorter = numpy.argsort(llr)
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
#     # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
#     # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
#     # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
#     llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])
#
#     # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
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
#     return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(
#         thresholdsOut)  # we return also the corresponding thresholds
#
# def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
#
#     Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
#     minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (
#                 1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
#     idx = numpy.argmin(minDCF)
#     if returnThreshold:
#         return minDCF[idx], th[idx]
#     else:
#         return minDCF[idx]
#
#
