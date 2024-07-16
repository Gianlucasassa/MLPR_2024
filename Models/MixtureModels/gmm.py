import numpy
import numpy as np
import scipy
import scipy.special


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    mu = vcol(mu)  # Ensure mu is a column vector for broadcasting
    x_centered = x - mu
    return -0.5 * x.shape[0] * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * np.sum(x_centered * (P @ x_centered), axis=0)

class GMMClass:
    def __init__(self, gmm_init=None, covariance_type='full'):
        self.gmm = gmm_init if gmm_init is not None else []
        self.covariance_type = covariance_type

    def logpdf_GAU_ND(self, X, mu, C):
        return logpdf_GAU_ND(X, mu, C)

    def logpdf_GMM(self, X):
        S = np.array([self.logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in self.gmm])
        logdens = scipy.special.logsumexp(S, axis=0)
        return logdens

    def smooth_covariance_matrix(self, C, psi):
        U, s, Vh = np.linalg.svd(C)
        s[s < psi] = psi
        C_upd = U @ np.diag(s) @ Vh
        return C_upd

    def train_GMM_EM_Iteration(self, X, psiEig=None):
        S = []
        for w, mu, C in self.gmm:
            logpdf_conditional = self.logpdf_GAU_ND(X, mu, C)
            logpdf_joint = logpdf_conditional + np.log(w)
            S.append(logpdf_joint)

        S = np.vstack(S)
        logdens = scipy.special.logsumexp(S, axis=0)
        gammaAllComponents = np.exp(S - logdens)

        gmmUpd = []
        for gIdx in range(len(self.gmm)):
            gamma = gammaAllComponents[gIdx]
            Z = gamma.sum()
            F = vcol((vrow(gamma) * X).sum(1))
            S = (vrow(gamma) * X) @ X.T
            muUpd = F / Z
            CUpd = S / Z - muUpd @ muUpd.T
            wUpd = Z / X.shape[1]
            if self.covariance_type == 'diag':
                CUpd = np.diag(np.diag(CUpd))
            gmmUpd.append((wUpd, muUpd, CUpd))

        if self.covariance_type == 'tied':
            CTied = sum(w * C for w, mu, C in gmmUpd)
            gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

        if psiEig is not None:
            gmmUpd = [(w, mu, self.smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]

        return gmmUpd

    def train_GMM_EM(self, X, psiEig=None, epsLLAverage=1e-6, verbose=True):
        llOld = self.logpdf_GMM(X).mean()
        llDelta = None
        if verbose:
            print(f'GMM - it {0:3d} - average ll {llOld:.8e}')
        it = 1
        while llDelta is None or llDelta > epsLLAverage:
            gmmUpd = self.train_GMM_EM_Iteration(X, psiEig=psiEig)
            llUpd = self.logpdf_GMM(X).mean()
            llDelta = llUpd - llOld
            if verbose:
                print(f'GMM - it {it:3d} - average ll {llUpd:.8e}')
            self.gmm = gmmUpd
            llOld = llUpd
            it += 1
        if verbose:
            print(f'GMM - it {it:3d} - average ll {llUpd:.8e} (eps = {epsLLAverage:.1e})')
        return self.gmm

    def split_GMM_LBG(self, alpha=0.1, verbose=True):
        gmmOut = []
        if verbose:
            print(f'LBG - going from {len(self.gmm)} to {len(self.gmm) * 2} components')
        for w, mu, C in self.gmm:
            U, s, Vh = np.linalg.svd(C)
            d = U[:, 0:1] * np.sqrt(s[0]) * alpha
            gmmOut.append((0.5 * w, vcol(mu - d), C))
            gmmOut.append((0.5 * w, vcol(mu + d), C))
        return gmmOut

    def train_GMM_LBG_EM(self, X, numComponents, psiEig=None, epsLLAverage=1e-6, alpha=0.1, verbose=True):
        mu, C = np.mean(X, axis=1), np.cov(X)
        mu = vcol(mu)
        if self.covariance_type == 'diag':
            C = np.diag(np.diag(C))  # Initial diagonal GMM to train a diagonal GMM

        if psiEig is not None:
            self.gmm = [(1.0, mu, self.smooth_covariance_matrix(C, psiEig))]
        else:
            self.gmm = [(1.0, mu, C)]

        while len(self.gmm) < numComponents:
            if verbose:
                print(f'Average ll before LBG: {self.logpdf_GMM(X).mean():.8e}')
            self.gmm = self.split_GMM_LBG(alpha, verbose=verbose)
            if verbose:
                print(f'Average ll after LBG: {self.logpdf_GMM(X).mean():.8e}')
            self.gmm = self.train_GMM_EM(X, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
        return self.gmm

    def predict(self, X):
        return self.logpdf_GMM(X)

def train_GMM(DTE, DTR, LTE, LTR):
    n_components_list = [1, 2, 4, 8, 16, 32]
    covariance_types = ['full', 'diagonal', 'tied']
    psiEig = 0.01
    alpha = 0.1
    epsLLAverage = 1e-6
    prior = 0.5
    Cfn = 1
    Cfp = 1

    results = {}

    for cov_type in covariance_types:
        results[cov_type] = {}
        for n_components in n_components_list:
            model = GMMClass(covariance_type=cov_type)
            model.train_GMM_LBG_EM(DTR, numComponents=n_components, psiEig=psiEig, alpha=alpha,
                                   epsLLAverage=epsLLAverage, verbose=True)
            scores = model.logpdf_GMM(DTE)
            min_dcf = compute_minDCF_binary_fast(scores, LTE, prior, Cfn, Cfp)
            actual_dcf = compute_actDCF_binary_fast(scores, LTE, prior, Cfn, Cfp)
            results[cov_type][n_components] = (min_dcf, actual_dcf)
            print(f'{cov_type} covariance with {n_components} components: minDCF={min_dcf}, actualDCF={actual_dcf}')
    return results







# Functions for computing DCF
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > th)

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels)  # Confusion matrix
    Pfn = M[0, 1] / (M[0, 1] + M[1, 1])
    Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
    bayesError = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_actDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp,
                                                              normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp,
                                               normalize=normalize)


def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(
        thresholdsOut)  # we return also the corresponding thresholds

def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (
                1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


