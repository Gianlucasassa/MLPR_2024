import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as confMAt

from Preprocess.PCA import apply_pca, apply_PCA_from_dim


def compute_dcf(predictions, labels, pi1, Cfn, Cfp):
    confusion_matrix = compute_confusion_matrix(predictions, labels)
    bayes_risk = compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
    normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
    return normalized_dcf

def compute_min_dcf(llrs, labels, pi1, Cfn, Cfp):
    thresholds = np.sort(llrs)
    min_dcf = float('inf')
    for t in thresholds:
        predictions = (llrs >= t).astype(int)
        confusion_matrix = compute_confusion_matrix(predictions, labels)
        bayes_risk = compute_bayes_risk(confusion_matrix, pi1, Cfn, Cfp)
        normalized_dcf = compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp)
        if normalized_dcf < min_dcf:
            min_dcf = normalized_dcf
    return min_dcf

class GaussianClassifier:
    def __init__(self, model_type='MVG'):
        self.means = None
        self.covariances = None
        self.model_type = model_type

    def logpdf_GAU_ND(self, X, mu, C):
        M = X.shape[0]
        inv_C = np.linalg.inv(C)
        log_det_C = np.linalg.slogdet(C)[1]
        diff = X - mu
        log_density = -0.5 * (M * np.log(2 * np.pi) + log_det_C + np.sum(diff * np.dot(inv_C, diff), axis=0))
        return log_density

    def compute_ml_estimates(self, D, L):
        classes = np.unique(L)
        means = []
        covariances = []
        for cls in classes:
            class_data = D[:, L == cls]
            mu_ML = class_data.mean(axis=1, keepdims=True)
            if self.model_type == 'NaiveBayes':
                C_ML = np.diag(np.diag(np.cov(class_data)))
            else:
                C_ML = np.cov(class_data)
            means.append(mu_ML)
            covariances.append(C_ML)
        if self.model_type == 'TiedCovariance':
            covariances = [np.mean(covariances, axis=0)] * len(classes)
        return means, covariances

    def train(self, DTR, LTR):
        self.means, self.covariances = self.compute_ml_estimates(DTR, LTR)

    def predict(self, DTE):
        logS = []
        for mu, C in zip(self.means, self.covariances):
            logS.append(self.logpdf_GAU_ND(DTE, mu, C))
        return np.array(logS)

    def compute_llrs(self, logS):
        return logS[1] - logS[0]

    def compute_predictions(self, logS, threshold=0):
        llrs = self.compute_llrs(logS)
        return (llrs >= threshold).astype(int)

    def compute_error_rate(self, predictions, true_labels):
        return np.mean(predictions != true_labels)

    def evaluate(self, DTR, LTR, DTE, LTE, output_dir='Output/ClassifierResults'):
        os.makedirs(output_dir, exist_ok=True)
        self.train(DTR, LTR)
        logS = self.predict(DTE)
        predictions = self.compute_predictions(logS)
        error_rate = self.compute_error_rate(predictions, LTE)
        self.plot_classification_results(DTE, LTE, predictions, os.path.join(output_dir, f'{self.model_type}_Results.png'))
        return error_rate

    def plot_classification_results(self, D, L_true, L_pred, output_file):
        plt.figure()
        for cls in np.unique(L_true):
            plt.scatter(D[0, L_true == cls], D[1, L_true == cls], label=f'True Class {cls}', alpha=0.5)
            plt.scatter(D[0, L_pred == cls], D[1, L_pred == cls], marker='x', label=f'Predicted Class {cls}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('Classification Results')
        plt.savefig(output_file)
        plt.close()

    def fit_univariate_gaussian_models(self, D, L, output_dir='Output/UnivariateGaussians'):
        os.makedirs(output_dir, exist_ok=True)
        classes = np.unique(L)
        for cls in classes:
            for i in range(D.shape[0]):
                feature_data = D[i, L == cls]
                mu_ML = feature_data.mean()
                C_ML = feature_data.var()
                XPlot = np.linspace(feature_data.min(), feature_data.max(), 1000)
                plt.figure()
                plt.hist(feature_data, bins=50, density=True, alpha=0.6, color='g')
                plt.plot(XPlot,
                         np.exp(self.logpdf_GAU_ND(XPlot.reshape(1, -1), np.array([[mu_ML]]), np.array([[C_ML]]))))
                plt.title(f'Class {cls}, Feature {i + 1}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.savefig(os.path.join(output_dir, f'Class_{cls}_Feature_{i + 1}.png'))
                plt.close()

    def logpdf_GAU_1D(self, X, mu, var):
        log_density = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
        return log_density

    def vrow(self, col):
        return col.reshape((1, col.size))

    def vcol(self, row):
        return row.reshape((row.size, 1))

    def plot_loglikelihood(self, X, mu, C, output_file):
        ll = np.sum(self.logpdf_GAU_ND(X, mu, C))
        plt.figure()
        plt.hist(X.ravel(), bins=50, density=True)
        XPlot = np.linspace(X.min(), X.max(), 1000)
        plt.plot(XPlot.ravel(), np.exp(self.logpdf_GAU_ND(self.vrow(XPlot), mu, C)))
        plt.title(f'Log-Likelihood: {ll}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.savefig(output_file)
        plt.close()
        return ll

    def analyze_covariances(self, DTR, LTR, output_dir='Output/Covariances'):
        os.makedirs(output_dir, exist_ok=True)
        means, covariances = self.compute_ml_estimates(DTR, LTR)
        for cls, (mu, C) in enumerate(zip(means, covariances)):
            print(f"Class {cls} Mean:\n{mu}")
            print(f"Class {cls} Covariance Matrix:\n{C}")
            corr = C / (self.vcol(np.diag(C) ** 0.5) * self.vrow(np.diag(C) ** 0.5))
            print(f"Class {cls} Correlation Matrix:\n{corr}")
            self.plot_matrix(C, os.path.join(output_dir, f'Class_{cls}_Covariance.png'))
            self.plot_matrix(corr, os.path.join(output_dir, f'Class_{cls}_Correlation.png'))

    def plot_matrix(self, matrix, output_file):
        plt.figure()
        plt.imshow(matrix, interpolation='nearest', cmap='coolwarm')
        plt.colorbar()
        plt.title('Matrix Heatmap')
        plt.savefig(output_file)
        plt.close()

    # Additional methods for confusion matrix, Bayes decisions, empirical Bayes risk, normalized detection cost, ROC curves, and Bayes error plots

    def compute_confusion_matrix(self, predictions, labels):
        K = len(np.unique(labels))
        confusion_matrix = np.zeros((K, K), dtype=int)
        for i in range(len(labels)):
            confusion_matrix[predictions[i], labels[i]] += 1
        return confusion_matrix

    def compute_optimal_bayes_decisions_class(self, llrs, pi1, Cfn, Cfp):
        t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
        return (llrs >= t).astype(int)

    def compute_bayes_risk(self, confusion_matrix, pi1, Cfn, Cfp):
        FN = confusion_matrix[0, 1]
        FP = confusion_matrix[1, 0]
        TN = confusion_matrix[0, 0]
        TP = confusion_matrix[1, 1]
        Pfn = FN / (FN + TP)
        Pfp = FP / (FP + TN)
        DCFu = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
        return DCFu

    def compute_normalized_dcf(self, bayes_risk, pi1, Cfn, Cfp):
        Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
        return bayes_risk / Bdummy

    def plot_roc_curve(self, llrs, labels, output_file='roc_curve.png'):
        out_dir = "Output/MVG/Roc"
        os.makedirs(out_dir, exist_ok=True)
        thresholds = np.sort(llrs)
        TPR = []
        FPR = []
        for t in thresholds:
            predictions = (llrs >= t).astype(int)
            confusion_matrix = self.compute_confusion_matrix(predictions, labels)
            FN = confusion_matrix[0, 1]
            FP = confusion_matrix[1, 0]
            TN = confusion_matrix[0, 0]
            TP = confusion_matrix[1, 1]
            TPR.append(TP / (TP + FN))
            FPR.append(FP / (FP + TN))
        plt.plot(FPR, TPR, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid()
        plt.savefig(os.path.join(out_dir, output_file))
        plt.close()

    def plot_bayes_error(self, llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
        out_dir = 'Output/Bayes'
        os.makedirs(out_dir, exist_ok=True)
        dcf = []
        mindcf = []
        for p in effPriorLogOdds:
            pi1 = 1 / (1 + np.exp(-p))
            decisions = self.compute_optimal_bayes_decisions_class(llrs, pi1, 1, 1)
            confusion_matrix = self.compute_confusion_matrix(decisions, labels)
            bayes_risk = self.compute_bayes_risk(confusion_matrix, pi1, 1, 1)
            normalized_dcf = self.compute_normalized_dcf(bayes_risk, pi1, 1, 1)
            dcf.append(normalized_dcf)

            min_bayes_risk = self.compute_bayes_risk(confusion_matrix, pi1, 1, 1)
            min_normalized_dcf = self.compute_normalized_dcf(min_bayes_risk, pi1, 1, 1)
            mindcf.append(min_normalized_dcf)

        plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
        plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.xlabel('Prior Log-Odds')
        plt.ylabel('DCF value')
        plt.legend()
        plt.grid()
        plt.title('Bayes Error Plot')
        plt.savefig(os.path.join(out_dir, output_file))
        plt.close()

def compute_confusion_matrix(predictions, labels):
    K = 2  # Since it's a binary classification problem
    confusion_matrix = np.zeros((K, K), dtype=int)

    # Flatten predictions if it's a 2D array
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    #print(f"Predictions: {predictions}")
    #print(f"Labels: {labels}")

    for i in range(len(labels)):
        pred = int(predictions[i])
        true = int(labels[i])
        if pred < 0 or pred >= K or true < 0 or true >= K:
            #print(f"Invalid prediction or label: pred={pred}, true={true}")
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

def plot_bayes_error(llrs, labels, effPriorLogOdds, output_file='bayes_error_plot.png'):
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    dcf = []
    mindcf = []
    for p in effPriorLogOdds:
        pi1 = 1 / (1 + np.exp(-p))
        decisions = compute_optimal_bayes_decisions_main(llrs, pi1, 1, 1)
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
    plt.xlim([-3, 3])
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF value')
    plt.legend()
    plt.grid()
    plt.title('Bayes Error Plot')
    plt.savefig(output_file)
    plt.close()

def compute_normalized_dcf(bayes_risk, pi1, Cfn, Cfp):
    Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
    return bayes_risk / Bdummy

def plot_confusion_matrix(cm, model_name, output_file):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(output_file)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file='roc_curve.png'):
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(output_file)
    plt.close()

def compute_optimal_bayes_decisions_main(llrs, pi1, Cfn, Cfp):
    t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    return (llrs >= t).astype(int)



def train_MVG_1(DTE, DTR, LTE, LTR):
    def evaluate_model(classifier, model_name, DTR, LTR, DTE, LTE, output_dir):
        classifier.train(DTR, LTR)
        logS = classifier.predict(DTE)
        predictions = classifier.compute_predictions(logS)
        error_rate = classifier.compute_error_rate(predictions, LTE)
        print(f"{model_name} Error Rate: {error_rate}")

        fpr, tpr, _ = roc_curve(LTE, logS[1])
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, model_name, os.path.join(output_dir, f'ROC_{model_name}.png'))

        cm = confMAt(LTE, predictions)
        plot_confusion_matrix(cm, model_name, os.path.join(output_dir, f'CM_{model_name}.png'))

        return error_rate

    output_dir = 'Output/ClassifierResults'
    # MVG
    mvg_classifier = GaussianClassifier(model_type='MVG')
    evaluate_model(mvg_classifier, 'MVG', DTR, LTR, DTE, LTE, output_dir)
    # Naive Bayes
    nb_classifier = GaussianClassifier(model_type='NaiveBayes')
    evaluate_model(nb_classifier, 'NaiveBayes', DTR, LTR, DTE, LTE, output_dir)
    # Tied Covariance
    tied_classifier = GaussianClassifier(model_type='TiedCovariance')
    evaluate_model(tied_classifier, 'TiedCovariance', DTR, LTR, DTE, LTE, output_dir)

    output_dir = 'Output/UnivariateGaussians'
    mvg_classifier.fit_univariate_gaussian_models(DTR, LTR, output_dir)
    classes = np.unique(LTR)
    for cls in classes:
        for i in range(DTR.shape[0]):
            feature_data = DTR[i, LTR == cls]
            mu_ML = feature_data.mean()
            C_ML = feature_data.var()
            ll = mvg_classifier.plot_loglikelihood(feature_data, np.array([[mu_ML]]), np.array([[C_ML]]),
                                                   os.path.join(output_dir,
                                                                f'LogLikelihood_Class_{cls}_Feature_{i + 1}.png'))
            print(f"Class {cls}, Feature {i + 1} Log-Likelihood: {ll}")
    # Compute covariance and correlation matrices for MVG model
    mvg_classifier.analyze_covariances(DTR, LTR)
    # Repeat the analysis using only features 1 to 4 (discarding the last 2 features)
    DTR_reduced = DTR[:4, :]
    DTE_reduced = DTE[:4, :]
    # MVG with reduced features
    mvg_classifier_reduced = GaussianClassifier(model_type='MVG')
    evaluate_model(mvg_classifier_reduced, 'MVG_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
    # Tied Covariance with reduced features
    tied_classifier_reduced = GaussianClassifier(model_type='TiedCovariance')
    evaluate_model(tied_classifier_reduced, 'TiedCovariance_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
    # Naive Bayes with reduced features
    nb_classifier_reduced = GaussianClassifier(model_type='NaiveBayes')
    evaluate_model(nb_classifier_reduced, 'NaiveBayes_Reduced', DTR_reduced, LTR, DTE_reduced, LTE, output_dir)
    # Use PCA to reduce the dimensionality and apply the three classification approaches

    # Reduce to 2 principal components for example
    pca_dim = 2

    # Apply PCA using the new functions
    DTR_pca, P_pca = apply_PCA_from_dim(DTR, pca_dim)
    DTE_pca = apply_pca(DTE, P_pca)

    # MVG with PCA
    mvg_classifier_pca = GaussianClassifier(model_type='MVG')
    evaluate_model(mvg_classifier_pca, 'MVG_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)

    # Tied Covariance with PCA
    tied_classifier_pca = GaussianClassifier(model_type='TiedCovariance')
    evaluate_model(tied_classifier_pca, 'TiedCovariance_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)

    # Naive Bayes with PCA
    nb_classifier_pca = GaussianClassifier(model_type='NaiveBayes')
    evaluate_model(nb_classifier_pca, 'NaiveBayes_PCA', DTR_pca, LTR, DTE_pca, LTE, output_dir)

def train_MVG(DTE, DTR, LTE, LTR):
    output_dir = "Output/MVG"

    models = {
        'MVG': GaussianClassifier(model_type='MVG'),
        'NaiveBayes': GaussianClassifier(model_type='NaiveBayes'),
        'TiedCovariance': GaussianClassifier(model_type='TiedCovariance')
    }

    priors = [0.5, 0.9, 0.1]
    costs = [(1.0, 1.0), (1.0, 9.0), (9.0, 1.0)]
    pca_dims = [2, 4, 6, None]  # None means no PCA

    for pca_dim in pca_dims:
        if pca_dim is not None:
            # Apply PCA
            DTR_pca, P_pca = apply_PCA_from_dim(DTR, pca_dim)
            DTE_pca = apply_pca(DTE, P_pca)
        else:
            # No PCA
            DTR_pca, DTE_pca = DTR, DTE

        for model_name, model in models.items():
            model.train(DTR_pca, LTR)

        for pi1, (Cfn, Cfp) in [(pi1, cost) for pi1 in priors for cost in costs]:
            print(f"Analyzing for pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}, PCA dim={pca_dim}")
            for model_name, model in models.items():
                logS = model.predict(DTE_pca)
                # print(f"{model_name} LLRs: {logS[:5]}")  # Print first 5 LLRs for inspection
                predictions = model.compute_predictions(logS)
                # print(f"{model_name} Predictions: {predictions[:5]}")  # Print first 5 predictions for inspection
                dcf = compute_dcf(predictions, LTE, pi1, Cfn, Cfp)
                min_dcf = compute_min_dcf(logS, LTE, pi1, Cfn, Cfp)
                print(f"{model_name} Normalized DCF: {dcf}")
                print(f"{model_name} Min DCF: {min_dcf}")

        # Plotting functions and other evaluations
        for model_name, model in models.items():
            llrs = model.compute_llrs(model.predict(DTE_pca))
            fpr, tpr, _ = roc_curve(LTE, llrs)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, f"{model_name}_PCA_{pca_dim}", os.path.join(output_dir, f'roc_curve_{model_name}_PCA_{pca_dim}.png'))

            plot_bayes_error(llrs, LTE, np.linspace(-3, 3, 21), os.path.join(output_dir, f'bayes_error_plot_{model_name}_PCA_{pca_dim}.png'))




'''

def plot_loglikelihood(X, mu, C, output_file):
    ll = loglikelihood(X, mu, C)
    plt.figure()
    plt.hist(X.ravel(), bins=50, density=True)
    XPlot = np.linspace(X.min(), X.max(), 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
    plt.title(f'Log-Likelihood: {ll}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(output_file)
    plt.close()
    return ll


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def compute_ml_estimates(D, L):
    classes = np.unique(L)
    means = []
    covariances = []
    for cls in classes:
        class_data = D[:, L == cls]
        mu_ML = class_data.mean(axis=1, keepdims=True)
        C_ML = np.cov(class_data)
        means.append(mu_ML)
        covariances.append(C_ML)
    return means, covariances


def mvg_classifier(DTR, LTR, DTE):
    means, covariances = compute_ml_estimates(DTR, LTR)
    logS = []
    for mu, C in zip(means, covariances):
        logS.append(logpdf_GAU_ND(DTE, mu, C))
    return np.array(logS)


def compute_llrs(logS):
    return logS[1] - logS[0]


def tied_covariance_classifier(DTR, LTR, DTE):
    means, _ = compute_ml_estimates(DTR, LTR)
    SW = compute_within_class_covariance(DTR, LTR)
    logS = []
    for mu in means:
        logS.append(logpdf_GAU_ND(DTE, mu, SW))
    return np.array(logS)


def naive_bayes_classifier(DTR, LTR, DTE):
    means, covariances = compute_ml_estimates(DTR, LTR)
    logS = []
    for mu, C in zip(means, covariances):
        C_diag = np.diag(np.diag(C))
        logS.append(logpdf_GAU_ND(DTE, mu, C_diag))
    return np.array(logS)


def compute_predictions(logS, threshold=0):
    llrs = compute_llrs(logS)
    return (llrs >= threshold).astype(int)


def compute_error_rate(predictions, true_labels):
    return np.mean(predictions != true_labels)


def evaluate_classifiers(DTR, LTR, DTE, LTE, output_dir='Output/ClassifierResults'):
    os.makedirs(output_dir, exist_ok=True)
    logS_mvg = mvg_classifier(DTR, LTR, DTE)
    predictions_mvg = compute_predictions(logS_mvg)
    error_mvg = compute_error_rate(predictions_mvg, LTE)
    plot_classification_results(DTE, LTE, predictions_mvg, os.path.join(output_dir, 'MVG_Results.png'))

    logS_tied = tied_covariance_classifier(DTR, LTR, DTE)
    predictions_tied = compute_predictions(logS_tied)
    error_tied = compute_error_rate(predictions_tied, LTE)
    plot_classification_results(DTE, LTE, predictions_tied, os.path.join(output_dir, 'Tied_Results.png'))

    logS_nb = naive_bayes_classifier(DTR, LTR, DTE)
    predictions_nb = compute_predictions(logS_nb)
    error_nb = compute_error_rate(predictions_nb, LTE)
    plot_classification_results(DTE, LTE, predictions_nb, os.path.join(output_dir, 'NaiveBayes_Results.png'))

    print(f"MVG Error Rate: {error_mvg}")
    print(f"Tied Covariance Error Rate: {error_tied}")
    print(f"Naive Bayes Error Rate: {error_nb}")

    return error_mvg, error_tied, error_nb


def plot_classification_results(D, L_true, L_pred, output_file):
    plt.figure()
    for cls in np.unique(L_true):
        plt.scatter(D[0, L_true == cls], D[1, L_true == cls], label=f'True Class {cls}', alpha=0.5)
        plt.scatter(D[0, L_pred == cls], D[1, L_pred == cls], marker='x', label=f'Predicted Class {cls}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Classification Results')
    plt.savefig(output_file)
    plt.close()


def analyze_covariances(DTR, LTR, output_dir='Output/Covariances'):
    os.makedirs(output_dir, exist_ok=True)
    means, covariances = compute_ml_estimates(DTR, LTR)
    for cls, (mu, C) in enumerate(zip(means, covariances)):
        print(f"Class {cls} Mean:\n{mu}")
        print(f"Class {cls} Covariance Matrix:\n{C}")
        corr = C / (vcol(np.diag(C) ** 0.5) * vrow(np.diag(C) ** 0.5))
        print(f"Class {cls} Correlation Matrix:\n{corr}")
        plot_matrix(C, os.path.join(output_dir, f'Class_{cls}_Covariance.png'))
        plot_matrix(corr, os.path.join(output_dir, f'Class_{cls}_Correlation.png'))


def plot_matrix(matrix, output_file):
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    plt.title('Matrix Heatmap')
    plt.savefig(output_file)
    plt.close()


def compute_confusion_matrix(predictions, labels):
    K = 2  # Since it's a binary classification problem
    confusion_matrix = np.zeros((K, K), dtype=int)

    # Flatten predictions if it's a 2D array
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    #print(f"Predictions: {predictions}")
    #print(f"Labels: {labels}")

    for i in range(len(labels)):
        pred = int(predictions[i])
        true = int(labels[i])
        if pred < 0 or pred >= K or true < 0 or true >= K:
            #print(f"Invalid prediction or label: pred={pred}, true={true}")
            continue  # Skip invalid values
        confusion_matrix[pred, true] += 1

    return confusion_matrix


def compute_optimal_bayes_decisions(llrs, pi1, Cfn, Cfp):
    t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    return (llrs >= t).astype(int)


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


def plot_roc_curve(fpr, tpr, roc_auc, model_name, output_file='roc_curve.png'):
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(output_file)
    plt.close()


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
    plt.xlim([-3, 3])
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF value')
    plt.legend()
    plt.grid()
    plt.title('Bayes Error Plot')
    plt.savefig(output_file)
    plt.close()
'''
