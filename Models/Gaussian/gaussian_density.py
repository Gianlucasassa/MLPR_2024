import os

import numpy
import matplotlib.pyplot as plt
import numpy as np


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


# Compute log-density for a single sample x (column vector). The result is a 1-D array with 1 element
def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5 * x.shape[0] * numpy.log(numpy.pi * 2) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu).T @ P @ (x - mu)).ravel()


# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5 * x.shape[0] * numpy.log(numpy.pi * 2) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()


def fit_univariate_gaussian_models(D, L, output_dir='Output/UnivariateGaussians'):
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
                     np.exp(logpdf_GAU_ND(XPlot.reshape(1, -1), np.array([[mu_ML]]), np.array([[C_ML]]))))
            plt.title(f'Class {cls}, Feature {i + 1}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.savefig(os.path.join(output_dir, f'Class_{cls}_Feature_{i + 1}.png'))
            plt.close()

##OLD LAB
# plt.figure()
# XPlot = numpy.linspace(-8, 12, 1000)
# m = numpy.ones((1, 1)) * 1.0
# C = numpy.ones((1, 1)) * 2.0
# plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
# # plt.show()
#
#
# pdfSol = numpy.load('llGAU.npy')
# pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
# print(numpy.abs(pdfSol - pdfGau).max())
#
# XND = numpy.load('XND.npy')
# mu = numpy.load('muND.npy')
# C = numpy.load('CND.npy')
#
# pdfSol = numpy.load('llND.npy')
# pdfGau = logpdf_GAU_ND(XND, mu, C)
# print(numpy.abs(pdfSol - pdfGau).max())
#
# # ML estimates - XND
# m_ML, C_ML = compute_mu_C(XND)
# print(m_ML)
# print(C_ML)
# print(compute_ll(XND, m_ML, C_ML))
#
# # ML estimates - X1D
# X1D = numpy.load('X1D.npy')
# m_ML, C_ML = compute_mu_C(X1D)
# print(m_ML)
# print(C_ML)
#
# plt.figure()
# plt.hist(X1D.ravel(), bins=50, density=True)
# XPlot = numpy.linspace(-8, 12, 1000)
# plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
# # plt.show()
#
# print(compute_ll(X1D, m_ML, C_ML))
# # Trying other values
# print(compute_ll(X1D, numpy.array([[1.0]]), numpy.array([[2.0]])))
# print(compute_ll(X1D, numpy.array([[0.0]]), numpy.array([[1.0]])))
# print(compute_ll(X1D, numpy.array([[2.0]]), numpy.array([[6.0]])))
