import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from .base import Kernel
# from ..signals import diag_indices
from .utils import get_dt, searchsorted

#TODO. using KernelRect with negative time bins doesnt work at all. FIX
class KernelRect(Kernel):

    def __init__(self, tbins, coefs=None, prior=None, prior_pars=None):
        self.nbasis = len(tbins) - 1
        self.tbins = np.array(tbins)
        self.support = np.array([tbins[0], tbins[-1]])
        self.coefs = np.array(coefs)
        self.prior = prior
        self.prior_pars = np.array(prior_pars)

    def evaluate(self, t, sorted_t=True):

        if sorted_t:
            res = self.interpolate_sorted(t)
        else:
            arg_bins = searchsorted(self.tbins, t, side='right') - 1

            idx = np.array(np.arange(len(arg_bins)), dtype=int)
            idx = idx[(arg_bins >= 0) & (arg_bins < len(self.tbins) - 1)]

            res = np.zeros(len(t))
            res[idx] = self.coefs[arg_bins[idx]]

        return res

    def interpolate_sorted(self, t):

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

        arg_bins = searchsorted(t, self.tbins, side='left')

        for ii, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            res[arg0:argf] = self.coefs[ii]

        return res

    def area(self, dt=None):
        return np.sum(self.coefs * np.diff(self.tbins))

    def interpolate_basis(self, t):

        arg_bins = searchsorted(t, self.tbins, side='left')
        res = np.zeros((len(t), self.nbasis))

        for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            res[arg0:argf, k] = 1.

        return res

    def plot_basis(self, t, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        arg_bins = searchsorted(t, self.tbins)

        for k, (arg0, argf) in enumerate(zip( arg_bins[:-1], arg_bins[1:] )):
            vals = np.zeros( (len(t)) )
            vals[arg0:argf] = 1.
            # ax.plot(t, vals, linewidth = 5. - 4 . * k /(len(arg_bins ) -1.) )

        return ax

    def copy(self):
        kernel = KernelRect(self.tbins.copy(), coefs=self.coefs.copy(), prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    @classmethod
    def kistler_kernels(cls, delta, dt):
        kernel1 = cls(np.array([-delta, delta + dt]), [1.])
        kernel2 = cls(np.array([0, dt]), [1. / dt])
        return kernel1, kernel2

    @classmethod
    def exponential(cls, tf=None, dt=None, tau=None, A=None, prior=None, prior_pars=None):
        tbins = np.arange(0, tf, dt)
        return cls(tbins, coefs=A * np.exp(-tbins[:-1] / tau), prior=prior, prior_pars=prior_pars)

    def convolve_basis_continuous(self, t, I):
        """# Given a 1d-array t and an nd-array x with x.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of x for all other axis values
        # so that X.shape = (x.shape, nbasis)
        # Discrete convolution can be achieved by using an x with 1/dt on the correct timing values
        Assumes sorted t"""

        dt = get_dt(t)
        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))

        t_kernel = np.arange(arg0, argf + 1, 1) * dt
        arg_bins = searchsorted(t_kernel, self.tbins)
#         print(arg_bins, len(t_kernel))
        X = np.zeros(I.shape + (self.nbasis, ))

#         basis_shape = tuple([len(t)] + [1 for ii in range(x.ndim - 1)] + [self.nbasis])
#         basis = np.zeros(basis_shape)
        basis_shape = tuple([len(t_kernel)] + [1 for ii in range(I.ndim - 1)] + [self.nbasis])
        basis = np.zeros(basis_shape)

        for k, (_arg0, _argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            basis[_arg0:_argf, ..., k] = 1.

        full_X = fftconvolve(basis, I[..., None], mode='full', axes=0)
        
#         if arg0 < 0 and argf > 0 and arg0 + argf - 1 >= 0:
#             print('a')
#             X[arg0 + argf - 1:, ...] = X[argf - 1:len(t) - arg0, ...] * dt
#         elif arg0 >= 0 and argf > 0:
#             X = X[:len(t), ...] * dt
#         elif arg0 < 0:
#             print('c')
#             print(arg0, argf)
#             X[:len(t) + arg0 + 1, ...] = X[-arg0:len(t) + 1, ...] * dt
            
        if arg0 >= 0:
            X[arg0:, ...] = full_X[:len(t) - arg0, ...]
        elif arg0 < 0 and argf >= 0:
            X = full_X[-arg0:len(t) - arg0, ...]
        elif arg0 < 0 and argf < 0:
            X[:len(t) + argf, ...] = full_X[-arg0:, ...]

        X = X * dt
            
        return X

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
#         arg_bins = searchsorted(t, self.tbins)
        arg_bins = np.array(np.floor(self.tbins / get_dt(t)), dtype=int)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)

        for ii, arg in enumerate(arg_s):
            for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
                _arg0, _argf = max(arg + arg0, 0), min(arg + argf, len(t))
                if ~(_arg0 == 0 and _argf <= 0) or ~(_arg0 >= len(t) and _argf == len(t)):
                    indices = tuple([slice(_arg0, _argf)] + [s[dim][ii] for dim in range(1, len(s))] + [k]) 
                    X[indices] += 1

        return X

    # def gh_log_prior(self, coefs):
    #
    #     if self.prior == 'exponential':
    #         lam, mu = self.prior_pars[0], np.exp(-self.prior_pars[1] * np.diff(self.tbins[:-1]))
    #
    #         log_prior = -lam * np.sum((coefs[1:] - mu * coefs[:-1]) ** 2)
    #
    #         g_log_prior = np.zeros(len(coefs))
    #         # TODO. somethingg odd with g_log_prior[0]. FIX
    #         g_log_prior[1] = -2 * lam * mu[0] * (coefs[1] - mu[0] * coefs[0])
    #         g_log_prior[2:-1] = 2 * lam * (-mu[:-1] * coefs[:-2] + (1 + mu[1:] ** 2) * coefs[1:-1] - mu[1:] * coefs[2:])
    #         g_log_prior[-1] = 2 * lam * (coefs[-1] - mu[-1] * coefs[-2])
    #         g_log_prior = -g_log_prior
    #
    #         h_log_prior = np.zeros((len(coefs), len(coefs)))
    #
    #         h_log_prior[1, 1], h_log_prior[1, 2] = mu[0] ** 2, -mu[0]
    #         h_log_prior[2:-1, 2:-1][diag_indices(len(coefs) - 2, k=0)] = 1 + mu[1:] ** 2
    #         h_log_prior[2:-1, 2:-1][diag_indices(len(coefs) - 2, k=1)] = -mu[1:-1]
    #         h_log_prior[-1, -1] = 1
    #         h_log_prior = -2 * lam * h_log_prior
    #
    #         h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
    #             np.tril_indices_from(h_log_prior, k=-1)]
    #
    #     elif self.prior == 'smooth_2nd_derivative':
    #
    #         lam = self.prior_pars[0]
    #
    #         log_prior = -lam * np.sum((coefs[:-2] + coefs[2:] - 2 * coefs[1:-1]) ** 2)
    #
    #         g_log_prior = np.zeros(len(coefs))
    #         g_log_prior[0] = -2 * lam * (coefs[0] - 2 * coefs[1] + coefs[2])
    #         g_log_prior[1] = -2 * lam * (-2 * coefs[0] + 5 * coefs[1] - 4 * coefs[2] + coefs[3])
    #         g_log_prior[2:-2] = -2 * lam * \
    #                     (coefs[:-4] - 4 * coefs[1:-3] + 6 * coefs[2:-2] - 4 * coefs[3:-1] + coefs[4:])
    #         g_log_prior[-2] = -2 * lam * (coefs[-4] - 4 * coefs[-3] + 5 * coefs[-2] - 2 * coefs[-1])
    #         g_log_prior[-1] = -2 * lam * (coefs[-3] - 2 * coefs[-2] + coefs[-1])
    #
    #         h_log_prior = np.zeros((len(coefs), len(coefs)))
    #         h_log_prior[0, 0], h_log_prior[0, 1], h_log_prior[0, 2] = 1, -2, 1
    #         h_log_prior[1, 1], h_log_prior[1, 2], h_log_prior[1, 3] = 5, -4, 1
    #         h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=0)] = 6
    #         h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=1)] = -4
    #         h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=2)] = 1
    #         h_log_prior[-2, -2], h_log_prior[-2, -1] = 5, -2
    #         h_log_prior[-1, -1] = 1
    #         h_log_prior = - 2 * lam * h_log_prior
    #         h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
    #             np.tril_indices_from(h_log_prior, k=-1)]
    #
    #     return log_prior, g_log_prior, h_log_prior