#!/usr/bin/env python

'''
Functions to estimate capping inversion parameters

These scripts started from scripts from Dries Allaerts. TODO add reference

Author: Sebastiaan Jamaer (RZse)
Date: May 22, 2017; June 2021
'''
from typing import List

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize, Bounds, least_squares, LinearConstraint, NonlinearConstraint
from scipy.linalg import solve
import matplotlib.pyplot as plt
from os import path
import sys
import time
sys.path.append(path.abspath('/home/u0115935/Documents/PhD/data'))


CONDITIONERS = [None, 'regularization', 'constraints', 'regularisation']

def RZmodel(z, a, b, thm, l, dh):
    '''
    Smooth curve representing the vertical potential temperature profile
    of a neutral atmospheric boundary layer with a capping inversion
    Rampanelli & Zardi (2004)

    Parameters
    ----------
    z: numpy 1D array
        height
    a,b,thm,l,dh: float
        fitting parameters

    Returns (th)
    ------------
    th: numpy 1D array
        vertical potential temperature profile
    '''
    ksi = 1.5
    c = 1.0 / (2 * ksi)
    eta = (z - l) / (c * dh)

    f = (np.tanh(eta) + 1.0) / 2.0
    with np.errstate(over='ignore'):  # If g goes to infinite, I display an error message and I ingore it
        g = (np.log(2 * np.cosh(eta)) + eta) / 2.0
    for i in np.where(np.isinf(g)):  # when g is infinite, I replace it with this (?)
        g[i] = (np.abs(eta[i]) + eta[i]) / 2.0

    th = thm + a * f + b * g
    return th
def RZfit(z, th, p0=np.array([1, 1, 300, 1000, 100]), dh_max=None, max_nfev=1000):
    '''
    Estimate parameters of a smooth analytical curve to fit a given
    vertical potential temperature profile

    Parameters
    ----------
    z, th: numpy 1D array
        height and potential temperature profile
    p0: list of length 5
        Initial guess for [a,b,thm,l,dh]
    dh_max:
        Optional upper bound for dh
    Returns (CIdata)
    ----------------
    CIdata: dict
        capping inversion parameters
    '''
    if not dh_max:
        # parameter list: a, b, thm, l, dh
        lbound = [0., 0., 0., 0., 0.]
        ubound = [np.inf, np.inf, np.inf, np.inf, np.inf]
    else:
        lbound = [0., 0., 0., 0., 0.]
        ubound = [np.inf, np.inf, np.inf, np.inf, dh_max]
    bounds = [lbound, ubound]

    # noinspection PyArgumentList
    def _wrap_func(z, th):
        def _residuals(params):
            return RZmodel(z, *params) - th
        return _residuals
    jac = '2-point'
    func = _wrap_func(z, th)
    method = 'trf'

    t_start = time.time()
    result = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                        max_nfev=max_nfev)
    t_end = time.time()


    [a, b, thm, l, dh] = result.x
    CIData = {}
    CIData['a'] = result.x[0]
    CIData['b'] = result.x[1]
    CIData['thm'] = result.x[2]
    CIData['l'] = result.x[3]
    CIData['dh'] = result.x[4]
    CIData['thfit'] = RZmodel(z=z, a=a, b=b, thm=thm, l=l, dh=dh)
    CIData['MAE'] = np.mean(np.abs(CIData['thfit'] - th))
    CIData['MSE'] = np.mean((CIData['thfit'] - th)**2)
    CIData['time'] = t_end - t_start
    CIData['success'] = result.success
    CIData['status'] = result.status
    CIData['nfev'] = result.nfev
    return CIData


def RZsemodel(z, a, b, c, thm, l, dh, alpha):
    '''
    Smooth curve representing the vertical potential temperature profile
    of a neutral atmospheric boundary layer with a capping inversion
    Rampanelli & Zardi (2004) extended with the surface layer function.

    Parameters
    ----------
    z: numpy 1D array
        height
    a,b,c,thm,l,dh,alpha: float
        fitting parameters

    Returns (th)
    ------------
    th: numpy 1D array
        vertical potential temperature profile
    '''

    _eta1 = eta1(z, l, dh)
    _eta2 = eta2(z, l, alpha)

    _f = f(_eta1)
    _g = g(_eta1)
    _h = h(_eta2)

    th = thm + a * _f + b * _g + c * _h
    return th
def RZSEfit(z, th, p0=np.array([1, 1, 0., 300, 1000, 100, .05]), initialGuess='RZ', dh_max=None,
            conditioning='constraints', reg_factor=0, part_linear=False, solver='trust-constr',
            jacobian='manual', hessian=None, options=None):

    if options is None:
        options = {}
    if part_linear:
        return None

    assert reg_factor >= 0 or reg_factor is None, 'regularization factor should be non-negative  (>=0)'
    assert conditioning in CONDITIONERS, 'Conditioning is should be in ' + ' '.join(map(str, CONDITIONERS))


    # Setting the bounds for the complete system
    # Parameter vector has form [ a, b, c, thm, l, dh, alpha]
    ubound = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0]
    lbound = [0., 0., -np.inf, 0., 0., 10, 0.]

    if dh_max is not None:
        ubound[5] = dh_max

    if conditioning == 'constraints':
        constraints = get_constraints()
    else:
        constraints = ()
    regularisation_multiplicators= get_regularisation_multiplicators()
    cost_function = get_cost_function(z, th, reg_factor, regularisation_multiplicators)
    if jacobian =='manual':
        jacobian = get_Jac(z, th, reg_factor=reg_factor, reg_mult=regularisation_multiplicators)
    if hessian == 'manual':
        hessian  = get_Hess(z, th, reg_factor=reg_factor, reg_mult=regularisation_multiplicators)
    t_start = time.time()

    used_p0RZ = 0
    # First compute initial guess with RZ
    if initialGuess == 'RZ':
        if 'maxiter' in options:
             max_nfev = int(options['maxiter']/5)
             options['maxiter'] -= max_nfev
        else:
             max_nfev = 100
        p0RZ = RZfit(z, th, p0=[p0[0], p0[1], p0[3], p0[4], p0[5]], dh_max=1000, max_nfev=max_nfev)
        if satisfiesConstraints(p0RZ):
            p0 = [p0RZ['a'], p0RZ['b'], p0[2], p0RZ['thm'], p0RZ['l'], p0RZ['dh'], p0[6]]
            used_p0RZ = 1
    result = minimize(cost_function, p0, args=(), method=solver, jac=jacobian, hess=hessian,
             bounds=Bounds(lbound, ubound), constraints=constraints, tol=None, callback=None, options=options)
    t_end = time.time()

    [a, b, c, thm, l, dh, alpha] = result.x
    CIData = {}
    CIData['a'] = result.x[0]
    CIData['b'] = result.x[1]
    CIData['c'] = result.x[2]
    CIData['thm'] = result.x[3]
    CIData['l'] = result.x[4]
    CIData['dh'] = result.x[5]
    CIData['alpha'] = result.x[6]
    CIData['thfit'] = RZsemodel(z=z, a=a, b=b, c=c, thm=thm, l=l, dh=dh, alpha=alpha)
    CIData['MAE'] = np.mean(np.abs(CIData['thfit'] - th))
    CIData['MSE'] = np.mean((CIData['thfit'] - th)**2)
    CIData['time'] = t_end - t_start
    CIData['success'] = result.success
    CIData['status'] = result.status
    CIData['nfev'] = result.nfev
    CIData['used_p0RZ'] = used_p0RZ
    return CIData

def get_regularisation_multiplicators():
    """
    Return the regularisation multiplicators. These multiplicators are the inverse of the regularization weights. Taking
    the inverse allows us to ignore dealing with infinity for some weights
    :return: dictionary with the inverse weights for each parameter in the RZSE model
    """
    multiplicators = {}
    multiplicators['a'] = 1/5  # Kelvin^-1
    multiplicators['b'] = 0
    multiplicators['c'] = 0
    multiplicators['thm'] = 0
    multiplicators['l'] = 1/1000  # meter^-1
    multiplicators['dh'] = 1/100  # meter^-1
    multiplicators['alpha'] = 1/0.2  # meter/meter
    return multiplicators
def get_constraints(max_height_CI=4000):
    constraints = []

    # l-dh/2>alpha*l, the bottom of the CI is higher than the top of the surface layer
    con = lambda x: x[4] - x[5]/2 - x[6]*x[4]
    jac = lambda x: np.array([0, 0, 0, 0, 1-x[6], -1/2, -x[4]])
    def hess(x, v):
        """
        The hessian of np.dot(con*v) where v is a lagrangian multiplier (since con has dim 1, just a real number)
        :param x: array of parameters [a, b, c, thm, l, dh, alpha]
        :param v: lagrangian multiplier
        :return: hessian of np.dot(con*v) with derivatives to the parameters
        """
        hessian = np.zeros(shape=(7,7))
        hessian[6, 4] = -v
        hessian[4, 6] = -v
        return hessian
    constraints.append(NonlinearConstraint(con, 0, np.inf, jac=jac, hess=hess))

    # l+dh/2<4000, the top of the CI is lower than 4000 m
    _A = [0, 0, 0, 0, 1, 1/2, 0]
    constraints.append(LinearConstraint(_A, -np.inf, max_height_CI))

    return constraints
def satisfiesConstraints(p0RZ, max_height_CI=4000, min_height_SL=10):
    if p0RZ['l'] - p0RZ['dh']/2 > min_height_SL and p0RZ['l'] + p0RZ['dh']/2 < max_height_CI:
        return True
    return False

def get_cost_function(z, th, reg_factor, reg_mult):
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])
    def cost_function(par):
        [a, b, c, thm, l, dh, alpha] = par
        th_approximation = RZsemodel(z, a=a, b=b, c=c, thm=thm, l=l, dh=dh, alpha=alpha)
        cost = np.mean((th - th_approximation)**2) + reg_factor*np.sum((par*multiplicator_list)**2)
        return cost
    return cost_function
def get_Jac(z, th, reg_factor, reg_mult):
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])
    def jacobian(par):
        [a, b, c, thm, l, dh, alpha] = par
        residuals = th - RZsemodel(z, a, b, c, thm, l, dh, alpha)
        N = len(th)

        return -2/N * residuals @ _grad_RZSE(z, par) + 2*reg_factor*par*multiplicator_list**2
    return jacobian

def get_Hess(z, th, reg_factor, reg_mult):
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])
    def hessian(par):
        [a, b, c, thm, l, dh, alpha] = par
        N = len(th)
        residuals = th - RZsemodel(z, a, b, c, thm, l, dh, alpha)
        grad_rzse = _grad_RZSE(z, par)
        return -2/N * (np.tensordot(residuals, _hess_RZSE(z, par), axes=(0,0)) -
                       np.tensordot(grad_rzse, grad_rzse, axes=(0,0)))\
               + reg_factor*np.diag(multiplicator_list**2)
    return hessian

def _grad_RZSE(z, par):
    [a, b, c, thm, l, dh, alpha] = par
    grad = np.zeros((len(z), 7))
    _eta1 = eta1(z=z, l=l, dh=dh)
    _eta2 = eta2(z=z, l=l, alpha=alpha)
    grad[:, 0] = f(_eta1)
    grad[:, 1] = g(_eta1)
    grad[:, 2] = h(_eta2)
    grad[:, 3] = 1

    _temp1 = a*dfde1(_eta1) + b*dgde1(_eta1)
    _temp2 = c * dhde2(_eta2)

    grad[:, 4] = de1dl(z=z, l=l, dh=dh) * _temp1 + de2dl(z=z, l=l, alpha=alpha) * _temp2
    grad[:, 5] = de1ddh(z=z, l=l, dh=dh) * _temp1 + de2ddh(z=z, l=l, alpha=alpha) * _temp2
    grad[:, 6] = de1dalpha(z=z, l=l, dh=dh) * _temp1 + de2dalpha(z=z, l=l, alpha=alpha) * _temp2
    return grad
def _hess_RZSE(z, par):
    [a, b, c, thm, l, dh, alpha] = par
    _eta1 = eta1(z, l, dh)
    _eta2 = eta2(z, l, alpha)

    hess = np.zeros(shape=(len(z), len(par), len(par)))
    hess[0:4, 0:4] = 0  # (da or db or dc or dthm)(da or db or dc or dthm)

    # temporary save partial derivatives for efficiency
    _dfde1 = dfde1(_eta1)
    _dgde1 = dgde1(_eta1)
    _dhde2 = dhde2(_eta2)
    _de1dl = de1dl(z, l, dh)
    _de1ddh = de1ddh(z, l, dh)
    _de2dl = de2dl(z, l, alpha)
    _de2dalpha = de2dalpha(z, l, alpha)

    _ddfde1de1 = ddfde1de1(_eta1)
    _ddgde1de1 = ddgde1de1(_eta1)
    _ddhde2de2 = ddhde2de2(_eta2)

    _dde1dlddh = dde1dlddh(z, l, dh)
    _dde1ddhddh = dde1ddhddh(z, l, dh)
    _dde2dldalpa = dde2dldalpha(z, l, alpha)

    hess[:, 4, 0] = _dfde1*_de1dl  # dadl
    hess[:, 0, 4] = hess[:, 4, 0]

    hess[:, 5, 0] = _dfde1*_de1ddh  # daddh
    hess[:, 0, 5] = hess[:, 5, 0]

    hess[:, 4, 1] = _dgde1*_de1dl  # dbdl
    hess[:, 1, 4] = hess[:, 4, 1]

    hess[:, 5, 1] = _dgde1*_de1ddh  # dbddh
    hess[:, 1, 5] = hess[:, 5, 1]

    hess[:, 4, 2] = _dhde2*_de2dl  # dcdl
    hess[:, 2, 4] = hess[:, 4, 2]

    hess[:, 6, 2] = _dhde2*_de2dalpha  # dcdalpha
    hess[:, 2, 6] = hess[:, 6, 2]

    _temp1 = a*_dfde1 + b*_dgde1
    _temp2 = a*_ddfde1de1 + b*_ddgde1de1

    hess[:, 4, 4] = _temp2*_de1dl**2 + c*_ddhde2de2*_de2dl**2  # dldl
    hess[:, 4, 5] = _temp1*_dde1dlddh + _temp2*_de1dl*_de1ddh  # dlddh
    hess[:, 5, 4] = hess[:, 4, 5]

    hess[:, 4, 6] = c*(_dhde2*_dde2dldalpa + _ddhde2de2*_de2dl*_de2dalpha)  # dldalpha
    hess[:, 6, 4] = hess[:, 4, 6]

    hess[:, 5, 5] = _temp1*_dde1ddhddh + _temp2*_de1ddh**2  # ddhddh

    hess[:, 6, 6] = c*_ddhde2de2*_de2dalpha**2  # dalphadalpha
    return hess

ksi = 1.5
C1 = 1.0 / (2 * ksi)
C2 = 100




def eta1(z, l, dh):
    _eta1 = (z - l) / (C1 * dh)
    return _eta1
def eta2(z, l, alpha):
    _eta2 = (z - alpha * l) / C2
    return _eta2

def f(eta1):
    """
    Definition of the function f of RZse.
    :param eta1: The scaled height coordinate ( (z-l)/(cte*dh) )
    :return: f(eta1)
    """
    _f = (np.tanh(eta1) + 1.0) / 2.0
    return _f
def g(eta1):
    """
    Definition of the function g of RZse
    :param eta1: The scaled height coordinate ( (z-l)/(cte*dh) )
    :return: g(eta1)
    """
    with np.errstate(over='ignore'):  # If g goes to infinite, I display an error message and I ingore it
        _g = (np.log(2 * np.cosh(eta1)) + eta1) / 2.0
    for i in np.where(np.isinf(_g)):  # when g is infinite, I replace it with this (?)
        _g[i] = (np.abs(eta1[i]) + eta1[i]) / 2.0
    return _g
def h(eta2):
    """
    Definition of the function h of RZse
    :param eta2: The scaled height coordinate  ( (z-alpha*l)/100 )
    :return: h(eta2)
    """
    with np.errstate(over='ignore'):
        _h = (eta2 - np.log(np.exp(eta2) + np.exp(-eta2))) / 2
    if type(eta2) == np.float64:
        if np.isinf(_h):
            _h = (eta2 - np.abs(eta2)) / 2
    elif len(eta2) > 1:
        for i in np.where(np.isinf(_h)):
            _h[i] = (eta2[i] - np.abs(eta2[i])) / 2
    else:
        if np.isinf(_h):
            _h = (eta2 - np.abs(eta2)) / 2
    return _h

def dfde1(eta1):
    return (1 - np.tanh(eta1)**2)/2
def dgde1(eta1):
    return (np.tanh(eta1)+1)/2
def dhde2(eta2):
    return (1-np.tanh(eta2))/2

def de1dl(z, l, dh):
    return -1 / (C1 * dh)
def de2dl(z, l, alpha):
    return -alpha/C2
def de1ddh(z, l, dh):
    return (l-z)/(C1*dh**2)
def de2ddh(z, l, alpha):
    return 0
def de1dalpha(z, l, dh):
    return 0
def de2dalpha(z, l, alpha):
    return -l/C2

def ddfde1de1(eta1):
    return -np.tanh(eta1)*(1-np.tanh(eta1)**2)
def ddgde1de1(eta1):
    return (1 - np.tanh(eta1)**2)/2
def ddhde2de2(eta2):
    return (np.tanh(eta2)**2 - 1)/2

def dde1dlddh(z, l, dh):
    return 1/(C1*dh**2)
def dde1ddhddh(z, l, dh):
    return 2*(z-l)/(C1*dh**3)
def dde2dldalpha(z, l, alpha):
    return -1/C1


def A(theta, eta1, eta2, reg_factor=0, reg_weights=None):
    """
    Function that calculates A of the linear system using the Gram matrix property. It is double as efficient as
    A_v1.
    :param theta:
    :param eta1:
    :param eta2:
    :return:
    """
    if reg_weights is None:
        reg_weights = [1, 1, 1, 1]
    _M = np.ones((len(theta), 4))

    _f = f(eta1)
    _g = g(eta1)
    _h = h(eta2)

    _M[:, 1] = _f
    _M[:, 2] = _g
    _M[:, 3] = _h
    return np.matmul(np.transpose(_M), _M) + reg_factor * np.diag(reg_weights)

def b(theta, eta1, eta2):
    _b = np.zeros(4)
    _b[0] = np.sum(theta)
    _b[1] = np.sum(f(eta1) * theta)
    _b[2] = np.sum(g(eta1) * theta)
    _b[3] = np.sum(h(eta2) * theta)
    return _b


def solve_linear_part(theta, eta1, eta2, reg_factor=0, reg_weights=None):
    _A = A(theta, eta1, eta2, reg_factor, reg_weights)
    _b = b(theta, eta1, eta2)
    sol = solve(_A, _b, assume_a='pos', overwrite_a=False, overwrite_b=False)  # TODO: add catch for errors
    return sol


def limited_curve_fit(z, th, p0, bounds, maxfev=100000):
    def RZse_limited(z, l, dh, alpha):
        _eta1 = eta1(z, l, dh)
        _eta2 = eta2(z, l, alpha)

        sol_lin = solve_linear_part(th, _eta1, _eta2)

        thm, a, b, c = sol_lin
        return RZsemodel(z, a, b, c, thm, l, dh, alpha)

    def get_params(z, l, dh, alpha):
        if np.isnan(l):
            return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        else:
            _eta1 = eta1(z, l, dh)
            _eta2 = eta2(z, l, alpha)
            sol_lin = solve_linear_part(th, _eta1, _eta2)

            thm, a, b, c = sol_lin
        return a, b, c, thm, l, dh, alpha

    try:
        popt, pcov = curve_fit(RZse_limited, z, th, p0,  # I fit th with the RZ model
                               bounds=bounds, maxfev=maxfev)
    except RuntimeError:
        print('curve fit failed')
        popt = np.full(p0.shape, np.NaN)
    return get_params(z, popt[0], popt[1], popt[2])

def dh2_v1(par, t=0.5):
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    if np.size(c)>1:
        c[np.where(np.abs(c)<1e-2)] = 1e-2  # For stability
    elif np.abs(c)<1e-2:
        c = 1e-2
    k = np.abs(t/c)
    return np.clip(alpha*l - C2/2 * np.log(np.exp(2*k) - 1), a_min=0, a_max=None)
def dh2_v2(par, p=0.1):
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    _eta0 = -alpha*l/C2
    return alpha*l + C2/2 * np.log(1/((1+np.exp(-2*_eta0))**p - 1))
def dtheta2_v1(par):
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    _eta0 = eta2(0, l, alpha)
    return c*h(_eta0)
def dtheta2_v2(par):
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    return c*np.minimum(-np.log(2)/2, eta2(0, l, alpha))

def _get_par_from_vector(par):
    if not isinstance(par, np.ndarray):
        par = np.array(par)
    [a, b, c, thm, l, dh, alpha] = np.squeeze(np.hsplit(par, 7))
    return [a, b, c, thm, l, dh, alpha]
def _get_par_from_vector_RZ(par):
    if not isinstance(par, np.ndarray):
        par = np.array(par)
    [a, b, thm, l, dh] = np.squeeze(np.hsplit(par, 5))
    return [a, b, thm, l, dh]
def physical_parameters(par, version=1, threshold=0.5):
    """
    Get the parameters from the RZSE fit and return the physical parameters of the profile
    :param par: list = [a, b, c, thm, l, dh, alpha]
    :return: dictionary with
    'thm' = temperature shift of profile (thm)
    'BLheight' = boundary layer height (l)
    'CIwidth' = capping inversion width (dh)
    'CIstrength' = capping inversion strength (dtheta)
    'gamma' = free lapse rate
    'SLheight' = surface layer height
    'SLstrength' = surface layer strength
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    physical = {}
    physical['thm'] = thm
    physical['blh'] = l
    physical['CIwidth'] = dh
    physical['CIstrength'] = a + b/(2*C1)
    physical['gamma'] = b/(C1*dh)
    if version==1:
        physical['SLheight'] = dh2_v1(par, t=threshold)
        physical['SLstrength'] = dtheta2_v1(par)
    elif version==2:
        physical['SLheight'] = dh2_v2(par)
        physical['SLstrength'] = dtheta2_v2(par)
    return physical
def physical_parameters_RZ(par):
    [a, b, thm, l, dh] = _get_par_from_vector_RZ(par)
    physical = {}
    physical['thm'] = thm
    physical['blh'] = l
    physical['CIwidth'] = dh
    physical['CIstrength'] = a + b / (2 * C1)
    physical['gamma'] = b / (C1 * dh)
    return physical

