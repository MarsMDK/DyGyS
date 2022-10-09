import numpy as np
from numba import jit, float64, prange, objmode, typeof
import math
from numba.typed import List
import ctypes
from scipy.stats import norm
from numba.extending import get_cython_function_address

lgamma = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    get_cython_function_address('scipy.special.cython_special', 'gammaln'))
digamma = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1psi'))

@jit(nopython=True,fastmath=True)
def binarize(input):
    """Computes binary adjacency matrix from weighted matrix.
    """
    dim = len(input)
    byn = np.zeros(dim)
    for i in range(dim):
        if input[i]> 0:
            byn[i] = 1.
    return byn 

@jit(nopython=True,fastmath=True)
def ll_logit(params,Wij,selection_variables,fixed_selection_params = np.array([])):
    """Computes opposite of the  log-likelihood for Logit model
    """
    Aij = binarize(Wij)
    n_params = len(params)
    params = np.concatenate((params,fixed_selection_params))
    Gi = np.exp(selection_variables @ params)
    
    n_countries = int(np.sqrt(len(Wij)))
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                Gij = Gi[ij]

                ll += aij*np.log(Gij) - np.log(1+Gij)
    
    return - ll

@jit(nopython=True,fastmath=True)
def jac_logit(params,Wij,selection_variables,fixed_selection_params = np.array([])):
    """Computes opposite of the  jacobian for Logit model"""
    
    Aij = binarize(Wij)
    n_params = len(params)
    params = np.concatenate((params,fixed_selection_params))
    Gi = np.exp(selection_variables @ params)
    
    G_matrix = np.zeros(len(Aij))
    n_countries = int(np.sqrt(len(Wij)))
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ai = Aij[ij]
                G_matrix[ij] = ai - Gi[ij]/(1+Gi[ij])
    
    jac = (selection_variables.T @ G_matrix)[:n_params]

    return - jac
    
@jit(nopython=True,fastmath=True)
def ll_BCM(params,Wij):
    """Computes opposite of the  log-likelihood for Undirected Binary Configuration Model """
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))

    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                xij = np.exp(-params[i]-params[j])

                ll += aij*np.log(xij) - np.log(1+xij)
    return - ll

@jit(nopython=True,fastmath=True)
def jac_BCM(params,Wij):
    """Computes opposite of the  jacobian for Undirected Binary Configuration Model """
    
    Aij = binarize(Wij)
    
    n_countries = int(np.sqrt(len(Aij)))
    jac = np.zeros(len(params))
    for i in range(n_countries):
        
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                xij = np.exp(-params[i]-params[j])

                jac[i]+= xij/(1+xij) - aij
    return - jac

@jit(nopython=True,fastmath=True)
def hess_BCM(params,Wij):
    """Computes opposite of the  hessian for Undirected Binary Configuration Model """
    
    Aij = binarize(Wij)
    
    n_countries = int(np.sqrt(len(Aij)))
    hess = np.zeros((len(params),len(params)))
    for i in range(n_countries):
        
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                xij = np.exp(-params[i]-params[j])
                pij = xij/(1.+xij)
                hess[i,i] += - pij*(1.-pij)
                hess[i,j] = -pij*(1.-pij)
                
                
    return - hess

@jit(nopython=True,fastmath=True)
def ll_DBCM(params,Wij):
    """Computes opposite of the  log-likelihood for Directed Binary Configuration Model"""
    
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))
    params_out = params[:n_countries]
    params_in = params[n_countries:]
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                xij = np.exp(-params_out[i]-params_in[j])

                ll += aij*np.log(xij) - np.log(1+xij)
    return - ll

@jit(nopython=True,fastmath=True)
def jac_DBCM(params,Wij):
    """Computes opposite of the  jacobian for Directed Binary Configuration Model"""
    
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))
    
    params_out = params[:n_countries]
    params_in = params[n_countries:]
    
    jac = np.zeros(len(params))
    for i in range(n_countries):
        
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ji = j*n_countries + i
                aij = Aij[ij]
                aji = Aij[ji]
                
                xij = np.exp(-params_out[i]-params_in[j])
                xji = np.exp(-params_out[j]-params_in[i])
                jac[i]+= xij/(1+xij) - aij
                jac[i+n_countries]+= xji/(1.+xji) - aji
    return - jac

@jit(nopython=True,fastmath=True)
def hess_DBCM(params,Wij):
    """Computes opposite of the  hessian for Directed Binary Configuration Model"""
    
    Aij = binarize(Wij)
    
    n_countries = int(np.sqrt(len(Aij)))
    hess = np.zeros((len(params),len(params)))
    params_out = params[:n_countries]
    params_in = params[n_countries:]
    
    for i in range(n_countries):
        
        for j in range(n_countries):
            if j==i:
                for k in range(n_countries):
                    if k!=i:
                        xik = np.exp(-params_out[i]-params_in[k])
                        xki = np.exp(-params_in[i]-params_out[k])
            
                        pij = xik/(1.+xik)
                        pji = xki/(1.+xki)
                         
                        hess[i,j] += - pij*(1.-pij)
                        hess[i+n_countries,j+n_countries] += - pji*(1.-pji) 
            elif j!=i:
                xij = np.exp(-params_out[i]-params_in[j])
                xji = np.exp(-params_out[j]-params_in[i])
                pij = xij/(1.+xij)
                pji = xji/(1.+xji)
                
                hess[i,j+n_countries] = - pij*(1.-pij)
                hess[i+n_countries,j] = -pji*(1.-pji)
                
                
    return - hess

@jit(nopython=True,fastmath=True)
def ll_CGeom(params,Wij,exogenous_variables):
    "Computes the opposite of the  log-likelihood for the weighted part of the Conditional Geometric model"
    params_exog = params[:-1]
    z_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                denominator = 1 + zi - z_0*zi
                if ai == 1:
                    ll += (yi-ai)*np.log(z_0*zi/(1+zi)) + ai*np.log(denominator/(1+zi))
                
    return - ll 

    


@jit(nopython=True,fastmath=True)
def jac_CGeom(params,Wij,exogenous_variables):
    "Computes the opposite of the  jacobian for the weighted part of the Conditional Geometric model"
    
    params_exog = params[:-1]
    
    z_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    jac = np.zeros(len(params))
    n_obs = len(Aij)
    mu_matrix = np.zeros(n_obs)
    adder = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    zi = mu[ij]
                    denominator = (1+zi-z_0*zi)
                    yi_hat = (1+zi)/denominator

                    mu_matrix[ij] = yi/(1+zi) - ai/denominator
                    adder += - yi + ai*yi_hat
        
        
    jac[:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder
    
    return - jac


@jit(nopython=True,fastmath=True)
def hess_CGeom(params,Wij,exogenous_variables):
    "Computes the opposite of the  hessian for the weighted part of the Conditional Geometric model"
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[:-1]
    beta_0 = params[-1]
    
    z_0 = np.exp(-beta_0)
    x1_beta1 = exogenous_variables @ params_exog
    mu = np.exp(x1_beta1)
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
        
    
    hess = np.zeros((len(params),len(params)))

    W_beta_1 = np.zeros((n_obs,n_obs))
    W_beta_0 = np.zeros(n_obs)
    W_beta_mixed = np.zeros(n_obs)

    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                ai = Aij[ij]
                yi = Wij[ij]
                zi = mu[ij]
                denominator = 1 + zi - z_0*zi
                yi_hat = (1+zi)/denominator
                
                W_beta_0[ij] = - ai*zi*z_0*(1+zi)/denominator**2

                W_beta_1[ij,ij] = - zi*(yi-ai*yi_hat)/(1+zi)**2 + ai*z_0/((denominator**2)*(1+zi))
                W_beta_mixed[ij] = - ai*z_0*zi/denominator**2

    
    hess[:n_exog_variables,:n_exog_variables] = exogenous_variables.T @ W_beta_1 @ exogenous_variables
    hess[-1,:n_exog_variables] = (exogenous_variables.T @ W_beta_mixed)
    hess[:n_exog_variables,-1] = (exogenous_variables.T @ W_beta_mixed)
    hess[-1,-1] = W_beta_0.sum()

    return - hess


@jit(nopython=True,fastmath=True)
def ll_CExp(params,Wij,exogenous_variables):
    "Computes the  opposite of the  log-likelihood for the weighted part of the Conditional Exponential model"
    
    params_exog = params[:-1]
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    zi = mu[ij]
                    ll += ai*np.log(1.+beta_0*zi) - ai* np.log(zi) - yi*(beta_0+1./zi)
                    
    return - ll 

    


@jit(nopython=True,fastmath=True)
def jac_CExp(params,Wij,exogenous_variables):
    "Computes the opposite of the  jacobian for the weighted part of the Conditional Exponential model"
    
    params_exog = params[:-1]
    
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    jac = np.zeros(len(params))
    n_obs = len(Aij)
    mu_matrix = np.zeros(n_obs)
    adder = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    zi = mu[ij]
                    yi_hat = zi/(1+beta_0*zi)

                    mu_matrix[ij] = (yi-ai*yi_hat)/zi
                    adder += - yi + ai*yi_hat
                    
        
    jac[:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder
    
    return - jac


    


@jit(nopython=True,fastmath=True)
def hess_CExp(params,Wij,exogenous_variables):
    "Computes the opposite of the  hessian for the weighted part of the Conditional Exponential model"
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[:-1]
    beta_0 = params[-1]
    
    
    x1_beta1 = exogenous_variables @ params_exog
    mu = np.exp(x1_beta1)
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
        
    
    hess = np.zeros((len(params),len(params)))

    W_beta_1 = np.zeros((n_obs,n_obs))
    W_beta_mixed = np.zeros(n_obs)

    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                ai = Aij[ij]
                yi = Wij[ij]
                zi = mu[ij]
                denominator = 1 + beta_0*zi
                
                hess[-1,-1] += - ai*zi**2/denominator**2

                W_beta_1[ij,ij] = - yi/zi + ai*beta_0*zi/denominator**2
                W_beta_mixed[ij] = - ai*zi/denominator**2

    
    hess[:n_exog_variables,:n_exog_variables] = exogenous_variables.T @ W_beta_1 @ exogenous_variables
    hess[-1,:n_exog_variables] = (exogenous_variables.T @ W_beta_mixed)
    hess[:n_exog_variables,-1] = (exogenous_variables.T @ W_beta_mixed)
    
    return - hess



@jit(nopython=True,fastmath=True)
def ll_CPareto(params_exog,Wij,exogenous_variables):
    "Computes the opposite of the  log-likelihood for the weighted part of the Conditional Pareto model"
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_countries = int(np.sqrt(len(Aij)))
    
    positive_weights = List()
    for i in range(len(Wij)):
        wij = Wij[i]
        if wij > 0:
            positive_weights.append(wij)
    w_min = min(positive_weights)
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                if ai == 1:
                    ll += (np.log(1+zi) - np.log(zi) + (1.+1./zi)*np.log(w_min) - (2.+1./zi) * np.log(yi))
                
                
    return - ll 



@jit(nopython=True,fastmath=True)
def jac_CPareto(params_exog,Wij,exogenous_variables):
    "Computes the opposite of the jacobian for the weighted part of the Conditional Pareto model"
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    jac = np.zeros(len(params_exog))
    n_obs = len(Aij)
    mu_matrix = np.zeros(n_obs)
    n_countries = int(np.sqrt(len(Aij)))
    positive_weights= List()
    for ij in range(len(Wij)):
        weight = Wij[ij]
        if weight>0:
            positive_weights.append(weight)
    w_min = min(positive_weights)
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                lnyi_hat = np.log(w_min) + zi/(1.+zi)
                if ai == 1:
                    mu_matrix[ij] = (np.log(yi)-ai*lnyi_hat)/zi
                
                
        
    jac = exogenous_variables.T @ mu_matrix
    
    return - jac



@jit(nopython=True,fastmath=True)
def ll_CGamma(params,Wij,exogenous_variables):
    "Computes the opposite of the  log-likelihood for the weighted part of the Conditional Gamma model"
    
    params_exog = params[:-2]
    beta_0 = params[-2]
    phi = params[-1]
    x1_beta1 = exogenous_variables @ params_exog
    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    zi = mu[ij]
                    ll += -np.log(yi) + phi*np.log(yi) + phi*np.log(beta_0 + 1./zi) - lgamma(phi) - (beta_0 + 1./zi)*yi
    return - ll 

@jit(nopython=True,fastmath=True)
def jac_CGamma(params,Wij,exogenous_variables):
    "Computes the opposite of the  jacobian for the weighted part of the Conditional Gamma model"
    
    params_exog = params[:-2]
    beta_0 = params[-2]
    phi = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    jac = np.zeros(len(params))
    n_obs = len(Aij)
    mu_matrix = np.zeros(n_obs)
    d_phi = 0
    d_beta_0 = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    zi = mu[ij]
                    yi_hat = phi*zi/(1.+beta_0*zi) 
                    lnyi_hat = - np.log(beta_0 + 1./zi) + digamma(phi)
                    
                    mu_matrix[ij] = (yi - ai*yi_hat)/zi
                    d_phi += np.log(yi) - lnyi_hat
                    d_beta_0  += -yi + yi_hat 
                
        
    jac[:-2] = exogenous_variables.T @ mu_matrix
    jac[-2] = d_beta_0
    jac[-1] = d_phi
    
    return - jac





@jit(nopython=True,fastmath=True)
def ll_CLognormal(params,Wij,exogenous_variables):
    "Computes the opposite of the  log-likelihood for the weighted part of the Conditional Lognormal model"
    
    params_exog = params[:-1]
    gamma_0 = params[-1]
    mu = exogenous_variables @ params_exog   
    
    Aij = binarize(Wij)
    ll = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ai = Aij[ij]
                if ai == 1:
                    
                    yi = Wij[ij]
                
                    log_yi = np.log(yi)
                    lnzi = mu[ij]

                    xi_ij = 1.-lnzi
                    ll+= - xi_ij**2/(4.*gamma_0) - xi_ij*log_yi - gamma_0*log_yi**2
                    ll+= -.5*np.log(math.pi/gamma_0) - (1.-2.*xi_ij)/(4.*gamma_0)

    return - ll 



@jit(nopython=True,fastmath=True)
def jac_CLognormal(params,Wij,exogenous_variables):
    "Computes the opposite of the  jacobian for the weighted part of the Conditional Lognormal model"
    
    params_exog = params[:-1]
    
    gamma_0 = params[-1]
    mu = exogenous_variables @ params_exog 
    
    Aij = binarize(Wij)
    jac = np.zeros(len(params))
    n_obs = len(Aij)
    mu_matrix = np.zeros(n_obs)
    adder = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                if ai == 1:
                    
                    lnzi = mu[ij]
                    log_yi = np.log(yi)
                    log_yi_hat = lnzi/(2.*gamma_0)
                    log_yi2_hat = 1./(2.*gamma_0) + log_yi_hat**2 
                    
                    mu_matrix[ij] = log_yi - log_yi_hat
                    adder += log_yi2_hat - log_yi**2.
                
    jac[:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder
    
    return - jac

@jit(nopython=True,fastmath=True)
def alpha_estimate_CLognormal(params_exog,Wij,exogenous_variables):
    """Computes an approximate estimate for the last parameter of the Lognormal model given the others."""
    mu = exogenous_variables @ params_exog
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))
    count = 0
    rhs = 0
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                if ai == 1:
                    count += 1
                    log_yi = np.log(yi)
                    rhs += (log_yi - zi)**2
                    
    sigma2 = rhs/(count)
    alpha = 1./(2.*sigma2)                
    return alpha




