import time
import numpy as np
from numba import jit, float64
import ctypes
from numba.extending import get_cython_function_address


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

lgamma = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    get_cython_function_address('scipy.special.cython_special', 'gammaln'))
digamma = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1psi'))

@jit(nopython=True,fastmath=True)
def ll_POIS(params,Wij,exogenous_variables):
    """Computes opposite of the  log-likelihood for Poisson model
    """
    
    ll = 0
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
    x_beta = exogenous_variables @ params
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                ll += Wij[ij]*x_beta[ij]- np.exp(x_beta[ij]) - lgamma(Wij[ij]+1)
    return - ll

@jit(nopython=True,fastmath=True)
def ll_POIS_binary(params,Wij,exogenous_variables):
    """Computes opposite of the  binary log-likelihood for Poisson model
    """
    
    ll = 0
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
    x_beta = exogenous_variables @ params
    mu = np.exp(x_beta)
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                pij = 1.-np.exp(-mu[ij])
                aij = Aij[ij]
                ll += aij * np.log(pij) + (1.-aij)*np.log(1.-pij)
    return - ll

@jit(nopython=True,fastmath=True)
def jac_POIS(params,Wij,exogenous_variables):
    """Computes opposite of the jacobian for Poisson model
    """
    
    jac = np.zeros(len(params))
    n_obs = len(Wij)
    x_beta = exogenous_variables @ params
    error_matrix = np.zeros(len(Wij))
    n_countries = int(np.sqrt(len(Wij)))
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                error_matrix[ij] = Wij[ij] - np.exp(x_beta)[ij]
    
    
    jac = exogenous_variables.T @ error_matrix
                
    return - jac

@jit(nopython=True,fastmath=True)
def ll_NB2(params,Wij,exogenous_variables):
    """Computes opposite of the  log-likelihood for Negative Binomial model
    """
    
    params_exog = params[:-1]
    x_beta = exogenous_variables @ params_exog
    mu = np.exp(x_beta)
    alpha = params[-1]
    m = 1/alpha
    n_countries = int(np.sqrt(len(Wij)))
    
    n_obs = len(Wij)
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                yi = Wij[ij]
                gamma_part = lgamma(m+yi) - lgamma(yi+1) - lgamma(m) 
                exog_part = -m*np.log(1+alpha*mu[ij])  + yi*np.log(mu[ij]) - yi * np.log(1+alpha*mu[ij])
                ll += gamma_part + exog_part + yi*np.log(alpha)
                        
    return - ll

@jit(nopython=True,fastmath=True)
def ll_NB2_binary(params,Wij,exogenous_variables):
    """Computes opposite of the binary log-likelihood for Negative Binomial model
    """
    
    params_exog = params[:-1]
    x_beta = exogenous_variables @ params_exog
    mu = np.exp(x_beta)
    alpha = params[-1]
    m = 1/alpha
    n_countries = int(np.sqrt(len(Wij)))
    Aij = binarize(Wij)
    n_obs = len(Wij)
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                yi = Wij[ij]
                ti = np.power(1./(1.+alpha*mu[ij]),m)
                pij = 1.-ti
                ai = Aij[ij]
                ll += ai*np.log(pij) + (1.-ai)*np.log(1.-pij)
                        
    return - ll


@jit(nopython=True,fastmath=True)
def jac_NB2(params,Wij,exogenous_variables):
    """Computes opposite of the  jacobian for Negative Binomial model
    """
    
    params_exog = params[:-1]
    alpha = params[-1]
    m = 1/alpha
    
    x_beta = exogenous_variables @ params_exog
    mu = np.exp(x_beta)
    auxiliar = np.ones(len(mu))
    auxiliar += alpha*mu
    jac = np.zeros(len(params))
    n_countries = int(np.sqrt(len(Wij)))
    error_matrix = np.zeros(len(Wij))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                chi = auxiliar[ij]
                yi = Wij[ij]
                error_matrix[ij] = (Wij[ij]-mu[ij])/auxiliar[ij]
                jac[-1] += -m**2 * (digamma(m+yi) - digamma(m)) + m**2 * np.log(chi) + (yi-mu[ij])*m/chi             
    
    jac[:-1] = exogenous_variables.T @ error_matrix 
    
    return - jac




@jit(nopython=True,fastmath=True)
def guess_constant_parameter(Wij,exogenous_variables):
    """Estimate approximate first parameter for the solver of Poisson model
    """
    exog_no_constant = np.zeros((exogenous_variables.shape[0],exogenous_variables.shape[1]-1))
    exog_no_constant[:,:] = exogenous_variables[:,1:]
    one_parameters = np.ones(exog_no_constant.shape[1])
    exog = exog_no_constant @ one_parameters
    W_tot = 0
    n_countries = int(np.sqrt(len(Wij)))
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                W_tot += Wij[ij]
                
    exog_sum = exog.sum()

    beta_0 = np.log(W_tot) - np.log(exog_sum)
    return beta_0


@jit(nopython=True,fastmath=True)
def ll_ZIP(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the log-likelihood for Zero-Inflated Poisson model"""
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:]

    params_select = np.concatenate((params_select,fixed_selection_params))


    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)

    ll = 0
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]

                ll += (1-ai)*np.log(1+Gi[ij]*np.exp(-mu[ij])) - np.log(1+Gi[ij]) + yi*x1_beta1[ij] - ai*mu[ij] - ai*lgamma(yi+1)

    return - ll

@jit(nopython=True,fastmath=True)
def ll_ZIP_binary(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the binary log-likelihood for Zero-Inflated Poisson model"""
    
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:]

    params_select = np.concatenate((params_select,fixed_selection_params))


    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)

    ll = 0
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]
                pij = Gi[ij]/(1.+Gi[ij])*(1.-np.exp(-mu[ij]))
                ll += ai*np.log(pij) + (1.-ai)*np.log(1.-pij)
        
    return - ll

    

@jit(nopython=True,fastmath=True)
def jac_ZIP(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the jacobian for Zero-Inflated Poisson model"""
    
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:]
    
    params_select = np.concatenate((params_select,fixed_selection_params))

    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)

    jac = np.zeros(len(params))
    Aij = binarize(Wij)
    n_obs = len(Aij)
    G_matrix = np.zeros(len(Aij))
    mu_matrix = np.zeros(len(Aij))
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]

                G_matrix[ij] = (1-ai)*Gi[ij]*np.exp(-mu[ij])/(1+Gi[ij]*np.exp(-mu[ij])) + ai - Gi[ij]/(1+Gi[ij])
                aux = ai + (1-ai)/(1+Gi[ij]*np.exp(-mu[ij]))
                mu_matrix[ij] = yi - mu[ij]*aux


    
    jac[:n_select_variables] = (selection_variables.T @ G_matrix)[:n_select_variables]
    jac[n_select_variables:] = exogenous_variables.T @ mu_matrix


    return - jac


@jit(nopython=True,fastmath=True)
def ll_ZINB(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the log-likelihood for Zero-Inflated Negative Binomial model"""
    
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:-1]
    au = params[-1]
    m = 1/au
    params_select = np.concatenate((params_select,fixed_selection_params))


    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)
    
    ll = 0
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]
                chi = 1 + au*mu[ij]
                ti = (1/chi)**m
                
                ll += - np.log(1+Gi[ij]) + (1-ai)* np.log(1+Gi[ij]*ti) + ai*np.log(Gi[ij])
                ll += ai*( lgamma(m+yi) - lgamma(yi+1) - lgamma(m) - m*np.log(chi))
                ll += yi*( np.log(au) + np.log(mu[ij]) - np.log(chi))
    return - ll 



@jit(nopython=True,fastmath=True)
def ll_ZINB_binary(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the binary log-likelihood for Zero-Inflated Negative Binomial model"""
    
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:-1]
    au = params[-1]
    m = 1/au
    params_select = np.concatenate((params_select,fixed_selection_params))


    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)
    
    ll = 0
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]
                chi = 1 + au*mu[ij]
                ti = (1/chi)**m
                pij = Gi[ij]/(1+Gi[ij]) * (1.-ti)

                ll += ai*np.log(pij) + (1.-ai)*np.log(1-pij)
    return - ll 



@jit(nopython=True,fastmath=True)
def jac_ZINB(params,Wij,selection_variables,exogenous_variables,fixed_selection_params = np.array([])):
    """Compute the opposite of the jacobian for Zero-Inflated Negative Binomial model"""
    
    n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
    n_exog_variables = exogenous_variables.shape[1]
    params_select = params[:n_select_variables]
    params_exog = params[n_select_variables:-1]
    au = params[-1]
    m = 1./au
    params_select = np.concatenate((params_select,fixed_selection_params))


    x0_beta0 = selection_variables @ params_select
    x1_beta1 = exogenous_variables @ params_exog

    Gi = np.exp(x0_beta0)
    mu = np.exp(x1_beta1)
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
    jac = np.zeros(len(params))

    G_matrix = np.zeros(len(Aij))
    mu_matrix = np.zeros(len(Aij))
    alpha_optimization = 0
    n_countries = int(np.sqrt(len(Wij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j

                yi = Wij[ij]
                ai = Aij[ij]
                chi = 1 + au*mu[ij]
                ti = (1/chi)**m
                xi = m*mu[ij]/(1+au*mu[ij]) - m**2 * np.log(1+au*mu[ij])
                
                
                alpha_optimization += ai*(digamma(m+yi)- digamma(m))*(-m**2)
                alpha_optimization += m*yi/(1+au*mu[ij]) - xi*(1+ai*Gi[ij]*ti)/(1+Gi[ij]*ti)
                
                G_matrix[ij] = (ai - Gi[ij]*(1-ti)/(1+Gi[ij]))/(1+Gi[ij]*ti)
                mu_matrix[ij] = (yi-mu[ij]*(ai+Gi[ij]*ti)/(1+Gi[ij]*ti))/(1+au*mu[ij])
            
    jac[:n_select_variables] = (selection_variables.T @ G_matrix)[:n_select_variables]
    jac[n_select_variables:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = alpha_optimization

    return - jac

