import time
import numpy as np
from numba import jit, float64
import math


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
def ll_LIGeom(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the L-constrained Integrated Geometric model"""
    params_exog = params[1:-1]
    
    x_0 = np.exp(-params[0])
    z_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                yi = Wij[ij]
                ai = Aij[ij]

                ll += yi*np.log(z_0) + ai*np.log(x_0)
                ll += yi * np.log(mu[ij]) - yi * np.log(1+mu[ij]) + np.log(1+ mu[ij] - z_0*mu[ij]) - np.log(1 + mu[ij] - z_0*mu[ij]+ x_0*z_0*mu[ij])

                
    return - ll 


@jit(nopython=True,fastmath=True)
def ll_LIGeom_binary(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the L-constrained Integrated Geometric model"""
    
    params_exog = params[1:-1]
    
    x_0 = np.exp(-params[0])
    z_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    zij = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                yi = Wij[ij]
                ai = Aij[ij]
                pij = x_0 * z_0 * zij[ij]/(1+zij[ij]-z_0*zij[ij]+x_0*z_0*zij[ij])
                ll += ai*np.log(pij) + (1.-ai)*np.log(1.-pij)
                
                
    return - ll 

    


@jit(nopython=True,fastmath=True)
def jac_LIGeom(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the L-constrained Integrated Geometric model"""
    
    params_exog = params[1:-1]
    theta_0 = params[0]
    x_0 = np.exp(-theta_0)
    z_0 = params[-1]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)

    n_obs = len(Aij)
    jac = np.zeros(len(params))

    mu_matrix = np.zeros(len(Aij))
    
    adder_0 = 0
    adder_last = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                pi = x_0 * z_0 * zi/(1+zi - z_0*zi + x_0*z_0*zi)
                den_2 = (1+zi-z_0*zi)*(1+zi-z_0*zi+x_0*z_0*zi)
                yi_hat = x_0*z_0*zi*(1+zi)/den_2

                adder_0 += pi
                adder_last += yi_hat
                mu_matrix[ij] = (yi-yi_hat)/(1+zi)
                
    jac[0] = adder_0 - Aij.sum()
    jac[1:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = (adder_last - Wij.sum())
    
    return - jac




@jit(nopython=True,fastmath=True)
def hess_LIGeom(params,Wij,exogenous_variables):
    """Computes the opposite of the hessian for the L-constrained Integrated Geometric model"""
    
    n_exog_variables = exogenous_variables.shape[1]
    n_countries = int(np.sqrt(len(Wij)))
    params_exog = params[1:-1]
    theta_0 = params[0]
    x_0 = np.exp(-theta_0)
    z_0 = params[-1]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    Aij = binarize(Wij)
    n_obs = len(Aij)
    hess = np.zeros((len(params),len(params)))
    
    W_matrix = np.zeros((len(Wij),len(Wij)))
    aux_theta_zij = np.zeros(len(Wij))
    aux_zij_z_0 = np.zeros(len(Wij))
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                aij = Aij[ij]
                wij = Wij[ij]
                zij = mu[ij]
                pij = x_0*z_0*zij/(1.+zij-z_0*zij+x_0*z_0*zij)
                den_1 = (1.+zij-z_0*zij+x_0*z_0*zij)
                den_2 = (1.+zij-z_0*zij)*(1.+zij-z_0*zij+x_0*z_0*zij)
                yi_hat = x_0*z_0*zij*(1+zij)/den_2

                d_den_2_z_0 = -zij*(1.+zij-z_0+x_0*z_0*zij) + (1.+zij-z_0*zij)*(-zij + x_0 * zij)
                d_den_2_zij = ((1.-z_0)*(1.+zij-z_0+x_0*z_0*zij) + (1.+zij-z_0*zij) * (1.- z_0 + x_0*z_0))*zij


                hess[0,0] += - pij*(1.-pij)
                aux_theta_zij[ij] = x_0*z_0*zij/(den_1)**2 
                hess[0,-1] += - x_0*z_0*zij/(den_1)**2
                hess[-1,0] += - x_0*z_0*zij/(den_1)**2
                
                num_W_matrix = x_0*z_0*den_2 - x_0*z_0*zij*d_den_2_zij
                W_matrix[ij,ij] = (- wij/(1+zij)**2 -  num_W_matrix/den_2**2)*zij

                num_zij_z_0 = x_0*zij*den_2 - x_0*z_0*zij*d_den_2_z_0
                aux_zij_z_0[ij] = - z_0*num_zij_z_0/den_2**2

                num_z_0_z_0 = x_0*zij*(1+zij)*den_2 - x_0*z_0*zij*(1+zij)*d_den_2_z_0
                hess[-1,-1] += - z_0*num_z_0_z_0/den_2**2

    hess[0,1:-1] = exogenous_variables.T @ aux_theta_zij
    hess[1:-1,0] = exogenous_variables.T @ aux_theta_zij
    hess[1:-1,1:-1] = exogenous_variables.T @ W_matrix @ exogenous_variables
    hess[-1,1:-1] = exogenous_variables.T @ aux_zij_z_0
    hess[1:-1,-1] = exogenous_variables.T @ aux_zij_z_0
    
    return - hess
    






@jit(float64(float64[:],float64[:],float64[:,:]),nopython=True,fastmath=False)
def ll_kIGeom_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the k-constrained Integrated Geometric model for Undirected networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[n_countries:-1]
    z_0 = params[-1]
    theta_i = params[:n_countries]
    x_i = np.exp(-theta_i)

    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params_exog = np.ascontiguousarray(params_exog)
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_i[i]*x_i[j]
                ll += ai*np.log(xij)  + yi*np.log(z_0) 
                ll += yi * np.log(mui) - yi * np.log(1.+mui) + np.log(1.+ mui - z_0*mui) - np.log(1. + mui - z_0*mui+ xij*z_0*mui)

    return - ll 

@jit(nopython=True,fastmath=True)
def ll_kIGeom_binary_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the k-constrained Integrated Geometric model for Undirected networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[n_countries:-1]
    z_0 = params[-1]
    theta_i = params[:n_countries]
    x_i = np.exp(-theta_i)

    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_i[i]*x_i[j]
                ll += ai*np.log(xij*z_0*mui) + (1.-ai)*np.log(1+ mui - z_0*mui) - np.log(1 + mui - z_0*mui+ xij*z_0*mui)

    return - ll 
    


@jit(float64[:](float64[:],float64[:],float64[:,:]),nopython=True,fastmath=True)
def jac_kIGeom_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the k-constrained Integrated Geometric model for Undirected networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[n_countries:-1]
    
    
    theta_i = params[:n_countries]
    x_i = np.exp(-theta_i)
    z_0 = params[-1]
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params_exog = np.ascontiguousarray(params_exog)
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(params))
    
    mu_matrix = np.zeros(len(Aij))
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_i[i]*x_i[j]
                pi = xij*z_0*mui/(1.+mui-z_0*mui+xij*z_0*mui)
                jac[i] += - ai + pi

                
                denominator_mu = (1.+mui-z_0*mui+xij*z_0*mui) * (1.+mui-z_0*mui)
                mu_matrix[ij] = yi/(1.+mui) - xij*z_0*mui/denominator_mu

                adder_last += - yi  + xij*z_0*mui*(1.+mui)/denominator_mu
    
    jac[n_countries:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder_last
    return - jac

@jit(float64[:](float64[:],float64[:],float64[:,:],float64[:],float64[:]),nopython=True,fastmath=True)
def jac_kIGeom_topological_undirected(theta_i,Wij,exogenous_variables,params_exog,z_0):
    """Computes the opposite of the topological jacobian for the k-constrained Integrated Geometric model for Undirected networks.
    Used in the solver."""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    
    z_0 = z_0[0]
    x_i = np.exp(-theta_i)
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params_exog = np.ascontiguousarray(params_exog)
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    jac = np.zeros(len(x_i))
    
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_i[i]*x_i[j]
                jac[i] +=  - ai + xij*z_0*mui/(1+mui-z_0*mui+xij*z_0*mui)
    
    return - jac




@jit(float64[:,:](float64[:],float64[:],float64[:,:]),nopython=True,fastmath=True)
def hess_kIGeom_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the hessian for the k-constrained Integrated Geometric model for Undirected networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    theta = params[:n_countries]
    params_exog = params[n_countries:-1]
    z_0 = params[-1]
    
    Aij = binarize(Wij)
    hess = np.zeros((len(params),len(params)))
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params_exog = np.ascontiguousarray(params_exog)
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    W_matrix = np.zeros((len(Wij),len(Wij)))
    aux_theta_zij = np.zeros(len(Wij))
    aux_zij_z_0 = np.zeros(len(Wij))
    for i in range(n_countries):
        for j in range(n_countries):
            
            if j!=i:
                ij = i*n_countries+j
                xij = np.exp(-theta[i]-theta[j])
                
                aij = Aij[ij]
                wij = Wij[ij]
                zij = mu[ij]
                
                
                den_2 = (1+zij-z_0*zij)*(1+zij-z_0*zij+xij*z_0*zij)
                d_den_2_z_0 = - 2*zij + xij*zij - 2*zij**2 + xij*zij**2 +2*z_0*zij**2 - 2*xij*z_0*zij**2
                d_den_2_zij = 2 -2*z_0 + xij*z_0 + 2*zij - 4*z_0*zij + 2*xij*z_0*zij + 2*z_0**2 * zij - 2*xij*z_0**2 * zij
                
                diff_d_den_z_0 = xij*z_0*zij**2 + 1 + zij**2 -z_0**2 * zij**2 + 2*zij
                diff_d_den_zij = 1 - zij**2 - (z_0 * zij)**2 + 2*z_0*zij**2 - xij*z_0*zij**2 + xij*(z_0 * zij)**2
                
                
                pij = xij * zij * z_0/(1+zij -z_0*zij + xij*z_0*zij)
                wij_hat = xij*zij*z_0*(1+zij) /den_2
                
                d_a_theta_i_theta_i = - pij*(1-pij)
                d_a_theta_i_theta_j = - pij*(1-pij)
                d_a_theta_i_zij = -xij*z_0/(1+zij -z_0*zij + xij*z_0*zij)**2
                d_a_theta_i_b0 = + xij*zij*(1+zij)*z_0/(1+zij -z_0*zij + xij*z_0*zij)**2
                hess[i,i] += d_a_theta_i_theta_i
                hess[i,j] = d_a_theta_i_theta_j
                hess[j,i] = d_a_theta_i_theta_j
                hess[i,-1] += d_a_theta_i_b0 
                hess[-1,i] += d_a_theta_i_b0
                
                
                aux_theta_zij[ij] = d_a_theta_i_zij*zij 
                d_a_phi_zij = - wij/(1+zij)**2 - xij*z_0*diff_d_den_zij/den_2**2
                d_a_phi_b0 = - xij*zij*z_0*diff_d_den_z_0/den_2**2

                d_a_b0_b0 = - xij*zij*z_0*(1+zij)*diff_d_den_z_0/den_2**2
                
                W_matrix[ij,ij] = d_a_phi_zij*zij
                aux_zij_z_0[ij] = d_a_phi_b0
                hess[-1,-1] += d_a_b0_b0

    for i in range(n_countries):
        hess[i,n_countries:-1] = exogenous_variables.T @ aux_theta_zij
        hess[n_countries:-1,i] = exogenous_variables.T @ aux_theta_zij       
    hess[n_countries:-1,n_countries:-1] = exogenous_variables.T @ W_matrix @ exogenous_variables
    hess[-1,n_countries:-1] = exogenous_variables.T @ aux_zij_z_0
    hess[n_countries:-1,-1] = exogenous_variables.T @ aux_zij_z_0
                

                                 
    return - hess




@jit(float64[:,:](float64[:],float64[:],float64[:,:],float64[:],float64[:]),nopython=True,fastmath=True)
def hess_kIGeom_topological_undirected(theta,Wij,exogenous_variables,params_exog,z_0):
    """Computes the opposite of the topological hessian for the k-constrained Integrated Geometric model for Undirected networks.
    Used in the solver"""
    
    n_countries = int(np.sqrt(len(Wij)))
    z_0 = z_0[0]
    Aij = binarize(Wij)
    hess = np.zeros((len(theta),len(theta)))
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params_exog = np.ascontiguousarray(params_exog)
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                xij = np.exp(-theta[i]-theta[j])
                
                aij = Aij[ij]
                wij = Wij[ij]
                zij = mu[ij]
                
                pij = xij * zij * z_0/(1+zij -z_0*zij + xij*z_0*zij)
                
                d_a_theta_i_theta_i = - pij*(1-pij)
                d_a_theta_i_theta_j = - pij*(1-pij)
                hess[i,i] += d_a_theta_i_theta_i
                hess[i,j] = d_a_theta_i_theta_j
                hess[j,i] = d_a_theta_i_theta_j
                                 
    return - hess


@jit(nopython=True,fastmath=True)
def ll_kIGeom_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the k-constrained Integrated Geometric model for Directed networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[2*n_countries:-1]
    z_0 = params[-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)

    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_out[i]*x_in[j]
                ll += ai*np.log(xij)  + yi*np.log(z_0) 
                ll += yi * np.log(mui) - yi * np.log(1+mui) + np.log(1+ mui - z_0*mui) - np.log(1 + mui - z_0*mui+ xij*z_0*mui)

    return - ll 

@jit(nopython=True,fastmath=True)
def ll_kIGeom_binary_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the k-constrained Integrated Geometric model for Directed networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[2*n_countries:-1]
    z_0 = params[-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)

    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    
    ll = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_out[i]*x_in[j]
                ll += ai*np.log(xij*z_0*mui) + (1.-ai)*np.log(1+ mui - z_0*mui) - np.log(1 + mui - z_0*mui+ xij*z_0*mui)

    return - ll 
    


@jit(nopython=True,fastmath=True)
def jac_kIGeom_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the k-constrained Integrated Geometric model for Directed networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    params_exog = params[2*n_countries:-1]
    z_0 = params[-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)

    

    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(params))
    
    mu_matrix = np.zeros(len(Aij))
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ji = j*n_countries + i
                
                yi = Wij[ij]
                aij = Aij[ij]
                aji = Aij[ji]
                muij = mu[ij]
                muji = mu[ji]
                xij = x_out[i]*x_in[j]
                pij = xij*z_0*muij/(1+muij-z_0*muij+xij*z_0*muij)
                xji = x_out[j]*x_in[i]
                pji = xji*z_0*muji/(1+muji-z_0*muji+xji*z_0*muji)
                
                jac[i] += - aij + pij
                jac[i+n_countries] += - aji + pji

                
                denominator_mu = (1+muij-z_0*muij+xij*z_0*muij) * (1+muij-z_0*muij)
                mu_matrix[ij] = yi/(1+muij) - xij*z_0*muij/denominator_mu

                adder_last += - yi  + xij*z_0*muij*(1+muij)/denominator_mu
    
    jac[2*n_countries:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder_last
    return - jac

@jit(nopython=True,fastmath=True)
def jac_kIGeom_topological_directed(theta,Wij,exogenous_variables,params_exog,z_0):
    """Computes the opposite of the topological jacobian for the k-constrained Integrated Geometric model for Directed networks"""
    
    n_countries = int(np.sqrt(len(Wij)))
    n_obs = len(Wij)
    Aij = binarize(Wij)
    
    
    z_0 = z_0[0]
    theta_out = theta[:n_countries]
    theta_in = theta[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)

    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    jac = np.zeros(len(theta))
    
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                mui = mu[ij]
                xij = x_out[i]*x_in[j]
                jac[i] +=  - ai + xij*z_0*mui/(1+mui-z_0*mui+xij*z_0*mui)
    
    return - jac






@jit(nopython=True,fastmath=True)
def ll_LIExp(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the L-constrained Integrated Exponential model"""
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[1:-1]
    
    x_0 = np.exp(-params[0])
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                bij = beta_0 + 1./zi
                ll += ai*np.log(x_0) + np.log(bij) - bij*yi - np.log(x_0+bij)
                
    return - ll 

@jit(nopython=True,fastmath=True)
def ll_LIExp_binary(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the L-constrained Integrated Exponential model"""
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[1:-1]
    
    x_0 = np.exp(-params[0])
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
    
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                pij = x_0 * mu[ij]/(1+beta_0*mu[ij]+x_0*mu[ij])

                ll += ai*np.log(pij) + (1.-ai)*np.log(1.-pij)
                
    return - ll 

@jit(nopython=True,fastmath=True)
def jac_LIExp(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the L-constrained Integrated Exponential model"""
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[1:-1]
    theta_0 = params[0]
    x_0 = np.exp(-theta_0)
    beta_0 = params[-1]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)

    n_obs = len(Aij)
    jac = np.zeros(len(params))

    mu_matrix = np.zeros(len(Aij))
    
    adder_0 = 0
    adder_last = 0
    n_countries = int(np.sqrt(len(Aij)))
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                pi = x_0 * zi/(1+ beta_0*zi + x_0*zi)
                yi_hat = pi*zi/(1+beta_0*zi)

                adder_0 += pi - ai
                adder_last += yi_hat - yi
                mu_matrix[ij] = (yi-yi_hat)/(zi)
                
    jac[0] = adder_0 
    jac[1:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder_last
    
    return - jac






@jit(nopython=True,fastmath=True)
def hess_LIExp(params,Wij,exogenous_variables):
    """Computes the opposite of the hessian for the L-constrained Integrated Exponential model"""
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[1:-1]
    theta_0 = params[0]
    x = np.exp(-theta_0)
    b0 = params[-1]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)

    n_obs = len(Aij)
    hess = np.zeros((len(params),len(params)))

    mu_matrix = np.zeros(len(Aij))
    
    adder_0 = 0
    adder_last = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    d_x_zij = np.zeros(n_obs)
    d_zij_zij = np.zeros((n_obs,n_obs))
    d_zij_b0 = np.zeros(n_obs)
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                aij = Aij[ij]
                wij = Wij[ij]
                zij = mu[ij]
                pij = x *zij/(1+zij*(b0+zij))
                den_1 = 1. + b0*zij + x*zij
                den_2 = den_1 * (1+b0*zij)
                
                hess[0,0] += - pij*(1-pij)
                hess[0,-1] += - x*zij**2/den_1**2
                hess[-1,0] += - x*zij**2/den_1**2
                d_x_zij[ij] = x*zij/den_1**2

                diffin = (b0+x)*(1.+b0*zij) + b0*(1.+b0*zij+x*zij)
                d_zij_zij[ij,ij] = -wij/zij - x*zij**2/den_2**2 + x*zij**2 * diffin/den_2**2
                d_zij_b0[ij] = x*zij**2 *(2.*zij*(1+b0*zij)+x*zij**2)/den_2**2

                hess[-1,-1] += - x*zij**2 * (2.*zij*(1.+b0*zij) + x*zij**2)/den_2**2
                
    hess[0,1:-1] = exogenous_variables.T @ d_x_zij
    hess[1:-1,0] = exogenous_variables.T @ d_x_zij
    hess[1:-1,1:-1] = exogenous_variables.T @ d_zij_zij @ exogenous_variables
    hess[1:-1,-1] = exogenous_variables.T @ d_zij_b0
    hess[-1,1:-1] = exogenous_variables.T @ d_zij_b0

    return - hess


@jit(nopython=True,fastmath=True)
def ll_kIExp_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the k-constrained Integrated Exponential model for Undirected networks"""
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[n_countries:-1]
    theta = params[:n_countries]
    x_i = np.exp(-theta)
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                xij = x_i[i]*x_i[j]
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]

                bij = beta_0 + 1./zi
                ll += ai*np.log(xij) + np.log(bij) - bij*yi - np.log(xij+bij)
                
                
    return - ll 



@jit(nopython=True,fastmath=True)
def ll_kIExp_binary_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the k-constrained Integrated Exponential model for Undirected networks"""
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[n_countries:-1]
    theta = params[:n_countries]
    x_i = np.exp(-theta)
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                xij = x_i[i]*x_i[j]
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                ll += ai*np.log(xij*zi) + (1.-ai)*np.log(1.+beta_0*zi) - np.log(1.+beta_0*zi + xij*zi) 
                

    return - ll 


@jit(nopython=True,fastmath=True)
def jac_kIExp_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the k-constrained Integrated Exponential model for Undirected networks"""
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[n_countries:-1]
    theta = params[:n_countries]
    x_i = np.exp(-theta)
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(params))

    mu_matrix = np.zeros(len(Aij))
    
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                xij = x_i[i]*x_i[j]
                pi = xij * zi/(1+ beta_0*zi + xij*zi)
                yi_hat = pi*zi/(1+beta_0*zi)

                jac[i] += - ai + pi
                adder_last += yi_hat - yi
                mu_matrix[ij] = (yi-yi_hat)/(zi)
                
    jac[n_countries:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder_last 
    
    return - jac

@jit(nopython=True,fastmath=True)
def jac_kIExp_topological_undirected(theta,Wij,exogenous_variables,params_exog,beta_0):
    """Computes the opposite of the topological jacobian for the k-constrained Integrated Exponential model for Undirected networks"""
    
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))
    
    x_i = np.exp(-theta)
    beta_0 = beta_0[0]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(theta))

    mu_matrix = np.zeros(len(Aij))
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                xij = x_i[i]*x_i[j]
                pi = xij * zi/(1+ beta_0*zi + xij*zi)
                yi_hat = pi*zi/(1+beta_0*zi)

                jac[i] += - ai + pi
                
    
    return - jac



@jit(nopython=True,fastmath=True)
def hess_kIExp_undirected(params,Wij,exogenous_variables):
    """Computes the opposite of the hessian for the k-constrained Integrated Exponential model for Undirected networks"""
    
    
    n_countries = int(np.sqrt(len(Wij)))
    params_exog = params[n_countries:-1]
    theta_0 = params[:n_countries]
    x = np.exp(-theta_0)
    b0 = params[-1]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    Aij = binarize(Wij)

    n_obs = len(Aij)
    hess = np.zeros((len(params),len(params)))

    n_obs = len(Aij)
    d_x_zij = np.zeros(n_obs)
    d_zij_zij = np.zeros((n_obs,n_obs))
    d_zij_b0 = np.zeros(n_obs)
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                aij = Aij[ij]
                wij = Wij[ij]
                zij = mu[ij]
                xij = x[i]*x[j]
                pij = xij *zij/(1+zij*(b0+zij))
                den_1 = 1. + b0*zij + xij*zij
                den_2 = den_1 * (1+b0*zij)
                
                hess[i,i] += - pij*(1-pij)
                hess[i,j] = - pij*(1-pij)
                hess[j,i] = - pij*(1-pij)
                
                hess[i,-1] += - xij*zij**2/den_1**2
                hess[-1,i] += - xij*zij**2/den_1**2
                d_x_zij[ij] = xij*zij/den_1**2

                diffin = (b0+xij)*(1.+b0*zij) + b0*(1.+b0*zij+xij*zij)
                d_zij_zij[ij,ij] = -wij/zij - xij*zij**2/den_2**2 + xij*zij**2 * diffin/den_2**2
                d_zij_b0[ij] = xij*zij**2 *(2.*zij*(1+b0*zij)+xij*zij**2)/den_2**2

                hess[-1,-1] += - xij*zij**2 * (2.*zij*(1.+b0*zij) + xij*zij**2)/den_2**2
    for i in range(n_countries):            
        hess[i,n_countries:-1] = exogenous_variables.T @ d_x_zij
        hess[n_countries:-1,i] = exogenous_variables.T @ d_x_zij
    hess[n_countries:-1,n_countries:-1] = exogenous_variables.T @ d_zij_zij @ exogenous_variables
    hess[n_countries:-1,-1] = exogenous_variables.T @ d_zij_b0
    hess[-1,n_countries:-1] = exogenous_variables.T @ d_zij_b0

    return - hess


@jit(nopython=True,fastmath=True)
def hess_kIExp_topological_undirected(theta_0,Wij,exogenous_variables,params_exog,b0):
    """Computes the opposite of the topological hessian for the k-constrained Integrated Exponential model for Undirected networks"""
    
    
    n_countries = int(np.sqrt(len(Wij)))
    x = np.exp(-theta_0)
    b0 = b0[0]
    
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    hess = np.zeros((len(theta_0),len(theta_0)))

    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                zij = mu[ij]
                xij = x[i]*x[j]
                pij = xij *zij/(1+zij*(b0+zij))
                
                hess[i,i] += - pij*(1-pij)
                hess[i,j] = - pij*(1-pij)
                hess[j,i] = - pij*(1-pij)
                
    return - hess


@jit(nopython=True,fastmath=True)
def ll_kIExp_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the log-likelihood for the k-constrained Integrated Exponential model for Directed networks"""
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[2*n_countries:-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)
    
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                xij = x_out[i]*x_in[j]
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]

                bij = beta_0 + 1./zi
                ll += ai*np.log(xij) + np.log(bij) - bij*yi - np.log(xij+bij)
                
                
    return - ll 



@jit(nopython=True,fastmath=True)
def ll_kIExp_binary_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the binary log-likelihood for the k-constrained Integrated Exponential model for Directed networks"""
    
    Aij = binarize(Wij)
    ll = 0
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[2*n_countries:-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                xij = x_out[i]*x_in[j]
                yi = Wij[ij]
                ai = Aij[ij]
                zi = mu[ij]
                ll += ai*np.log(xij*zi) + (1.-ai)*np.log(1.+beta_0*zi) - np.log(1.+beta_0*zi + xij*zi) 
                

    return - ll 


@jit(nopython=True,fastmath=True)
def jac_kIExp_directed(params,Wij,exogenous_variables):
    """Computes the opposite of the jacobian for the k-constrained Integrated Exponential model for Directed networks"""
    
    Aij = binarize(Wij)
    n_obs = len(Aij)
    n_countries = int(np.sqrt(len(Aij)))
    
    n_exog_variables = exogenous_variables.shape[1]
    params_exog = params[2*n_countries:-1]
    theta_out = params[:n_countries]
    theta_in = params[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)
    
    beta_0 = params[-1]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(params))

    mu_matrix = np.zeros(len(Aij))
    
    adder_last = 0
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ji = j*n_countries + i
                
                yi = Wij[ij]
                aij = Aij[ij]
                aji = Aij[ji]
                zij = mu[ij]
                zji = mu[ji]
                
                xij = x_out[i]*x_in[j]
                xji = x_in[i]*x_out[j]
                
                pij = xij * zij/(1.+ beta_0*zij + xij*zij)
                pji = xji * zji/(1.+ beta_0*zji + xji*zji)
                
                yi_hat = pij*zij/(1+beta_0*zij)

                jac[i] += - aij + pij
                jac[i+n_countries] += - aji + pji
                adder_last += yi_hat - yi
                mu_matrix[ij] = (yi-yi_hat)/(zij)
                
    jac[2*n_countries:-1] = exogenous_variables.T @ mu_matrix
    jac[-1] = adder_last 
    
    return - jac

@jit(nopython=True,fastmath=True)
def jac_kIExp_topological_directed(theta,Wij,exogenous_variables,params_exog,beta_0):
    """Computes the opposite of the topological jacobian for the k-constrained Integrated Exponential model for Directed networks"""
    
    Aij = binarize(Wij)
    n_countries = int(np.sqrt(len(Aij)))
    
    theta_out = theta[:n_countries]
    theta_in = theta[n_countries:2*n_countries]
    x_out = np.exp(-theta_out)
    x_in = np.exp(-theta_in)
    beta_0 = beta_0[0]
    x1_beta1 = exogenous_variables @ params_exog

    mu = np.exp(x1_beta1)
    
    jac = np.zeros(len(theta))

    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries + j
                ji = j*n_countries + i
                yi = Wij[ij]
                aij = Aij[ij]
                aji = Aij[ji]
                zij = mu[ij]
                zji = mu[ji]
                
                xij = x_out[i]*x_in[j]
                xji = x_in[i]*x_out[j]
                
                pij = xij * zij/(1+ beta_0*zij + xij*zij)
                pji = xji * zji/(1+ beta_0*zji + xji*zji)
                
                jac[i] += - aij + pij
                jac[i+n_countries] += - aji + pji
    
    return - jac


