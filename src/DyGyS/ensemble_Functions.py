import numpy as np
from numba import jit, float64,int64,typeof, prange
from scipy.special import erfinv,gammaincinv
from numpy.random import default_rng

lett = typeof('ciao')


    

@jit(nopython=True,fastmath=True,cache=False)
def IT_sampling_Exponential(random_array,zij,beta_0,n_ensemble):
    """Inverse Transform Sampling routine for the Exponential models.
    """

    x_array = -zij*np.log(1.-random_array)/(beta_0*zij+1.)
    return x_array

@jit(nopython=True,fastmath=True,cache=False)
def IT_sampling_Pareto(random_array,shape,scale,n_ensemble):
    """Inverse Transform Sampling routine for the Pareto models.
    """

    x_array = scale*(1.-random_array)**(-1./shape)
    return x_array
    

@jit(nopython=True,fastmath=True,cache=False)
def IT_sampling_Lognormal(random_array,lnzij,gamma_0,n_ensemble):
    """Inverse Transform Sampling routine for the Lognormal models.
    """

    x_array = np.empty(n_ensemble)
    sqrt_gamma_0 = np.sqrt(gamma_0)
    for i in range(n_ensemble):
        u = random_array[i]
        aux = (1.-lnzij)**2/(4.*gamma_0)
        lnx = lnzij/(2.*gamma_0) + erfinv(2.*u-1.)/sqrt_gamma_0 
        x_array[i] = np.exp(lnx)
        
    return x_array

@jit(nopython=True,fastmath=True,cache=False)
def IT_sampling_Gamma(random_array,zij,beta_0,phi,n_ensemble):
    """Inverse Transform Sampling routine for the Gamma models.
    """

    bij = beta_0 + 1./zij
    x_array = np.empty(n_ensemble)
    for i in range(n_ensemble):
        u = random_array[i]
        x_array[i] = gammaincinv(phi,u)/bij
    
    return x_array

def discrete_ensemble_matrix_undirected(params,Wij,model,exogenous_variables,
    selection_variables = np.array([]),fixed_selection_params = np.array([]),n_ensemble=1000):
    """Generate ensemble of graphs for undirected networks with discrete-valued weights


    Args:
        params (np.ndarray): params after optimization
        Wij (np.ndarray): weighted adjacency matrix
        model (string): requested model for discrete-valued weights
        exogenous_variables (np.ndarray): regressor matrix for the weighted gravity specification 
        selection_variables (np.ndarray, optional): topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
        fixed_selection_params (np.ndarray, optional): fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
        n_ensemble (int, optional): Number of graphs in the ensemble. Defaults to 1000.

    Raises:
        TypeError: If model is not a discrete count data model or is not implemented.

    Returns:
        w_mat_ensemble (np.ndarray): Weighted adjacency matrices in the ensemble. Each column refers to each graph.
    """
    
    n_obs = exogenous_variables.shape[0]
    n_countries = int(np.sqrt(n_obs))
    rng = default_rng()
    w_mat_ensemble = np.zeros((n_obs,n_ensemble))
    implemented_models = ["POIS","ZIP","NB2","ZINB",
                        "L-CGeom","k-CGeom","L-IGeom","k-IGeom"]
                        
    if len(selection_variables)!= 0:
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        
    if model not in implemented_models:
        raise TypeError("Model not yet implemented! You can see the available models compiling self.implement_models.")
    
    if model == "POIS":
        params_exog = params
        mu = np.exp(exogenous_variables@params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                w_mat_ensemble[ij,:] = rng.poisson(lam=mu[ij],size=n_ensemble)
                w_mat_ensemble[ji,:] = w_mat_ensemble[ij,:]

    if model == "ZIP":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:]
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])

                random_array = rng.random(n_ensemble)
                w_mat_array = rng.poisson(lam=mu[ij],size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

                    
    if model == "NB2":
        params_exog = params[:-1]
        alpha = params[-1]
        m = 1./alpha
        mu = np.exp(exogenous_variables@params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                prob = 1./(1.+alpha*mu[ij])
                w_mat_ensemble[ij,:] = rng.negative_binomial(n=m,p=prob,size=n_ensemble)
                w_mat_ensemble[ji,:] = w_mat_ensemble[ij,:]
            
    if model == "ZINB":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        alpha = params[-1]
        m = 1./alpha
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                prob = 1./(1.+alpha*mu[ij])
                random_array = rng.random(n_ensemble)
                w_mat_array = rng.negative_binomial(n=m,p=prob,size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    if model == "L-CGeom":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        z_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                random_array = rng.random(n_ensemble)
                w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

                
    if model == "k-CGeom":
        theta_i = params[:n_countries]
        params_exog = params[n_countries:-1]
        x_i = np.exp(-theta_i)
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x_i[i]*x_i[j]
                pij = xij/(1.+xij)
                prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                random_array = rng.random(n_ensemble)
                w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    
    if model == "L-IGeom":
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        params_exog = params[1:-1]
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = x_0 * z_0 * mu[ij]/(1+mu[ij]-z_0*mu[ij]+x_0*z_0*mu[ij])
                prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                random_array = rng.random(n_ensemble)
                w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    
    if model == "k-IGeom":
        theta = params[:n_countries]
        x = np.exp(-theta)
        params_exog = params[n_countries:-1]
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x[i]*x[j]
                pij = xij * z_0 * mu[ij]/(1+mu[ij]-z_0*mu[ij]+xij*z_0*mu[ij])
                prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                random_array = rng.random(n_ensemble)
                w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                indices = np.where(random_array < pij)
                
                for idx in indices:
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    return w_mat_ensemble

@jit(nopython=True,fastmath=True,parallel=False,nogil=False)
def faster_ensemble_matrix_undirected(params,Wij,model,exogenous_variables,
    selection_variables = np.array([]),fixed_selection_params = np.array([]),n_ensemble=1000):
    """Generate ensemble of graphs for undirected networks with continuous-valued weights

    Args:
        params (np.ndarray): params after optimization
        Wij (np.ndarray): weighted adjacency matrix
        model (string): requested model for discrete-valued weights
        exogenous_variables (np.ndarray): regressor matrix for the weighted gravity specification 
        selection_variables (np.ndarray, optional): topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
        fixed_selection_params (np.ndarray, optional): fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
        n_ensemble (int, optional): Number of graphs in the ensemble. Defaults to 1000.

    Returns:
        w_mat_ensemble (np.ndarray): Weighted adjacency matrices in the ensemble. Each column refers to each graph.
    """
    
    implemented_models = ["L-IExp","k-IExp","k-CExp","L-CExp","L-CPareto","k-CPareto","L-CGamma","k-CGamma","L-CLognormal","k-CLognormal"]
    if model not in implemented_models:
        raise TypeError("Model not yet implemented! You can see the available models compiling self.implement_models.")
    
    
    n_obs = exogenous_variables.shape[0]
    n_countries = int(np.sqrt(n_obs))
    w_mat_ensemble = np.zeros((n_obs,n_ensemble))
    # rng = default_rng()
    
                        
    if len(selection_variables)!= 0:
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
    selection_variables = np.ascontiguousarray(selection_variables)
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params = np.ascontiguousarray(params)    
    
    if model == "L-CExp":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        beta_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                random_array = np.random.rand(n_ensemble)
                w_hat = mu[ij]/(1.+beta_0*mu[ij])
                # w_mat_array = rng.exponential(scale=w_hat,size=n_ensemble)
                w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)              
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_array[idx]
                    

                
    if model == "k-CExp":
        theta_i = params[:n_countries]
        params_exog = params[n_countries:-1]
        x_i = np.exp(-theta_i)
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x_i[i]*x_i[j]
                pij = xij/(1.+xij)
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    
    if model == "L-IExp":
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        params_exog = params[1:-1]
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = x_0 * mu[ij]/(1+beta_0*mu[ij]+x_0*mu[ij])
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    if model == "k-IExp":
        theta = params[:n_countries]
        x = np.exp(-theta)
        params_exog = params[n_countries:-1]
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x[i]*x[j]
                pij = xij * mu[ij]/(1+beta_0*mu[ij]+xij*mu[ij])
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

                    
    
    
    if model == "L-CPareto":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:]
        
        
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        positive_weights = []
        for ij in range(len(Wij)):
            weight = Wij[ij]
            if weight>0:
                positive_weights.append(weight)
        
        w_min = min(positive_weights)
        
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                random_array = np.random.rand(n_ensemble)
                shape = 1.+1./mu[ij]
                scale = w_min
                w_mat_array = IT_sampling_Pareto(random_array,shape,scale,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    
    

    if model == "k-CPareto":
        theta = params[:n_countries]
        x = np.exp(-theta)
        params_exog = params[n_countries:]
        
        
        mu = np.exp(exogenous_variables @ params_exog)

        positive_weights = []
        for weight in Wij:
            if weight>0:
                positive_weights.append(weight)
        w_min = min(positive_weights)

        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x[i]*x[j]
                pij = xij/(1.+xij)
                random_array = np.random.rand(n_ensemble)
                shape = 1.+1./mu[ij]
                scale = w_min
                w_mat_array = IT_sampling_Pareto(random_array,shape,scale,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    

    if model == "L-CGamma":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-2]
        beta_0 = params[-2]
        phi = params[-1]
        
        
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                scale = mu[ij]/(1.+beta_0*mu[ij])
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Gamma(random_array,mu[ij],beta_0,phi,n_ensemble)                
                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    
    if model == "k-CGamma":
        theta = params[:n_countries]
        x = np.exp(-theta)
        params_exog = params[n_countries:-2]
        beta_0 = params[-2]
        phi = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x[i]*x[j]
                pij = xij/(1.+xij)
                random_array = np.random.rand(n_ensemble)
                scale = mu[ij]/(1.+beta_0*mu[ij])
                w_mat_array = IT_sampling_Gamma(random_array,mu[ij],beta_0,phi,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]
    
    if model == "L-CLognormal":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        
        
        
        gamma_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = exogenous_variables @ params_exog
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                pij = Gij[ij]/(1.+Gij[ij])
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Lognormal(random_array,mu[ij],gamma_0,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    if model == "k-CLognormal":
        theta = params[:n_countries]
        x = np.exp(-theta)
        params_exog = params[n_countries:-1]
        gamma_0 = params[-1]
        
        
        mu = exogenous_variables @ params_exog
        for i in range(n_countries):
            for j in range(i+1,n_countries):
                ij = i*n_countries+j
                ji = j*n_countries+i
                xij = x[i]*x[j]
                pij = xij/(1.+xij)
                random_array = np.random.rand(n_ensemble)
                w_mat_array = IT_sampling_Lognormal(random_array,mu[ij],gamma_0,n_ensemble)                
                indices = np.where(random_array < pij)[0]
                
                for k in range(len(indices)):
                    idx = indices[k]
                    w_mat_ensemble[ij,idx] = w_mat_array[idx]
                    w_mat_ensemble[ji,idx] = w_mat_ensemble[ij,idx]

    
    return w_mat_ensemble



#####directed
def discrete_ensemble_matrix_directed(params,Wij,model,exogenous_variables,
    selection_variables = np.array([]),fixed_selection_params = np.array([]),n_ensemble=1000):
    """Generate ensemble of graphs for directed networks with discrete-valued weights

    Args:
        params (np.ndarray): params after optimization
        Wij (np.ndarray): weighted adjacency matrix
        model (string): requested model for discrete-valued weights
        exogenous_variables (np.ndarray): regressor matrix for the weighted gravity specification 
        selection_variables (np.ndarray, optional): topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
        fixed_selection_params (np.ndarray, optional): fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
        n_ensemble (int, optional): Number of graphs in the ensemble. Defaults to 1000.

    Raises:
        TypeError: If model is not a discrete count data model or is not implemented.

    Returns:
        w_mat_ensemble (np.ndarray): Weighted adjacency matrices in the ensemble. Each column refers to each graph.
    """
    n_obs = exogenous_variables.shape[0]
    n_countries = int(np.sqrt(n_obs))
    rng = default_rng()
    w_mat_ensemble = np.zeros((n_obs,n_ensemble))
    implemented_models = ["POIS","ZIP","NB2","ZINB",
                        "L-CGeom","k-CGeom","L-IGeom","k-IGeom"]
                        
    if len(selection_variables)!= 0:
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        
    if model not in implemented_models:
        raise TypeError("Model not yet implemented! You can see the available models compiling self.implement_models.")
    
    if model == "POIS":
        params_exog = params
        mu = np.exp(exogenous_variables@params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    w_mat_ensemble[ij,:] = rng.poisson(lam=mu[ij],size=n_ensemble)
                    
    if model == "ZIP":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:]
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])

                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.poisson(lam=mu[ij],size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
                    
    if model == "NB2":
        params_exog = params[:-1]
        alpha = params[-1]
        m = 1./alpha
        mu = np.exp(exogenous_variables@params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    prob = 1./(1.+alpha*mu[ij])
                    w_mat_ensemble[ij,:] = rng.negative_binomial(n=m,p=prob,size=n_ensemble)
                    
    if model == "ZINB":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        alpha = params[-1]
        m = 1./alpha
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    prob = 1./(1.+alpha*mu[ij])
                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.negative_binomial(n=m,p=prob,size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "L-CGeom":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        z_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
                
    if model == "k-CGeom":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        
        params_exog = params[2*n_countries:-1]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij/(1.+xij)
                    prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "L-IGeom":
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        params_exog = params[1:-1]
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = x_0 * z_0 * mu[ij]/(1+mu[ij]-z_0*mu[ij]+x_0*z_0*mu[ij])
                    prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "k-IGeom":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        params_exog = params[2*n_countries:-1]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        z_0 = params[-1]
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij * z_0 * mu[ij]/(1+mu[ij]-z_0*mu[ij]+xij*z_0*mu[ij])
                    prob = 1. - z_0*mu[ij]/(1.+mu[ij])
                    random_array = rng.random(n_ensemble)
                    w_mat_array = rng.geometric(p=prob,size=n_ensemble)
                    indices = np.where(random_array < pij)
                    
                    for idx in indices:
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    return w_mat_ensemble


@jit(nopython=True,fastmath=True,parallel=False,nogil=False)
def faster_ensemble_matrix_directed(params,Wij,model,exogenous_variables,
    selection_variables = np.array([]),fixed_selection_params = np.array([]),n_ensemble=1000):
    """Generate ensemble of graphs for directed networks with continuous-valued weights

    Args:
        params (np.ndarray): params after optimization
        Wij (np.ndarray): weighted adjacency matrix
        model (string): requested model for discrete-valued weights
        exogenous_variables (np.ndarray): regressor matrix for the weighted gravity specification 
        selection_variables (np.ndarray, optional): topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
        fixed_selection_params (np.ndarray, optional): fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
        n_ensemble (int, optional): Number of graphs in the ensemble. Defaults to 1000.

    Raises:
        TypeError: If model is not a continuous model or is not implemented.

    Returns:
        w_mat_ensemble (np.ndarray): Weighted adjacency matrices in the ensemble. Each column refers to each graph.
    """
    n_obs = exogenous_variables.shape[0]
    n_countries = int(np.sqrt(n_obs))
    w_mat_ensemble = np.zeros((n_obs,n_ensemble))
                   
    implemented_models = ["L-IExp","k-IExp","k-CExp","L-CExp","L-CPareto","k-CPareto","L-CGamma","k-CGamma","L-CLognormal","k-CLognormal"]
    if model not in implemented_models:
        raise TypeError("Model not yet implemented! You can see the available models compiling self.implement_models.")
         
    if len(selection_variables)!= 0:
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
    selection_variables = np.ascontiguousarray(selection_variables)
    exogenous_variables = np.ascontiguousarray(exogenous_variables)
    params = np.ascontiguousarray(params)    
    # rng = default_rng()
    
    
    if model == "L-CExp":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        beta_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    random_array = np.random.rand(n_ensemble)
                    w_hat = mu[ij]/(1.+beta_0*mu[ij])
                    # w_mat_array = rng.exponential(scale=w_hat,size=n_ensemble)
                
                    w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)              
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        

                
    if model == "k-CExp":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        params_exog = params[2*n_countries:-1]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij/(1.+xij)
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "L-IExp":
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        params_exog = params[1:-1]
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = x_0 * mu[ij]/(1+beta_0*mu[ij]+x_0*mu[ij])
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "k-IExp":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        params_exog = params[2*n_countries:-1]
        beta_0 = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij * mu[ij]/(1+beta_0*mu[ij]+xij*mu[ij])
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Exponential(random_array,mu[ij],beta_0,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
                    
    
    
    if model == "L-CPareto":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:]
        
        
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        positive_weights = []
        for ij in range(len(Wij)):
            weight = Wij[ij]
            if weight>0:
                positive_weights.append(weight)
        
        w_min = min(positive_weights)
        
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    random_array = np.random.rand(n_ensemble)
                    shape = 1.+1./mu[ij]
                    scale = w_min
                    w_mat_array = IT_sampling_Pareto(random_array,shape,scale,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    
    

    if model == "k-CPareto":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        params_exog = params[2*n_countries:]
        
        
        mu = np.exp(exogenous_variables @ params_exog)

        positive_weights = []
        for weight in Wij:
            if weight>0:
                positive_weights.append(weight)
        w_min = min(positive_weights)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij/(1.+xij)
                    random_array = np.random.rand(n_ensemble)
                    shape = 1.+1./mu[ij]
                    scale = w_min
                    w_mat_array = IT_sampling_Pareto(random_array,shape,scale,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        

    if model == "L-CGamma":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-2]
        beta_0 = params[-2]
        phi = params[-1]
        
        
        
        Gij = np.exp(selection_variables @ params_select)
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    scale = mu[ij]/(1.+beta_0*mu[ij])
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Gamma(random_array,mu[ij],beta_0,phi,n_ensemble)                
                    
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "k-CGamma":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        params_exog = params[2*n_countries:-2]
        beta_0 = params[-2]
        phi = params[-1]
        
        
        mu = np.exp(exogenous_variables @ params_exog)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij/(1.+xij)
                    random_array = np.random.rand(n_ensemble)
                    scale = mu[ij]/(1.+beta_0*mu[ij])
                    w_mat_array = IT_sampling_Gamma(random_array,mu[ij],beta_0,phi,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "L-CLognormal":
        params_select = params[:n_selection_variables]
        params_select = np.concatenate((params_select,fixed_selection_params))
        params_exog = params[n_selection_variables:-1]
        
        
        
        gamma_0 = params[-1]
        Gij = np.exp(selection_variables @ params_select)
        mu = exogenous_variables @ params_exog
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    pij = Gij[ij]/(1.+Gij[ij])
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Lognormal(random_array,mu[ij],gamma_0,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    if model == "k-CLognormal":
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        x_out = np.exp(-theta_out)
        x_in = np.exp(-theta_in)
        params_exog = params[2*n_countries:-1]
        gamma_0 = params[-1]
        
        
        mu = exogenous_variables @ params_exog
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    
                    ij = i*n_countries+j
                    ji = j*n_countries+i
                    xij = x_out[i]*x_in[j]
                    pij = xij/(1.+xij)
                    random_array = np.random.rand(n_ensemble)
                    w_mat_array = IT_sampling_Lognormal(random_array,mu[ij],gamma_0,n_ensemble)                
                    indices = np.where(random_array < pij)[0]
                    
                    for k in range(len(indices)):
                        idx = indices[k]
                        w_mat_ensemble[ij,idx] = w_mat_array[idx]
                        
    
    return w_mat_ensemble





