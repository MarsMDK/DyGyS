import numpy as np
from numba import jit, float64, typeof, int64, prange

lett = typeof('ciao')

@jit(nopython=True)
def symmetrize(input):
    """Symmetrize weighted adjacency matrix using the average between $$w_{ij}$$ and $$w_{ji}$$. """
    n_countries = int(np.sqrt(len(input)))
    dim = len(input)
     
    output_w = np.zeros(dim)
    for i in range(n_countries):
        for j in range(n_countries):
            if j!=i:
                ij = i*n_countries+j
                ji = j*n_countries + i

                output_w[ij] = input[ij] + input[ji]
                output_w[ij]/=2
                
    
    return output_w

@jit(nopython=True,fastmath=True)
def is_symmetric(input):
    """Check if weighted adjacency matrix is symmetric."""
    
    n_countries = int(np.sqrt(len(input)))
    flag = True
    for i in range(n_countries):
        for j in range(n_countries):
            if j>i:
                ij = i*n_countries+j
                ji = j*n_countries + i
                if input[ij] != input[ji]:
                    flag = False
    return flag

@jit(nopython=True,fastmath=True,cache=False)
def flatten(input):
    """Flatten numpy 2-D matrix to numpy 1-D array."""
    
    n = len(input)
    dim = n*n
    array = np.zeros(dim)
    
    for i in range(n):
        for j in range(n):
            ij = i*n+j
            array[ij] = input[i,j]
        
    return array


@jit(nopython=True,fastmath=True,cache=False)
def matrixate(input):
    """Turns 1-D numpy array into a numpy 2-D matrix."""
    
    n = int(np.sqrt(len(input)))
    mat = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            ij = i*n+j
            mat[i,j] = input[ij]
        
    return mat 

@jit(nopython=True,fastmath=True,cache=False)
def binarize(input):
    """Return binary adjacency matrix from weighted adjacency matrix."""
    dim = len(input)
    byn = np.zeros(dim)
    for i in range(dim):
        if input[i]> 0:
            byn[i] = 1.
    return byn 
        

@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def deg(adj):
    """Compute (out-)degree centrality for (Directed)Undirected Networks."""
    
    n_countries = int(np.sqrt(len(adj)))
    
    deg = np.zeros(n_countries)
    for i in range(n_countries):
        for j in range(n_countries):
            if j != i:
                ij = i*n_countries + j
                deg[i] += adj[ij]
    return deg 

@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def deg_in(adj):
    """Compute in-degree centrality for Directed Networks."""
    n_countries = int(np.sqrt(len(adj)))
    
    deg = np.zeros(n_countries)
    for i in range(n_countries):
        for j in range(n_countries):
            if j != i:
                ij = i*n_countries + j
                ji = j*n_countries + i
                deg[i] += adj[ji]
    return deg 




@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def knn_single(adj):
    """Compute (out/out)average neighbor degree."""
    n_countries = int(np.sqrt(len(adj)))
    
    knn_out_out = np.zeros(n_countries)
    k_i= deg(adj)
    
    for i in range(n_countries):
        numerator = 0
        if k_i[i] > 0:
            for j in range(n_countries):
                ij = i*n_countries+j
                if j != i and adj[ij] ==1:
                    numerator += adj[ij]*k_i[j]
                    
            knn_out_out[i] = numerator/k_i[i]
            
    return knn_out_out    


    
@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def clust_single_fast(adj):
    """Computes binary clustering coefficient(cycle-type)."""
    n_countries = int(np.sqrt(len(adj)))
    adj_mat = np.ascontiguousarray(matrixate(adj))
    adj3_mat = adj_mat @ adj_mat @ adj_mat
    
    clust_cyc = np.zeros(n_countries)
    k_i = deg(adj)
    for i in range(n_countries):
        deg_i = k_i[i]
        if deg_i > 1:
            numerator = adj3_mat[i,i]
            for j in range(n_countries):
                ij = i*n_countries+j
                aij = adj_mat[i,j]
                aii = adj_mat[i,i]
                
                aux1 = aij*aij*aii
                aux2 = aij*aij*aii
                aux3 = aii*aii*aii
                aux4 = aii*aij*aij
                tot_aux = aux1+aux2+aux3+aux4
                
                numerator -= tot_aux
                
            clust_cyc[i] = numerator/(deg_i*(deg_i-1))
    
    return clust_cyc





@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def strength(w_adj):
    """Computes (out-)strength sequence for (Directed)Undirected networks"""
    n_countries = int(np.sqrt(len(w_adj)))
    
    strg = np.zeros(n_countries)
    for i in range(n_countries):
        for j in range(n_countries):
            if j != i:
                ij = i*n_countries+j
                strg[i] += w_adj[ij]
    return strg

@jit(float64[:](float64[:]),nopython=True,fastmath=True,cache=False)
def strength_in(w_adj):
    """Computes in-strength sequence"""
    n_countries = int(np.sqrt(len(w_adj)))
    
    strg = np.zeros(n_countries)
    for i in range(n_countries):
        for j in range(n_countries):
            if j != i:
                ij = i*n_countries+j
                ji = j*n_countries+i
                strg[i] += w_adj[ji]
    return strg


@jit(float64[:](float64[:],float64[:]),nopython=True,fastmath=True,cache=False)
def stnn_single(w_adj,adj):
    """Computes (out/out) average neighbor strength."""
    n_countries = int(np.sqrt(len(w_adj)))
    
    s_nn = np.zeros(n_countries)
    k_i = deg(adj)
    s_i = strength(w_adj)
    
    for i in range(n_countries):
        deg_i = k_i[i]
        if deg_i > 0:
            numerator = 0
            for j in range(n_countries):
                ij = i*n_countries+j
                if j!=i and adj[ij] == 1:
                    numerator += adj[ij]*s_i[j]
                    
            s_nn[i] = numerator/deg_i

    return s_nn





@jit(float64[:](float64[:],float64[:]),nopython=True,fastmath=True,cache=False)
def clust_w_single_fast(w_adj,adj):
    """Computes linear weighted clustering coefficient (cycle type)."""
    
    n_countries = int(np.sqrt(len(adj)))
    w_adj_mat = np.ascontiguousarray(matrixate(w_adj))
    
    w_adj_mat3 = w_adj_mat @ w_adj_mat @ w_adj_mat 
    
    cw = np.zeros(n_countries)
    k_i = deg(adj)
    for i in range(n_countries):
        deg_i = k_i[i]
        if deg_i > 1:
            numerator = w_adj_mat3[i,i]
            for j in range(n_countries):
                ij = i*n_countries + j
                
                wij = w_adj_mat[i,j]
                wii = w_adj_mat[i,i]
                ####use symmetric assumption
                aux1 = wij*wij*wii
                aux2 = wij*wij*wii
                aux3 = wii*wii*wii
                aux4 = wii*wij*wij
                tot_aux = aux1+aux2+aux3+aux4
                
                numerator -= tot_aux
                
            cw[i] = numerator/(deg_i*(deg_i-1))
    return cw
    
#@jit(nopython=True,fastmath=True)
def pij_matrix_undirected(params,method,Wij,selection_variables = np.array([]),exogenous_variables = np.array([]),
                            fixed_selection_params = np.array([])):
    """Computes expected binary adjacency matrix according to model of choice for Undirected networks.
    
    :param params: params after optimization
    :type params: np.ndarray
    :param Wij: weighted adjacency matrix
    :type Wij: np.ndarray
    :param model: requested model for discrete-valued weights
    :type model: str
    :param exogenous_variables: regressor matrix for the weighted gravity specification 
    :type exogenous_variables: np.ndarray
    :param selection_variables: topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
    :type selection_variables: np.ndarray, optional
    :param fixed_selection_params: fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
    :type fixed_selection_params: np.ndarray, optional
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    
    
    :return top_mat: topological expected matrix in 1-D form
    :rtype top_mat: np.ndarray
    """
    
    
    top_mat = np.zeros(len(Wij))
    n_countries = int(np.sqrt(len(Wij)))
    L_conditionals = ["Logit","L-CGeom","L-CExp","L-CPareto","L-CGamma","L-CLognormal"]
    k_conditionals = ["UBCM","k-CGeom","k-CExp","k-CPareto","k-CGamma","k-CLognormal"]
            
    if method == 'POIS':
        
        x1_beta1 = exogenous_variables @ params
        zij = np.exp(x1_beta1)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    top_mat[ij] = 1. - np.exp(-zij[ij])
                    
    if method == 'NB2':
        
        params_exog = params[:-1]
        alpha = params[-1]
        m = 1./alpha

        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    chi = 1. + alpha*zij[ij]
                    tij = np.power(1/chi,m)

                    top_mat[ij] = 1. - tij
    
    if method == 'ZIP':
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_exog = params[n_selection_variables:]
        params_top = np.concatenate((params_top,fixed_selection_params))

        x0_beta0 = selection_variables @ params_top
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    tij = np.exp(-zij[ij])
                    G_fun = Gij[ij]/(1+Gij[ij])


                    top_mat[ij] = G_fun *(1. - tij)

    
    if method == "ZINB":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_exog = params[n_selection_variables:-1]
        params_top = np.concatenate((params_top,fixed_selection_params))

        alpha = params[-1]
        m = 1./alpha

        x0_beta0 = selection_variables @ params_top
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    G_fun = Gij[ij]/(1+Gij[ij])
                    chi = 1. + alpha*zij[ij]
                    tij = np.power(1/chi,m)

                    top_mat[ij] = G_fun *(1. - tij)



    if method in L_conditionals:    
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_top = np.concatenate((params_top,fixed_selection_params))
        
        x0_beta0 = selection_variables @ params_top
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    G_fun = Gij[ij]/(1+Gij[ij])
                    top_mat[ij] = G_fun 

                
                    
    if method in k_conditionals:
        theta = params[:n_countries]
        
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta[i]-theta[j])

                    G_fun = xij/(1+xij)

                    top_mat[ij] = G_fun 

                
                    
    if method == 'L-IGeom':    
        params_exog = params[1:-1]
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        # x_0 = params[0]
        z_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j

                    pij = x_0 * z_0 * zij[ij]/(1+zij[ij]-z_0*zij[ij]+x_0*z_0*zij[ij])
                    
                    top_mat[ij] = pij
        
        
    if method == 'k-IGeom':    
        theta = params[:n_countries]
        params_exog = params[n_countries:-1]
        
        z_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta[i]-theta[j])
                    pij = xij * z_0 * zij[ij]/(1+zij[ij]-z_0*zij[ij]+xij*z_0*zij[ij])
                    
                    top_mat[ij] = pij
        
    
            
                    
    if method == 'L-IExp':    
        params_exog = params[1:-1]
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        beta_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j

                    pij = x_0 * zij[ij]/(1+beta_0*zij[ij]+x_0*zij[ij])
                    wij_hat = pij*(zij[ij])/(1+beta_0*zij[ij])

                    top_mat[ij] = pij
        
        
    if method == 'k-IExp':    
        theta = params[:n_countries]
        params_exog = params[n_countries:-1]
        
        beta_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta[i]-theta[j])
                    pij = xij * zij[ij]/(1+beta_0*zij[ij]+xij*zij[ij])
                    wij_hat = pij*(zij[ij])/(1+beta_0*zij[ij])

                    top_mat[ij] = pij

    #change from here
                    
                    
                    
                
                    
                                    
        
    return top_mat


def pij_matrix_directed(params,method,Wij,selection_variables = np.array([]),exogenous_variables = np.array([]),
                            fixed_selection_params = np.array([])):
    """Computes expected binary adjacency matrix according to model of choice for Directed networks.
    
    
    :param params: params after optimization
    :type params: np.ndarray
    :param Wij: weighted adjacency matrix
    :type Wij: np.ndarray
    :param model: requested model for discrete-valued weights
    :type model: str
    :param exogenous_variables: regressor matrix for the weighted gravity specification 
    :type exogenous_variables: np.ndarray
    :param selection_variables: topological regressor matrix for zero-inflated and L-C models. Defaults to np.array([]).
    :type selection_variables: np.ndarray, optional
    :param fixed_selection_params: fixed parameters for the topological stage for zero-inflated and L-C models. Defaults to np.array([]).
    :type fixed_selection_params: np.ndarray, optional
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    
    
    :return top_mat: topological expected matrix in 1-D form
    :rtype top_mat: np.ndarray
    """
    
    top_mat = np.zeros(len(Wij))
    n_countries = int(np.sqrt(len(Wij)))
    L_conditionals = ["Logit","L-CGeom","L-CExp","L-CPareto","L-CGamma","L-CLognormal"]
    k_conditionals = ["DBCM","k-CGeom","k-CExp","k-CPareto","k-CGamma","k-CLognormal"]
    
        
    if method == 'POIS':
        
        x1_beta1 = exogenous_variables @ params
        zij = np.exp(x1_beta1)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    top_mat[ij] = 1. - np.exp(-zij[ij])
                    
    if method == 'NB2':
        
        params_exog = params[:-1]
        alpha = params[-1]
        m = 1./alpha

        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    chi = 1. + alpha*zij[ij]
                    tij = np.power(1/chi,m)

                    top_mat[ij] = 1. - tij
    
    if method == 'ZIP':
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_exog = params[n_selection_variables:]
        params_top = np.concatenate((params_top,fixed_selection_params))

        x0_beta0 = selection_variables @ params_top
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    tij = np.exp(-zij[ij])
                    G_fun = Gij[ij]/(1+Gij[ij])


                    top_mat[ij] = G_fun *(1. - tij)

    
    if method == "ZINB":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_exog = params[n_selection_variables:-1]
        params_top = np.concatenate((params_top,fixed_selection_params))

        alpha = params[-1]
        m = 1./alpha

        x0_beta0 = selection_variables @ params_top
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    G_fun = Gij[ij]/(1+Gij[ij])
                    chi = 1. + alpha*zij[ij]
                    tij = np.power(1/chi,m)

                    top_mat[ij] = G_fun *(1. - tij)

    if method in L_conditionals:
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
    
        params_top = params[:n_selection_variables]
        params_top = np.concatenate((params_top,fixed_selection_params))
        
        x0_beta0 = selection_variables @ params_top
        Gij = np.exp(x0_beta0)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    G_fun = Gij[ij]/(1+Gij[ij])
                    top_mat[ij] = G_fun 

    if method in k_conditionals:
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        
        
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta_out[i]-theta_in[j])
                    G_fun = xij/(1+xij)
                    top_mat[ij] = G_fun 
                    
    if method == 'L-IGeom':    
        params_exog = params[1:-1]
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        # x_0 = params[0]
        z_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j

                    pij = x_0 * z_0 * zij[ij]/(1+zij[ij]-z_0*zij[ij]+x_0*z_0*zij[ij])
                    
                    top_mat[ij] = pij
        
        
    if method == 'k-IGeom':    
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        params_exog = params[2*n_countries:-1]
        
        z_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta_out[i]-theta_in[j])
                    pij = xij * z_0 * zij[ij]/(1+zij[ij]-z_0*zij[ij]+xij*z_0*zij[ij])
                    
                    top_mat[ij] = pij
        
                    
    if method == 'L-IExp':    
        params_exog = params[1:-1]
        theta_0 = params[0]
        x_0 = np.exp(-theta_0)
        beta_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j

                    pij = x_0 * zij[ij]/(1+beta_0*zij[ij]+x_0*zij[ij])
                    wij_hat = pij*(zij[ij])/(1+beta_0*zij[ij])

                    top_mat[ij] = pij
        
        
    if method == 'k-IExp':    
        theta_out = params[:n_countries]
        theta_in = params[n_countries:2*n_countries]
        
        params_exog = params[2*n_countries:-1]
        
        beta_0 = params[-1]
        
        x1_beta1 = exogenous_variables @ params_exog
        zij = np.exp(x1_beta1)

        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    xij = np.exp(-theta_out[i]-theta_in[j])
                    pij = xij * zij[ij]/(1+beta_0*zij[ij]+xij*zij[ij])
                    wij_hat = pij*(zij[ij])/(1+beta_0*zij[ij])

                    top_mat[ij] = pij
                
        
    return top_mat


@jit(nopython=True,fastmath=True)
def AIC(ll,nparams):
    """Computes Akaike Measure given log-likelihood value and number of free parameters"""
    logl = ll
    ak = (2*nparams - 2*logl)

    return  ak

@jit(nopython=True,fastmath=True)
def BIC(ll,nparams,nobs):
    """Computes BIC Measure given log-likelihood value, number of free parameters and number of observations."""
    
    logl = ll
    BIC = (nparams*np.log(nobs) - 2*logl)

    return BIC


@jit(nopython=True,fastmath=True) 
def TPR(adj_emp,adj_ens):
    """Computes True Positive Rate given empirical and expected binary adjacency matrices."""
    n_countries = int(np.sqrt(len(adj_emp)))
    
    count_tp = 0
    count_fn = 0

    for i in range(n_countries):
        for j in range(n_countries):
            ij = i*n_countries + j
            count_tp += adj_emp[ij]*adj_ens[ij]
            count_fn += (1-adj_emp[ij])*adj_ens[ij]
    soi = count_fn + count_tp
    tpr = count_tp / soi
    return tpr


@jit(nopython=True,fastmath=True) 
def SPC(adj_emp,adj_ens):
    """Computes Specificity given empirical and expected binary adjacency matrices."""
    n_countries = int(np.sqrt(len(adj_emp)))
    count_tn = 0
    count_fp = 0

    for i in range(n_countries):
        for j in range(n_countries):
            ij = i*n_countries + j
            count_tn += (1-adj_emp[ij])*(1-adj_ens[ij])
            count_fp += adj_emp[ij]*(1-adj_ens[ij])
    soi = count_fp + count_tn
    spc = count_tn / soi
    return spc



@jit(nopython=True,fastmath=True) 
def PPV(adj_emp,adj_ens):
    """Computes Precision given empirical and expected binary adjacency matrices."""
    n_countries = int(np.sqrt(len(adj_emp)))
    count_tp = 0
    count_fp = 0

    for i in range(n_countries):
        for j in range(n_countries):
            ij = i*n_countries + j
            count_tp += adj_emp[ij]*adj_ens[ij]
            count_fp += adj_emp[ij]*(1-adj_ens[ij])
    soi = count_fp + count_tp
    ppv = count_tp / soi
    return ppv

@jit(nopython=True,fastmath=True) 
def ACC(adj_emp,adj_ens):
    """Computes Accuracy given empirical and expected binary adjacency matrices."""
    n_countries = int(np.sqrt(len(adj_emp)))
    
    count_tp = 0
    count_tn = 0
    count_fp = 0
    count_fn = 0

    for i in range(n_countries):
        for j in range(n_countries):
            ij = i*n_countries + j
            count_tp += adj_emp[ij]*adj_ens[ij]
            count_fn += (1-adj_emp[ij])*adj_ens[ij]
            count_fp += adj_emp[ij]*(1-adj_ens[ij])
            count_tn += (1-adj_emp[ij])*(1-adj_ens[ij])

    soi = count_fn + count_tp + count_tn + count_fp
    son = count_tp + count_tn
    acc = son / soi
    return acc

@jit(nopython=True,fastmath=True) 
def BACC(adj_emp,adj_ens):
    """Computes Balanced Accuracy given empirical and expected binary adjacency matrices."""
    tpr_model = TPR(adj_emp,adj_ens)
    spc_model = SPC(adj_emp,adj_ens)


    bacc = (tpr_model + spc_model)/2.
    return bacc

@jit(nopython=True,fastmath=True) 
def F1_score(adj_emp,adj_ens):
    """Computes F1 Score given empirical and expected binary adjacency matrices."""
    tpr_model = TPR(adj_emp,adj_ens)
    ppv_model = PPV(adj_emp,adj_ens)


    f1 = 2*tpr_model*ppv_model/(tpr_model+ppv_model)
    return f1



@jit(nopython=True,fastmath=True,parallel=True)
def TPR_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for True Positive Rate in the Graph Ensemble.
    It returns the ensemble average, standard deviation, percentiles and array for TPR.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    
   """
    n_countries = int(np.sqrt(len(aij_emp)))
    array_TPR = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_TPR[step] = TPR(aij_emp,aij)
        
    avg_TPR = array_TPR.mean()
    std_TPR = array_TPR.std()
    percentiles_TPR = (np.percentile(array_TPR,percentiles[0]),np.percentile(array_TPR,percentiles[1]))

    return avg_TPR, std_TPR, percentiles_TPR, array_TPR

@jit(nopython=True,fastmath=True,parallel=True)
def SPC_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for Specificity in the Graph Ensemble.

    It returns the ensemble average, standard deviation, percentiles and array for SPC.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    """
    
    n_countries = int(np.sqrt(len(aij_emp)))
    array_SPC = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_SPC[step] = SPC(aij_emp,aij)
        
    avg_SPC = array_SPC.mean()
    std_SPC = array_SPC.std()
    percentiles_SPC = (np.percentile(array_SPC,percentiles[0]),np.percentile(array_SPC,percentiles[1]))

    return avg_SPC, std_SPC, percentiles_SPC, array_SPC

@jit(nopython=True,fastmath=True,parallel=True)
def PPV_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for Precision in the Graph Ensemble.

   It returns the ensemble average, standard deviation, percentiles and array for PPV.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    """
    
    n_countries = int(np.sqrt(len(aij_emp)))
    array_PPV = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_PPV[step] = PPV(aij_emp,aij)
        
    avg_PPV = array_PPV.mean()
    std_PPV = array_PPV.std()
    percentiles_PPV = (np.percentile(array_PPV,percentiles[0]),np.percentile(array_PPV,percentiles[1]))

    return avg_PPV, std_PPV, percentiles_PPV, array_PPV

@jit(nopython=True,fastmath=True,parallel=True)
def ACC_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for Accuracy in the Graph Ensemble.

    It returns the ensemble average, standard deviation, percentiles and array for ACC.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    """
    
    n_countries = int(np.sqrt(len(aij_emp)))
    array_ACC = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_ACC[step] = ACC(aij_emp,aij)
        
    avg_ACC = array_ACC.mean()
    std_ACC = array_ACC.std()
    percentiles_ACC = (np.percentile(array_ACC,percentiles[0]),np.percentile(array_ACC,percentiles[1]))

    return avg_ACC, std_ACC, percentiles_ACC, array_ACC

@jit(nopython=True,fastmath=True,parallel=True)
def BACC_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for Balanced Accuracy in the Graph Ensemble.

    It returns the ensemble average, standard deviation, percentiles and array for BACC.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    """
    
    n_countries = int(np.sqrt(len(aij_emp)))
    array_BACC = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_BACC[step] = BACC(aij_emp,aij)
        
    avg_BACC = array_BACC.mean()
    std_BACC = array_BACC.std()
    percentiles_BACC = (np.percentile(array_BACC,percentiles[0]),np.percentile(array_BACC,percentiles[1]))

    return avg_BACC, std_BACC, percentiles_BACC, array_BACC
    
@jit(nopython=True,fastmath=True,parallel=True)
def F1_score_ensemble(pij_mat,aij_emp,n_ensemble=1000,percentiles=(2.5,97.5)):
    """Computes various statistics for F1 Score in the Graph Ensemble.

    It returns the ensemble average, standard deviation, percentiles and array for F1score.

    :param pij_mat: expected binary adjacency matrix
    :type pij_mat: np.ndarray
    :param aij_emp: empirical binary adjacency matrix
    :type aij_emp: np.ndarray
    :param n_ensemble: Number of graphs in the ensemble. Defaults to 1000.
    :type n_ensemble: int, optional
    :param percentiles: Percentiles to compute. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    """
    
    n_countries = int(np.sqrt(len(aij_emp)))
    array_F1_score = np.empty(n_ensemble)
    
    for step in prange(n_ensemble):
        aij = np.zeros(len(pij_mat))
        for i in range(n_countries):
            for j in range(n_countries):
                if j!=i:
                    ij = i*n_countries + j
                    uniform = np.random.random()
                    pij = pij_mat[ij]
                    if uniform <= pij:
                        aij[ij] = 1
        array_F1_score[step] = F1_score(aij_emp,aij)
        
    avg_F1_score = array_F1_score.mean()
    std_F1_score = array_F1_score.std()
    percentiles_F1_score = (np.percentile(array_F1_score,percentiles[0]),np.percentile(array_F1_score,percentiles[1]))

    return avg_F1_score, std_F1_score, percentiles_F1_score, array_F1_score

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def degree_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for (out-)degree centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        array_stat[step,:] = deg(aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def degree_in_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for in-degree centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
 
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        array_stat[step,:] = deg_in(aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def annd_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for average neighbor degree centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        array_stat[step,:] = knn_single(aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat


@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def clust_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for binary clustering coefficient centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        array_stat[step,:] = clust_single_fast(aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat
        

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def st_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for (out-)strength centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        array_stat[step,:] = strength(wij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def st_in_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for in-strength centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        array_stat[step,:] = strength_in(wij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def anns_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for average neighbor strength centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        
        array_stat[step,:] = stnn_single(wij_model,aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat

  
   

@jit(nopython=True,fastmath=True,nogil=False,parallel=True)               
def cw_ensemble(w_mat_ensemble,percentiles=(2.5,97.5)):
    """Computes various statistics for (linear) weighted clustering coefficient centrality in the Graph Ensemble, consisting of ensemble average, standard deviation, 
    percentiles and ensemble distribution.


    :params w_mat_ensemble: weighted adjacency matrices for the Graph Ensemble
    :params percentiles: percentages for percentile CI extracted from the ensemble. Defaults to (2.5,97.5).
    :type w_mat_ensemble: np.ndarray
    :type percentiles: tuple, optional
    """
    
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    avg_stat = np.empty(n_countries)
    std_stat = np.empty(n_countries)
    ci_stat = np.empty((n_countries,2))
    
    array_stat = np.empty((n_ensemble,n_countries))
    
    for step in prange(n_ensemble):
        wij_model = w_mat_ensemble[:,step]
        aij_model = binarize(wij_model)
        
        array_stat[step,:] = clust_w_single_fast(wij_model,aij_model)
        
    array_stat = array_stat.T
    for i in prange(n_countries):
        avg_stat[i] = array_stat[i,:].mean()
        std_stat[i] = array_stat[i,:].std()
        ci_stat[i,0] = np.percentile(array_stat[i,:],percentiles[0])
        ci_stat[i,1] = np.percentile(array_stat[i,:],percentiles[1])
        
    return avg_stat, std_stat, ci_stat, array_stat


@jit(nopython=True,fastmath=True)               
def ensemble_coverage(w_mat_ensemble,wij_emp, percentiles=(2.5,97.5),stats=["degree","annd","clust","strength","anns","cw"]):
    """Computes Statistic Reproduction Accuracy for the network statistics in the stats-list
    
    :param w_mat_ensemble: weighted adjacency matrices in the graph ensemble
    :type w_mat_ensemble: np.ndarray
    :param wij_emp: empirical weighted adjacency matrix
    :type wij_emp: np.ndarray
    :param percentiles: percentages for ensemble percentiles. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    :param stats:  list of statistics to compute. Defaults to ["degree","annd","clust","strength","anns","cw"].
    :type stats: list, optional
    
    :return count_array: list of reproduction accuracies for the network statistics in input list stats.
    :rtype count_array: list
    """
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    aij_emp = binarize(wij_emp)
    count_array = np.empty(len(stats))
    
    if "degree" in stats:
        deg_emp = deg(aij_emp)
        avg_deg, std_deg, ci_deg, array_deg = degree_ensemble(w_mat_ensemble,percentiles)
    if "degree_in" in stats:
        deg_in_emp = deg_in(aij_emp)
        avg_deg_in, std_deg_in, ci_deg_in, array_deg_in = degree_in_ensemble(w_mat_ensemble,percentiles)
    if "annd" in stats:    
        annd_emp = knn_single(aij_emp)
        avg_annd, std_annd, ci_annd, array_annd = annd_ensemble(w_mat_ensemble,percentiles)
    if "clust" in stats:
        clust_emp = clust_single_fast(aij_emp)
        avg_clust, std_clust, ci_clust, array_clust = clust_ensemble(w_mat_ensemble,percentiles)
    if "strength" in stats:    
        st_emp = strength(wij_emp)
        avg_st, std_st, ci_st, array_st = st_ensemble(w_mat_ensemble,percentiles)
    if "strength_in" in stats:    
        st_in_emp = strength_in(wij_emp)
        avg_st_in, std_st_in, ci_st_in, array_st_in = st_in_ensemble(w_mat_ensemble,percentiles)
    if "anns" in stats:    
        anns_emp = stnn_single(wij_emp,aij_emp)
        avg_anns, std_anns, ci_anns, array_anns = anns_ensemble(w_mat_ensemble,percentiles)
    if "clust_w" in stats:
        cw_emp = clust_w_single_fast(wij_emp,aij_emp)
        avg_cw, std_cw, ci_cw, array_cw = cw_ensemble(w_mat_ensemble,percentiles)
    
    count_deg = 0.
    count_deg_in = 0.
    count_annd = 0.
    count_clust = 0.
    count_st = 0.
    count_st_in = 0.
    count_anns = 0.
    count_cw = 0.
    
    for i in range(n_countries):
        if "degree" in stats:
            if deg_emp[i] >= ci_deg[i,0] and deg_emp[i] <= ci_deg[i,1]:
                count_deg += 1./n_countries
        if "degree_in" in stats:
            if deg_in_emp[i] >= ci_deg_in[i,0] and deg_in_emp[i] <= ci_deg_in[i,1]:
                count_deg_in += 1./n_countries
        if "annd" in stats:
            if annd_emp[i] >= ci_annd[i,0] and annd_emp[i] <= ci_annd[i,1]:
                count_annd += 1./n_countries
        if "clust" in stats:
            if clust_emp[i] >= ci_clust[i,0] and clust_emp[i] <= ci_clust[i,1]:
                count_clust += 1./n_countries
        if "strength" in stats:
            if st_emp[i] >= ci_st[i,0] and st_emp[i] <= ci_st[i,1]:
                count_st += 1./n_countries
        if "strength_in" in stats:
            if st_in_emp[i] >= ci_st_in[i,0] and st_in_emp[i] <= ci_st_in[i,1]:
                count_st_in += 1./n_countries
        if "anns" in stats:
            if anns_emp[i] >= ci_anns[i,0] and anns_emp[i] <= ci_anns[i,1]:
                count_anns += 1./n_countries
        if "clust_w" in stats:
            if cw_emp[i] >= ci_cw[i,0] and cw_emp[i] <= ci_cw[i,1]:
                count_cw += 1./n_countries
        
    if "degree" in stats:
        index_degree = stats.index("degree")
        count_array[index_degree] = count_deg
    if "degree_in" in stats:
        index_degree_in = stats.index("degree_in")
        count_array[index_degree_in] = count_deg_in
    if "annd" in stats:
        index_annd = stats.index("annd")
        count_array[index_annd] = count_annd
    if "clust" in stats:
        index_clust = stats.index("clust")
        count_array[index_clust] = count_clust
    if "strength" in stats:
        index_strength = stats.index("strength")
        count_array[index_strength] = count_st
    if "strength_in" in stats:
        index_strength_in = stats.index("strength_in")
        count_array[index_strength_in] = count_st_in
    if "anns" in stats:
        index_anns = stats.index("anns")
        count_array[index_anns] = count_anns
    if "clust_w" in stats:
        index_cw = stats.index("clust_w")
        count_array[index_cw] = count_cw
            
    
    return count_array



@jit(nopython=True,fastmath=True,cache=False,parallel=False)               
def weighted_coverage(w_mat_ensemble,wij_emp, percentiles=(2.5,97.5)):
    """Computes Reproduction Accuracy for the weights
    
    :param w_mat_ensemble: weighted adjacency matrices in the graph ensemble
    :type w_mat_ensemble: np.ndarray
    :param wij_emp: empirical weighted adjacency matrix
    :type wij_emp: np.ndarray
    :param percentiles: percentages for ensemble percentiles. Defaults to (2.5,97.5).
    :type percentiles: tuple, optional
    
    :return count_array: list of reproduction accuracies for the weights.
    :rtype count_array: list
    """
    n_obs = w_mat_ensemble.shape[0]
    n_ensemble = w_mat_ensemble.shape[1]
    n_countries = int(np.sqrt(n_obs))
    
    w_mat_percentile = np.empty((n_obs,2))
    
    for ij in range(n_obs):
        w_mat_percentile[ij,0] = np.percentile(w_mat_ensemble[ij,:],percentiles[0])
        w_mat_percentile[ij,1] = np.percentile(w_mat_ensemble[ij,:],percentiles[1])
        
    
    count_w = 0.
    
    
    for ij in range(n_obs):
        
        
        if wij_emp[ij] >= w_mat_percentile[ij,0] and wij_emp[ij] <= w_mat_percentile[ij,1]:
            count_w += 1./n_obs
                    
    count_array = np.empty(1)
    count_array[0] = count_w
    
    return count_array



