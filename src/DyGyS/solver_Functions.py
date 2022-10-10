import numpy as np
from numba import jit,float64
from scipy import optimize as opt

from . import ll_Functions_conditional as lfC
from . import ll_Functions_integrated as lfI
from . import ll_Functions_econometrics as lfE

# @jit(float64[:](float64[:]),nopython=True,fastmath=True)
@jit(nopython=True)
def binarize(input):
    """Computes binary adjacency matrix from weighted adjacency matrix."""
    dim = len(input)
    byn = np.zeros(dim)
    for i in range(dim):
        if input[i]> 0:
            byn[i] = 1.
    return byn 

def solver(model,Wij,selection_variables,exogenous_variables,fixed_selection_params=np.array([]),tol=1e-5,
           use_guess= np.array([]), verbose=False, print_steps= 1, maxiter=20 ):  
    """Solves chosen model using scipy.optimize and scipy.least_squares routines.
    
    :param model: chosen model
    :type model: str
    :param Wij:  weighted adjacency matrix
    :type Wij: np,ndarray
    :param selection_variables: regressor matrix for the topological optimization of Zero-Inflated and L-constrained Conditional Models.
    :param exogenous_variables: regressor matrix for the weighted optimization
    :param fixed_selection_variables: fixed parameters for the topological optimization of Zero-Inflated and L-constrained Conditional Models. Defaults to np.array([1.00000,1.0000,0.0000]).
    :param tol: tolerance for optimization. Defaults to 1e-5.
    :param use_guess: optional starter guess for the optimization process. Defaults to np.array([]).
    :param verbose: True if you want to print iteration values of infinite norm. Defaults to False.
    :param print_steps: If verbose is True, you decide after how many steps you print on the screen. Defaults to 1.
    :param maxiter: Maxiter for optimization process. Defaults to 10.
    
    :type selection_variables: np.ndarray
    :type exogenous_variables: np.ndarray
    :type fixed_selection_variables: np.ndarray
    :type tol: float, optional
    :type use_guess: np.ndarray, optional
    :type verbose: bool, optional
    :type print_steps: int, optional
    :type maxiter: int, optional
    
    :raise TypeError: If chosen model is not correctly written or not implemented. See self.implemented_models to see available models.

    :return sol: estimated model solution
    :rtype sol: np.ndarray
    """
    
    if model == "Logit":
        n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess) == 0:
            sol = np.ones(n_select_variables)
        else:
            sol = use_guess    
        norm = 10000
        step = 0
        
        while norm > tol and step < maxiter:    
            
            sol = opt.least_squares(fun=lfC.jac_logit,x0 = sol, verbose=0,args=(Wij,selection_variables,fixed_selection_params)).x
            
            ll_lgt = - lfC.ll_logit(sol,Wij,selection_variables,fixed_selection_params)
            jac_lgt = - lfC.jac_logit(sol,Wij,selection_variables,fixed_selection_params)
            norm =np.linalg.norm(jac_lgt,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps==0:
                    print('iteration:', step, 'norm:', norm, 'll:',ll_lgt)
        
        return sol
        
    elif model == "UBCM":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            sol = np.random.uniform(-10,10,n_countries)
        else:
            sol = use_guess
            
        norm = 1000
        step = 0
        
        while norm > tol and step < maxiter:
            sol =  opt.least_squares(fun=lfC.jac_BCM,x0 = sol, args=(Wij,)).x
            ll_bcm = - lfC.ll_BCM(sol,Wij)
            jac_bcm = lfC.jac_BCM(sol,Wij)
            norm = np.linalg.norm(jac_bcm,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps==0:
                    print('iteration:',step,'norm:',norm,'ll:',ll_bcm)
        
        return sol
    
    elif model == "DBCM":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            sol = np.random.uniform(-10,10,2*n_countries)
        else:
            sol = use_guess
        norm = 1000
        step = 0
        
        while norm > tol and step < maxiter:
        
            sol = opt.least_squares(fun=lfC.jac_DBCM,x0 = sol, args=(Wij,)).x
            ll_bcm = -lfC.ll_DBCM(sol,Wij)
            jac_bcm = lfC.jac_DBCM(sol,Wij)
            norm = np.linalg.norm(jac_bcm,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps ==0:
                    print('iteration:', step,'norm:',norm, 'll:',ll_bcm)
                    
        return sol
    
    elif model == "POIS":
        if len(use_guess) ==0:

            beta_0 = lfE.guess_constant_parameter(Wij,exogenous_variables)
            # print("beta_0: ",- beta_0)
            guess_aux = np.zeros(exogenous_variables.shape[1]-1)
            guess_UPOIS = np.array([- beta_0])
            sol = np.concatenate((guess_UPOIS,guess_aux))
            
        else:
            sol = use_guess
        
        norm = 1000
        step = 0
        
        while norm > tol and step < maxiter:
        
            sol = opt.minimize(fun=lfE.ll_POIS,x0=sol,method='Nelder-Mead',args=(Wij,exogenous_variables)).x
            sol = opt.least_squares(fun=lfE.jac_POIS,x0=sol,verbose=0,args=(Wij,exogenous_variables)).x
            ll_POIS = - lfE.ll_POIS(sol,Wij,exogenous_variables)
            jac_POIS = - lfE.jac_POIS(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac_POIS,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps ==0:
                    print('iteration:', step,'norm:',norm,'ll:',ll_POIS)
                    
            
        return sol
    
    elif model == "NB2":
        if len(use_guess) == 0:
            sol = np.ones(exogenous_variables.shape[1]+1)
        
        else:
            sol = use_guess
        
        
        norm = 1000
        step = 0
        
        while norm > tol and step < maxiter:
            
            sol = opt.minimize(fun=lfE.ll_NB2,x0=sol,method='Nelder-Mead',args=(Wij,exogenous_variables)).x
            sol = opt.least_squares(fun=lfE.jac_NB2,x0=sol,verbose=0,args=(Wij,exogenous_variables)).x
            ll_NB2 = - lfE.ll_NB2(sol,Wij,exogenous_variables)
            jac_NB2 = lfE.jac_NB2(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac_NB2,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm:',norm,'ll:',ll_NB2)
        
        return sol    
    
    
    elif model == "ZIP":
        n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess) == 0:
            sol = np.random.random(n_select_variables + exogenous_variables.shape[1]) 
        else:
            sol = use_guess
        norm = 1000
        step = 0
        
        while norm > tol and step < maxiter:
            # try:
            #     sol = opt.minimize(fun = lfE.ll_ZIP,jac=lfE.jac_ZIP, x0 = sol, method="BFGS",
            #                                         args=(Wij,selection_variables,exogenous_variables,fixed_selection_params)).x
            # except ZeroDivisionError:
            #     pass
            
            try:
                sol = opt.least_squares(fun = lfE.jac_ZIP, x0 = sol, verbose=0, 
                                                    args=(Wij,selection_variables,exogenous_variables,fixed_selection_params)).x
            except ZeroDivisionError:
                pass
            ll_ZIP = -lfE.ll_ZIP(sol,Wij,selection_variables,exogenous_variables,fixed_selection_params)
            jac = lfE.jac_ZIP(sol,Wij,selection_variables,exogenous_variables,fixed_selection_params)
            norm = np.linalg.norm(jac,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm:',norm,'ll:',ll_ZIP)
                    
        return sol
    
    elif model == "ZINB":
        n_select_variables = selection_variables.shape[1] - len(fixed_selection_params)
        n_exog_variables = exogenous_variables.shape[1]
        if len(use_guess)==0:
            sol = np.ones(n_select_variables+n_exog_variables+1)
        else:
            sol = use_guess    
        
        step = 0
        norm = 10000
        while norm > tol and step < maxiter:
            try:
                sol = opt.minimize(fun = lfE.ll_ZINB, x0 = sol, method='Nelder-Mead',
                                                    args=(Wij,selection_variables,exogenous_variables,fixed_selection_params)).x
            except ZeroDivisionError:
                pass
            try:
                sol = opt.minimize(fun = lfE.ll_ZINB, jac = lfE.jac_ZINB,x0 = sol, method='BFGS',
                                                    args=(Wij,selection_variables,exogenous_variables,fixed_selection_params)).x
            except ZeroDivisionError:
                pass
            try:
                sol = opt.least_squares(fun = lfE.jac_ZINB, x0 = sol, 
                                                    args=(Wij,selection_variables,exogenous_variables,fixed_selection_params)).x
            except ZeroDivisionError:
                pass
            norm = np.linalg.norm(lfE.jac_ZINB(sol,Wij,selection_variables,exogenous_variables,fixed_selection_params),ord=np.inf)
            ll = - lfE.ll_ZINB(sol,Wij,selection_variables,exogenous_variables,fixed_selection_params)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm:',norm,'ll:',ll)
            
            
             
        return sol
    
    elif model == "CGeom":
        guess = solver("POIS",Wij,selection_variables,exogenous_variables,verbose=False)
        guess_b0 = np.ones(1)
        sol = np.concatenate((guess,guess_b0))
        result_linspace = []
        norm_linspace = []
        inf_bounds_ls = []
        sup_bounds_ls = []
        ll_linspace = []
        for i in range(len(sol)):
            inf_bounds_ls.append(-np.inf)
            sup_bounds_ls.append(np.inf)
        inf_bounds_ls[-1] = 1e-8
        
        bounds_ls = (inf_bounds_ls,sup_bounds_ls)
        step = 0
        norm = 10000
        while norm > tol and step < maxiter:
            
            try:
                sol = opt.minimize(fun=lfC.ll_CGeom,args=(Wij,exogenous_variables),x0 = sol, method='Nelder-Mead').x
            except:
                pass
            
            try:
                sol = opt.least_squares(fun=lfC.jac_CGeom,args=(Wij,exogenous_variables),x0 = sol,bounds=bounds_ls).x
            except:
                pass
            ll = lfC.ll_CGeom(sol,Wij,exogenous_variables)
            jac = lfC.jac_CGeom(sol,Wij,exogenous_variables)
            # print('iterazione',step,'jac',np.abs(jac))
            norm = np.linalg.norm(jac,ord=np.inf)
            
            result_linspace.append(sol)
            norm_linspace.append(norm)
            ll_linspace.append(ll)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm_w:',norm,'ll_w:',-ll)
                
            if norm < tol:
                if verbose:
                    print("Cond Geom: ",norm)
                    
                return sol 
                    
        minIndex = norm_linspace.index(min(norm_linspace))
        minIndex = ll_linspace.index(min(ll_linspace))
        
        true_result = result_linspace[minIndex]
        return true_result
    
    elif model == "CExp":
        sol = np.ones(exogenous_variables.shape[1]+1)
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        
        step = 0
        norm = 10000
        while norm > tol and step < maxiter:
        
            sol = opt.minimize(fun=lfC.ll_CExp,method="Nelder-Mead",args=(Wij,exogenous_variables),x0 = sol).x
            
            sol = opt.least_squares(fun=lfC.jac_CExp,args=(Wij,exogenous_variables),x0 = sol,verbose=0).x
            ll = lfC.ll_CExp(sol,Wij,exogenous_variables)
            jac = lfC.jac_CExp(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)    
            result_linspace.append(sol)
            norm_linspace.append(norm)
            ll_linspace.append(ll)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm_w:',norm,'ll_w:',-ll)
            
            
            if norm < tol:
                return sol
                    
        # minIndex = norm_linspace.index(min(norm_linspace))
        minIndex = ll_linspace.index(min(ll_linspace))
        
        true_result = result_linspace[minIndex]
            
        return true_result
            
    elif model == "CPareto":
        sol = solver("POIS",Wij,selection_variables,exogenous_variables,verbose=False)
        ll =  lfC.ll_CPareto(sol,Wij,exogenous_variables)
        # sol = np.zeros(exogenous_variables.shape[1])
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        step = 0
        norm = 10000
        
        while norm > tol and step < maxiter:
            sol = opt.least_squares(fun=lfC.jac_CPareto,args=(Wij,exogenous_variables),
            x0 = sol).x
            ll = lfC.ll_CPareto(sol,Wij,exogenous_variables)
            jac = lfC.jac_CPareto(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)    
            result_linspace.append(sol)
            norm_linspace.append(norm)
            ll_linspace.append(ll)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm_w:',norm,'ll_w:',-ll)
            
            if norm < tol:
                if verbose:
                    print("Cond Pareto: ", norm)
                return sol
                    
        # minIndex = norm_linspace.index(min(norm_linspace))
        minIndex = ll_linspace.index(min(ll_linspace))
        
        true_result = result_linspace[minIndex]
            
        return true_result
                
    elif model == "CLognormal":
        sol = np.ones(exogenous_variables.shape[1]+1)
        step = 0
        norm = 10000
        while norm > tol and step < maxiter:
            
            sol = opt.least_squares(fun=lfC.jac_CLognormal,x0=sol,args=(Wij,exogenous_variables),
                                    ).x
            jac = lfC.jac_CLognormal(sol,Wij,exogenous_variables)
            ll =  lfC.ll_CLognormal(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm_w:',norm,'ll_w:',-ll)
            
        
        return sol

    elif model == "CGamma":
        sol = np.ones(exogenous_variables.shape[1]+2)
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        step = 0
        norm = 10000
        while norm > tol and step < maxiter:
            try:    
                sol = opt.least_squares(fun=lfC.jac_CGamma,args=(Wij,exogenous_variables),x0 = sol,verbose=0).x
            except ZeroDivisionError:
                pass
            
            ll = lfC.ll_CGamma(sol,Wij,exogenous_variables)
            jac = lfC.jac_CGamma(sol,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)  
            # print('iterazione',it,'jac',jac)
            ll_linspace.append(ll)  
            result_linspace.append(sol)
            norm_linspace.append(norm)
            step += 1
            
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'norm_w:',norm,'ll_w:',-ll)
            
        
            if norm < tol:
                return sol
                    
        minIndex = norm_linspace.index(min(norm_linspace))
        # minIndex = ll_linspace.index(min(ll_linspace))
        true_result = result_linspace[minIndex]
            
        return true_result
                        
    elif model == "L-CGeom":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess)==0:
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps,use_guess=np.array([]))
            
        else:  
            
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps, use_guess=use_guess[:n_selection_variables])
            

        if len(use_guess) == 0:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,print_steps = print_steps)
        else:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,print_steps = print_steps,
                                      use_guess=use_guess[n_selection_variables:])
                
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
                
    elif model == "k-CGeom_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,maxiter=maxiter,verbose=verbose,print_steps = print_steps,use_guess=use_guess[n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
            
    elif model == "k-CGeom_directed":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params=np.array([1.00000,1.0000,0.0000]),tol=1e-5,
           use_guess= np.array([]),verbose=verbose,maxiter=maxiter)
        else:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params=np.array([1.00000,1.00000,0.0000]),tol=1e-5,
           use_guess= use_guess[:2*n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CGeom",Wij,selection_variables,exogenous_variables,maxiter=maxiter,verbose=verbose,print_steps = print_steps,
                                      use_guess=use_guess[2*n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
        
    elif model == "L-CExp":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess)==0:
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,
                                        print_steps = print_steps,use_guess=np.array([]))
            
        else:  
            
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,
                                        print_steps = print_steps,use_guess=use_guess[:n_selection_variables])
            

        if len(use_guess) == 0:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,
                                      print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,
                                      print_steps = print_steps,verbose=verbose,
                                      use_guess=use_guess[n_selection_variables:])
                
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
                    
    elif model == "k-CExp_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,maxiter=maxiter)
        else:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,print_steps = print_steps,use_guess=use_guess[n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
                       
    elif model == "k-CExp_directed":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:2*n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CExp",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,print_steps = print_steps,use_guess=use_guess[2*n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result

    elif model == "L-CPareto":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess)==0:
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps,use_guess=np.array([]))
            
        else:  
            
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps,use_guess=use_guess[:n_selection_variables])
            

        if len(use_guess) == 0:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[n_selection_variables:])
                
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
        
    elif model == "k-CPareto_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "k-CPareto_directed":
        
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:2*n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CPareto",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[2*n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
        
        
        
    elif model == "L-CGamma":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess)==0:
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,print_steps = print_steps,verbose=verbose,use_guess=np.array([]))
            
        else:  
            
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,print_steps = print_steps,verbose=verbose,use_guess=use_guess[:n_selection_variables])
            

        if len(use_guess) == 0:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[n_selection_variables:])
                
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "k-CGamma_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,maxiter=maxiter)
        else:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:n_countries],verbose=verbose,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,print_steps = print_steps,maxiter=maxiter,verbose=verbose)
        else:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "k-CGamma_directed":
        
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:2*n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose)
        else:
            cond_Geom_params = solver("CGamma",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,use_guess=use_guess[2*n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
        
    elif model == "L-CLognormal":
        n_selection_variables = selection_variables.shape[1] - len(fixed_selection_params)
        if len(use_guess)==0:
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps,use_guess=np.array([]))
            
        else:  
            
            topological_params = solver("Logit",Wij,selection_variables,exogenous_variables,fixed_selection_params,verbose=verbose,print_steps = print_steps,use_guess=use_guess[:n_selection_variables])
            

        if len(use_guess) == 0:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,verbose=verbose,print_steps = print_steps,use_guess=use_guess[n_selection_variables:])
                
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "k-CLognormal_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "k-CLognormal_directed":
        
        n_countries = int(np.sqrt(len(Wij)))
        if len(use_guess) == 0:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=verbose,print_steps = print_steps,maxiter=maxiter)
        else:
            topological_params = solver("DBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= use_guess[:2*n_countries],verbose=verbose,print_steps = print_steps,maxiter=maxiter)
            
        if len(use_guess) == 0:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose)
        else:
            cond_Geom_params = solver("CLognormal",Wij,selection_variables,exogenous_variables,fixed_selection_params,maxiter=maxiter,print_steps = print_steps,verbose=verbose,use_guess=use_guess[2*n_countries:])
            
        result = np.concatenate((topological_params,cond_Geom_params))
        return result
    
    elif model == "L-IGeom":
        guess_weighted = solver("POIS",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
           use_guess= np.array([]),verbose=False,maxiter=maxiter)
        guess_theta_0 = np.array([6.])      
        beta_linspace = np.linspace(0.99,1.,num=5)[:-1]
        result_linspace = []
        norm_linspace = []
        ll_linspace = []

        for lin in range(len(beta_linspace)):  
        
            guess_beta_0 = np.array([beta_linspace[lin]])
            sol = np.concatenate((guess_theta_0,guess_weighted,guess_beta_0))
            if len(use_guess)!=0:
                guess = use_guess
            norm = 10000
            step = 0
            while norm > tol and step < maxiter:    
                sol = opt.minimize(fun=lfI.ll_LIGeom,method="Nelder-Mead",args=(Wij,exogenous_variables),x0=sol).x
                sol = opt.least_squares(fun=lfI.jac_LIGeom,args=(Wij,exogenous_variables),x0 = sol).x
                jac = - lfI.jac_LIGeom(sol,Wij,exogenous_variables)
                ll =  lfI.ll_LIGeom(sol,Wij,exogenous_variables)
                norm = np.linalg.norm(jac,ord=np.inf)
                result_linspace.append(sol)
                norm_linspace.append(norm)
                ll_linspace.append(ll)
                step += 1
                if np.isnan(ll):
                    break
                if verbose:
                    if step % print_steps == 0:
                        print("round",lin,'iteration:',step,"norm",norm,'ll:',-ll)
                        
                if norm < tol:
                    return sol
                if len(norm_linspace) > 2:
                    if norm_linspace[-1] == norm_linspace[-2]:
                        break
                
                    
        minIndex = norm_linspace.index(min(norm_linspace))
        # minIndex = ll_linspace.index(min(ll_linspace))
        true_result = result_linspace[minIndex]
        true_norm = min(norm_linspace)
        true_jac = lfI.jac_LIGeom(true_result,Wij,exogenous_variables)
                    
        return true_result

    elif model == "L-IExp":
        guess_weighted = solver("POIS",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
        use_guess= np.array([]),verbose=False,maxiter=maxiter)
        guess_theta_0 = np.array([6.])      
        # beta_linspace = np.linspace(0.99,1.,num=5)[:-1]
        beta_linspace = np.array([0.99])
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        
        norm = 10000
        step = 0
        
        for lin in range(len(beta_linspace)):  
            guess_beta_0 = np.array([beta_linspace[lin]])
            sol = np.concatenate((guess_theta_0,guess_weighted,guess_beta_0))
            
            norm = 10000
            step = 0
            while norm > tol and step < maxiter:
                
                sol = opt.minimize(fun=lfI.ll_LIExp,method="Nelder-Mead",args=(Wij,exogenous_variables),x0=sol).x
                sol = opt.least_squares(fun=lfI.jac_LIExp,args=(Wij,exogenous_variables),x0 = sol).x
                jac = lfI.jac_LIExp(sol,Wij,exogenous_variables)
                ll =  lfI.ll_LIExp(sol,Wij,exogenous_variables)
                norm = np.linalg.norm(jac,ord=np.inf)
                if np.isnan(ll):
                    break
                step += 1
                if verbose:
                    if step % print_steps == 0:
                        
                        print("round",lin,'iteration:',step,"norm",norm,'ll:',-ll)
                        
                if norm < tol:
                    if verbose:
                        print("LIA1: ",norm)
                    return sol
                result_linspace.append(sol)
                norm_linspace.append(norm)
                ll_linspace.append(ll)
                
                if len(norm_linspace) > 2:
                    if norm_linspace[-1] == norm_linspace[-2]:
                        break
                
                
        minIndex = ll_linspace.index(min(ll_linspace))
        true_result = result_linspace[minIndex]
                    
        return true_result

    elif model == "k-IGeom_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        n_select_variables = n_countries
        Aij = binarize(Wij)
        
        if len(use_guess) == 0:
            guess_theta_0 = solver("UBCM",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess_other = solver("L-IGeom",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess = np.concatenate((guess_theta_0,guess_other[1:]))
            
        if len(use_guess) != 0:
            guess = use_guess
        
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        
        norm = 100000
        step = 0
        while step < maxiter and norm > tol:
            
            try:
                guess = opt.minimize(fun=lfI.ll_kIGeom_undirected,jac=lfI.jac_kIGeom_undirected,method="BFGS",
                                        args=(Wij,exogenous_variables),x0=guess).x
            except ZeroDivisionError:
                pass
            
            
            try:
                guess_beta_0 = np.array([guess[-1]])
                guess_theta_0 = guess[:n_countries]
                guess_weighted = guess[n_countries:-1]
                theta_0 = opt.least_squares(fun=lfI.jac_kIGeom_topological_undirected,jac=lfI.hess_kIGeom_topological_undirected,
                                                x0 = guess_theta_0, verbose=0,
                                            args=(Wij,exogenous_variables, guess_weighted,guess_beta_0))
                
                guess = np.concatenate((theta_0.x,guess_weighted,guess_beta_0))
            except ZeroDivisionError:
                pass
            
            
            
            try:
                guess = opt.least_squares(fun=lfI.jac_kIGeom_undirected,args=(Wij,exogenous_variables),
                                                    x0 = guess,verbose=0,max_nfev=100).x
            except ZeroDivisionError:
                pass
            guess_theta_0 = guess[:n_countries]
            guess_weighted = guess[n_countries:-1]
            guess_beta_0 = np.array([guess[-1]])
            
            jac = lfI.jac_kIGeom_undirected(guess,Wij,exogenous_variables)
            ll =  lfI.ll_kIGeom_undirected(guess,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)
            norm_top = np.linalg.norm(jac[:n_countries],ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'ll:',-ll,'norm_top:',norm_top,'norm:',norm)
                    
            if np.isnan(ll):
                break
                    
            if norm < tol:
                return guess
            result_linspace.append(guess)
            norm_linspace.append(norm)
            ll_linspace.append(-ll)
            if len(norm_linspace) > 2:
                if norm_linspace[-1] == norm_linspace[-2]:
                    break
                    
                        
        minIndex = ll_linspace.index(min(ll_linspace))
        true_result = result_linspace[minIndex]
                    
        return true_result

    elif model == "k-IGeom_directed":
        n_countries = int(np.sqrt(len(Wij)))
        n_select_variables = n_countries
        Aij = binarize(Wij)
        
        if len(use_guess) == 0:
            guess_theta_0 = solver("DBCM",Wij,selection_variables,exogenous_variables,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess_other = solver("L-IGeom",Wij,selection_variables,exogenous_variables,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess = np.concatenate((guess_theta_0,guess_other[1:]))
            
            
            
            
        if len(use_guess) != 0:
            guess = use_guess
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
            
        norm = 100000
        step = 0
        while norm > tol and step < maxiter:
        
            
            try:
                guess = opt.minimize(fun=lfI.ll_kIGeom_directed,jac=lfI.jac_kIGeom_directed,method="BFGS",
                                        args=(Wij,exogenous_variables),x0=guess).x
            except ZeroDivisionError:
                pass
            
            
            try:
                guess_beta_0 = np.array([guess[-1]])
                guess_theta_0 = guess[:2*n_countries]
                guess_weighted = guess[2*n_countries:-1]
                theta_0 = opt.least_squares(fun=lfI.jac_kIGeom_topological_directed,
                                                x0 = guess_theta_0, verbose=0,
                                            args=(Wij,exogenous_variables, guess_weighted,guess_beta_0))
                
                guess = np.concatenate((theta_0.x,guess_weighted,guess_beta_0))
            except ZeroDivisionError:
                pass
            
            
            
            try:
                guess = opt.least_squares(fun=lfI.jac_kIGeom_directed,args=(Wij,exogenous_variables),
                                                    x0 = guess,verbose=0,max_nfev=100).x
            except ZeroDivisionError:
                pass
            guess_theta_0 = guess[:2*n_countries]
            guess_weighted = guess[2*n_countries:-1]
            guess_beta_0 = np.array([guess[-1]])
            
            jac = lfI.jac_kIGeom_directed(guess,Wij,exogenous_variables)
            ll =  lfI.ll_kIGeom_directed(guess,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)
            norm_top = np.linalg.norm(jac[:n_countries],ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'ll:',-ll,'norm_top:',norm_top,'norm:',norm)
                    
            if np.isnan(ll):
                break
                    
            if norm < tol:
                return guess
            result_linspace.append(guess)
            norm_linspace.append(norm)
            ll_linspace.append(-ll)
            if len(norm_linspace) > 2:
                if norm_linspace[-1] == norm_linspace[-2]:
                    break
                    
                        
        minIndex = norm_linspace.index(min(norm_linspace))
        minIndex = ll_linspace.index(min(ll_linspace))
        
        true_result = result_linspace[minIndex]
                    
        return true_result
    
    
    elif model == "k-IExp_undirected":
        n_countries = int(np.sqrt(len(Wij)))
        n_select_variables = n_countries
        Aij = binarize(Wij)
        if len(use_guess) == 0:
            Exp_TS = solver("k-CExp_undirected",Wij,selection_variables,exogenous_variables,fixed_selection_params,tol=1e-5,
                use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess_weighted = Exp_TS[n_countries:-1]
            guess_theta_0 = Exp_TS[:n_countries]
            guess = Exp_TS
        if len(use_guess) != 0:
            guess = use_guess
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
        norm = 10000
        step = 0
        while norm > tol and step < maxiter:
            
            try:
                guess_beta_0 = np.array([guess[-1]])
                guess_theta_0 = guess[:n_countries]
                guess_weighted = guess[n_countries:-1]
                theta_0 = opt.least_squares(fun=lfI.jac_kIExp_topological_undirected,
                                                x0 = guess_theta_0, verbose=0,
                                            args=(Wij,exogenous_variables, guess_weighted,guess_beta_0))
                
                guess = np.concatenate((theta_0.x,guess_weighted,guess_beta_0))
            except ZeroDivisionError:
                pass    
            try:
                guess = opt.minimize(fun=lfI.ll_kIExp_undirected,jac=lfI.jac_kIExp_undirected,method='BFGS',
                                            args=(Wij,exogenous_variables),x0=guess).x
            except ZeroDivisionError:
                pass
            
            try:
                guess = opt.least_squares(fun=lfI.jac_kIExp_undirected,args=(Wij,exogenous_variables),
                                                    x0 = guess,verbose=0,max_nfev=100).x
            except ZeroDivisionError:
                pass    
            
            # guess = result_minimize.x
            guess_theta_0 = guess[:n_countries]
            guess_weighted = guess[n_countries:-1]
            guess_beta_0 = np.array([guess[-1]])
            
            jac = lfI.jac_kIExp_undirected(guess,Wij,exogenous_variables)
            jac_top = lfI.jac_kIExp_topological_undirected(guess_theta_0,Wij,exogenous_variables,guess_weighted,guess_beta_0)
            
            ll =  lfI.ll_kIExp_undirected(guess,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)
            norm_top = np.linalg.norm(jac_top,ord=np.inf)
            
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'ll:',-ll,'norm_top:',norm_top,'norm:',norm)
                    
            if np.isnan(ll):
                break
            if norm < tol:
                return guess
            result_linspace.append(guess)
            norm_linspace.append(norm)
            ll_linspace.append(ll)
            if len(norm_linspace) > 2:
                if norm_linspace[-1] == norm_linspace[-2]:
                    break
                    
                        
        minIndex = ll_linspace.index(min(ll_linspace))
        true_result = result_linspace[minIndex]
        true_norm = min(norm_linspace)
                    
        return true_result
    
    elif model == "k-IExp_directed":
        n_countries = int(np.sqrt(len(Wij)))
        n_select_variables = n_countries
        Aij = binarize(Wij)
        
        if len(use_guess) == 0:
            guess_theta_0 = solver("DBCM",Wij,selection_variables,exogenous_variables,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess_other = solver("L-IExp",Wij,selection_variables,exogenous_variables,tol=1e-5,
                    use_guess= np.array([]),verbose=False,maxiter=maxiter)
            guess = np.concatenate((guess_theta_0,guess_other[1:]))
            
            
            
            
        if len(use_guess) != 0:
            guess = use_guess
        result_linspace = []
        norm_linspace = []
        ll_linspace = []
    
        norm = 10000
        step = 0
        while norm > tol and step < maxiter:
                
            try:
                guess = opt.minimize(fun=lfI.ll_kIExp_directed,jac=lfI.jac_kIExp_directed,method="BFGS",
                                        args=(Wij,exogenous_variables),x0=guess).x
            except ZeroDivisionError:
                pass
            
            
            try:
                guess_beta_0 = np.array([guess[-1]])
                guess_theta_0 = guess[:2*n_countries]
                guess_weighted = guess[2*n_countries:-1]
                theta_0 = opt.least_squares(fun=lfI.jac_kIExp_topological_directed,
                                                x0 = guess_theta_0, verbose=0,
                                            args=(Wij,exogenous_variables, guess_weighted,guess_beta_0))
                
                guess = np.concatenate((theta_0.x,guess_weighted,guess_beta_0))
            except ZeroDivisionError:
                pass
            
            
            
            try:
                guess = opt.least_squares(fun=lfI.jac_kIExp_directed,args=(Wij,exogenous_variables),
                                                    x0 = guess,verbose=0,max_nfev=100).x
            except ZeroDivisionError:
                pass
            guess_theta_0 = guess[:2*n_countries]
            guess_weighted = guess[2*n_countries:-1]
            guess_beta_0 = np.array([guess[-1]])
            
            jac = lfI.jac_kIExp_directed(guess,Wij,exogenous_variables)
            ll =  lfI.ll_kIExp_directed(guess,Wij,exogenous_variables)
            norm = np.linalg.norm(jac,ord=np.inf)
            norm_top = np.linalg.norm(jac[:n_countries],ord=np.inf)
            step += 1
            if verbose:
                if step % print_steps == 0:
                    print('iteration:',step,'ll:',-ll,'norm_top:',norm_top,'norm:',norm)
                    
            if np.isnan(ll):
                break
                    
            if norm < tol:
                return guess
            result_linspace.append(guess)
            norm_linspace.append(norm)
            ll_linspace.append(-ll)
            
            if len(norm_linspace) > 2:
                if norm_linspace[-1] == norm_linspace[-2]:
                    break
                    
                        
        minIndex = norm_linspace.index(min(norm_linspace))
        minIndex = ll_linspace.index(min(ll_linspace))
        
        true_result = result_linspace[minIndex]
                    
        return true_result
    

    else:
        raise TypeError('model wrongly defined. See the available models using .implemented_models')
            
            
