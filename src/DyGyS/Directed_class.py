import numpy as np
from . import ll_Functions_conditional as lfC
from . import ll_Functions_integrated as lfI
from . import ll_Functions_econometrics as lfE
from . import solver_Functions as sF
from . import netstats_Functions as nF
from . import ensemble_Functions as eF
from numba.typed import List

class DirectedGraph:
    
    """"Directed Graph instance must be initialised with adjacency weighted matrix. An exploratory analysis is conducted during initialization.
    See .implemented_models, .implemented_network_statistics and .implemented_classifier_statistics for available methods and statistics.
    
    :params adjacency (np.ndarray,list): Weighted adjacency matrix in numpy 1D, numpy 2D or list format.
    
    """
    
    
    def __init__(
        self,
        adjacency=None,
    ):
        if adjacency is not None:
            if isinstance(adjacency,(np.ndarray,)):
                
                if len(adjacency.shape) == 2:
                    adjacency = nF.flatten(adjacency)
                elif len(adjacency.shape) > 2:
                    raise TypeError("Adjacency must be a 1-D or 2-D numpy array")
                
                        
            if not isinstance(adjacency,(np.ndarray,list)):
                raise TypeError("Adjacency must be a 1-D or 2-D numpy array")
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be semi-definite positive."
                    )
                if isinstance(
                    adjacency, list
                ):
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
        else:
            raise TypeError("Adjacency matrix must be initialized.")
        
        self.binary_adjacency = nF.binarize(adjacency)
        
        self.degree = nF.deg(self.binary_adjacency)
        self.degree_in = nF.deg_in(self.binary_adjacency)
        self.annd = nF.knn_single(self.binary_adjacency)
        self.clust = nF.clust_single_fast(self.binary_adjacency)
        self.strength = nF.strength(adjacency)
        self.strength_in = nF.strength_in(adjacency)
        self.anns = nF.stnn_single(adjacency,self.binary_adjacency)
        self.clust_w = nF.clust_w_single_fast(adjacency,self.binary_adjacency)
        self.n_edges = self.binary_adjacency.sum()
        self.n_nodes = int(np.sqrt(len(adjacency)))
        
        
        
        self.implemented_models = ["POIS","ZIP","NB2","ZINB","L-CGeom","k-CGeom","L-IGeom","k-IGeom",
                        "L-CExp","k-CExp","L-IExp","k-IExp","L-CPareto","k-CPareto","L-CGamma","k-CGamma","L-CLognormal","k-CLognormal"]
        self.discrete_models = ["POIS","ZIP","NB2","ZINB","L-IGeom","k-IGeom","L-CGeom","k-CGeom"]
        self.continuous_models = ["L-IExp","k-IExp","k-CExp","L-CExp","L-CPareto","k-CPareto","L-CGamma","k-CGamma","L-CLognormal","k-CLognormal"]
        self.implemented_network_statistics = ["degree","degree_in","annd","clust","strength","strength_in","anns","clust_w"]
        self.implemented_classifier_statistics = ["TPR","SPC","PPV","ACC","BACC","F1_score"]
        
        
        
        # Model Matrices
        self.model_weighted_adjacency = None
        self.model_binary_adjacency = None

        # Problem solutions
        self.params = None
        self.model = None
        self.optimization_time = None
        
        # Problem (reduced) residuals
        self.ll = None
        self.ll_binary = None
        self.jacobian = None
        self.norm = None
        self.aic = None
        self.aic_binary = None
        self.bic = None
        self.export_name = None

        
        # function
        self.args = None

        #classifier ensembler
        self.avg_nlinks = None
        self.std_nlinks = None
        self.avg_TPR = None
        self.std_TPR = None
        self.avg_SPC = None
        self.std_SPC = None
        self.avg_PPV = None
        self.std_PPV = None
        self.avg_ACC = None
        self.std_ACC = None
        self.avg_BACC = None
        self.std_BACC = None
        self.avg_F1score = None
        self.std_F1score = None

        #netstats ensembler
        self.avg_degree = None
        self.avg_annd = None
        self.avg_clust = None
        self.avg_strength = None
        self.avg_anns = None
        self.avg_clust_w = None
        
        self.std_degree = None
        self.std_annd = None
        self.std_clust = None
        self.std_strength = None
        self.std_anns = None
        self.std_clust_w = None

        #netstats_direct
        self.model_degree = None
        self.model_annd = None
        self.model_clust = None
        self.model_strength = None
        self.model_anns = None
        self.model_clust_w = None
        
        self.adjacency = adjacency  

        #weighted_prediction
        self.cond_w_score = None          
                
                     
        
    
    def solver(
        self,
        model,
        exogenous_variables = np.array([]),
        selection_variables = np.array([]),
        fixed_selection_params = np.array([]),
        verbose = False,
        print_steps = 1,
        imported_params = np.array([]),
        use_guess = np.array([]),
        maxiter = 10,
        tol = 1e-5
    ):
        """Optimize chosen model for Directed Graphs and compute model selection measures such as AIC and BIC. 
         
        :param model: Chosen model
        :type model: string
        
        :param exogenous_variables: Exogenous Variables used for the weighted step, in numpy 2-D format of the type N X k where N are the observations and k are the params,
        :type exogenous_variables: np.ndarray 
        :param selection_variables: Exogenous Variables used for the topological step, in numpy 2-D format of the type N X k where N are the observations and k are the params,
        :type selection_variables: np.ndarray 
        
        :param fixed_selection_params: Optional Constant Parameters for the topological step. When initialized different from null vector, it fixes the last "n" topological parameters, 
                                    when selection_variables are used and n is the size of fixed_selection_params 
        :type fixed_selection_params: np.ndarray 
        
        :param verbose: True if you want to see every n*print_steps iterations, default is False
        :type verbose: boolean
        
        :param print_steps: If verbose is True, you print on screen every n*print_steps iterations. default is 1
        :type print_steps: int
        
        :param print_steps: number of steps for which you can investigate iterations, if Verbose is True
        :type print_steps: int 
        
        :param imported_params: If used, uses wanted params as solution.
        :type imported_params: np.ndarray 
        
        :param use_guess: If used, uses wanted params as starters in the optimization.
        :type use_guess: np.ndarray 
        
        :param maxiter: Maximum Iterations of solver function.
        :type maxiter: int
        
        :param tol: tolerance for infinite norm in the optimization process
        :type tol: float 
        
        """
        self.model = model
        self.exogenous_variables = exogenous_variables
        self.selection_variables = selection_variables
        self.fixed_selection_params = fixed_selection_params
        n_countries = int(np.sqrt(len(self.adjacency)))
          
        if model in ["ZIP", "ZINB", "L-CGeom", "L-CExp", "L-CPareto",  "L-CGamma", "L-CLognormal"]:
            self.args = (self.adjacency,self.selection_variables,self.exogenous_variables,self.fixed_selection_params)
        elif model == "Logit":
            self.args =   (self.adjacency,self.selection_variables,self.fixed_selection_params)     
        elif model in ["UBCM","DBCM"]:
            self.args = (self.adjacency,)
        else:
            self.args = (self.adjacency,self.exogenous_variables)

        self.conditional_models = ["L-CGeom","k-CGeom","L-CExp","k-CExp","L-CPareto","k-CPareto","L-CGamma","k-CGamma","L-CLognormal","k-CLognormal"]
        
        ll_fun = {
            "Logit": lambda x: -lfC.ll_logit(x, self.adjacency,self.selection_variables,self.fixed_selection_params),
            "UBCM": lambda x: -lfC.ll_BCM(
                x, self.adjacency),
            "DBCM": lambda x: -lfC.ll_DBCM(
                x, self.adjacency),
            "POIS": lambda x: -lfE.ll_POIS(x, *self.args),
            "ZIP": lambda x: -lfE.ll_ZIP(
                x, *self.args),
            "NB2": lambda x: -lfE.ll_NB2(x, *self.args),
            "ZINB": lambda x: -lfE.ll_ZINB(
                x, *self.args),
            "L-IGeom": lambda x: -lfI.ll_LIGeom(x, *self.args),
            "k-IGeom": lambda x: -lfI.ll_kIGeom_directed(
                x, *self.args),
            "L-IExp": lambda x: -lfI.ll_LIExp(x, *self.args),
            "k-IExp": lambda x: -lfI.ll_kIExp_directed(
                x, *self.args),
            "CGeom": lambda x: -lfC.ll_CGeom(x, self.adjacency,self.exogenous_variables),
            "CExp": lambda x: -lfC.ll_CExp(x, self.adjacency,self.exogenous_variables),
            "CPareto": lambda x: -lfC.ll_CPareto(x, self.adjacency,self.exogenous_variables),
            "CGamma": lambda x: -lfC.ll_CGamma(x, self.adjacency,self.exogenous_variables),
            "CLognormal": lambda x: -lfC.ll_CLognormal(x, self.adjacency,self.exogenous_variables), 
               
        }
        
        ll_fun_binary = {
            "Logit": lambda x: -lfC.ll_logit(x, self.adjacency,self.selection_variables,self.fixed_selection_params),
            "UBCM": lambda x: -lfC.ll_BCM(
                x, self.adjacency),
            "DBCM": lambda x: -lfC.ll_DBCM(
                x, self.adjacency),
            "POIS": lambda x: -lfE.ll_POIS_binary(x, *self.args),
            "ZIP": lambda x: -lfE.ll_ZIP_binary(
                x, *self.args),
            "NB2": lambda x: -lfE.ll_NB2_binary(x, *self.args),
            "ZINB": lambda x: -lfE.ll_ZINB_binary(
                x, *self.args),
            "L-IGeom": lambda x: -lfI.ll_LIGeom_binary(x, *self.args),
            "k-IGeom": lambda x: -lfI.ll_kIGeom_binary_directed(
                x, *self.args),
            "L-IExp": lambda x: -lfI.ll_LIExp_binary(x, *self.args),
            "k-IExp": lambda x: -lfI.ll_kIExp_binary_directed(
                x, *self.args),
                
        }
        
        jac_fun = {
            "Logit": lambda x: -lfC.jac_logit(x, self.adjacency,self.selection_variables,self.fixed_selection_params),
            "UBCM": lambda x: -lfC.jac_BCM(
                x, self.adjacency),
            "DBCM": lambda x: -lfC.jac_DBCM(
                x, self.adjacency),
            "POIS": lambda x: -lfE.jac_POIS(x, *self.args),
            "ZIP": lambda x: -lfE.jac_ZIP(
                x, *self.args),
            "NB2": lambda x: -lfE.jac_NB2(x, *self.args),
            "ZINB": lambda x: -lfE.jac_ZINB(
                x, *self.args),
            "L-IGeom": lambda x: -lfI.jac_LIGeom(x, *self.args),
            "k-IGeom": lambda x: -lfI.jac_kIGeom_directed(
                x, *self.args),
            "L-IExp": lambda x: -lfI.jac_LIExp(x, *self.args),
            "k-IExp": lambda x: -lfI.jac_kIExp_directed(
                x, *self.args),
            "CGeom": lambda x: -lfC.jac_CGeom(x, self.adjacency,self.exogenous_variables),
            "CExp": lambda x: -lfC.jac_CExp(x, self.adjacency,self.exogenous_variables),
            "CPareto": lambda x: -lfC.jac_CPareto(x, self.adjacency,self.exogenous_variables),
            "CGamma": lambda x: -lfC.jac_CGamma(x, self.adjacency,self.exogenous_variables),
            "CLognormal": lambda x: -lfC.jac_CLognormal(x, self.adjacency,self.exogenous_variables),    
        }
        
        hess_fun = {
            "UBCM": lambda x: -lfC.hess_BCM(
                x, self.adjacency),
            "DBCM": lambda x: -lfC.hess_DBCM(
                x, self.adjacency),
            "L-IGeom": lambda x: -lfI.hess_LIGeom(x, *self.args),
            "L-IExp": lambda x: -lfI.hess_LIExp(x, *self.args),
            "CGeom": lambda x: -lfC.hess_CGeom(x, self.adjacency,self.exogenous_variables),
            "CExp": lambda x: -lfC.hess_CExp(x, self.adjacency,self.exogenous_variables),    
        }
        
        if model in self.conditional_models:
            if model[:3] == "L-C":
                if len(imported_params) ==0:
                    solution = sF.solver(model=self.model,Wij=self.adjacency,selection_variables=self.selection_variables,exogenous_variables=self.exogenous_variables,
                                         fixed_selection_params=self.fixed_selection_params,tol=tol,use_guess= use_guess, 
                                         verbose=verbose, print_steps= print_steps, maxiter=maxiter )
                    self.params = solution
                else:
                    self.params = imported_params  
                n_selection_params = self.selection_variables.shape[1] - len(self.fixed_selection_params)

                ll_topological = ll_fun["Logit"](self.params[:n_selection_params])
                #lfC.ll_logit(self.params[:n_selection_params],self.adjacency,self.selection_variables,self.fixed_selection_params)
                ll_weighted = ll_fun[self.model[2:]](self.params[n_selection_params:])
                
                self.ll = ll_topological + ll_weighted
                self.ll_binary = ll_topological
                jac_topological = jac_fun["Logit"](self.params[:n_selection_params])
                jac_weighted = jac_fun[self.model[2:]](self.params[n_selection_params:])
                # print('norm_top',np.linalg.norm(jac_topological,ord=np.inf),'norm_w',np.linalg.norm(jac_weighted,ord=np.inf))
                
                self.jacobian = np.concatenate((jac_topological,jac_weighted))
                self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
                self.aic = nF.AIC(self.ll,len(self.params))
                self.aic_binary = nF.AIC(self.ll_binary,len(self.params[:n_selection_params]))
                self.bic = nF.BIC(self.ll,len(self.params),len(self.adjacency))
            
            elif model[:3] == "k-C":
                if len(imported_params) == 0:
                    self.input_model = self.model
                    self.input_model = self.input_model + "_directed"
                    
                    solution = sF.solver(model=self.input_model,Wij=self.adjacency,selection_variables=self.selection_variables,exogenous_variables=self.exogenous_variables,
                                         fixed_selection_params=self.fixed_selection_params,tol=tol,use_guess= use_guess, 
                                         verbose=verbose, print_steps= print_steps, maxiter=maxiter )
                    self.params = solution
                else:
                    self.params = imported_params
                n_countries = int(np.sqrt(len(self.adjacency)))
                ll_topological = ll_fun["DBCM"](self.params[:2*n_countries])
                #lfC.ll_logit(self.params[:n_countries],self.adjacency,self.selection_variables,self.fixed_selection_params)
                ll_weighted = ll_fun[self.model[2:]](self.params[2*n_countries:])
                
                self.ll = ll_topological + ll_weighted
                self.ll_binary = ll_topological
                jac_topological = jac_fun["DBCM"](self.params[:2*n_countries])
                jac_weighted = jac_fun[self.model[2:]](self.params[2*n_countries:])
                # print('norm_top',np.linalg.norm(jac_topological,ord=np.inf),'norm_w',np.linalg.norm(jac_weighted,ord=np.inf))
                self.jacobian = np.concatenate((jac_topological,jac_weighted))
                self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
                self.aic = nF.AIC(self.ll,len(self.params))
                self.aic_binary = nF.AIC(self.ll_binary,len(self.params[:2*n_countries]))
                self.bic = nF.BIC(self.ll,len(self.params),len(self.adjacency))
                
        else:
                    
            if len(imported_params) == 0:
                self.input_model = self.model
                if self.input_model in ["k-IExp","k-IGeom"]:
                    self.input_model = self.input_model + "_directed"
                solution = sF.solver(model=self.input_model,Wij=self.adjacency,selection_variables=self.selection_variables,exogenous_variables=self.exogenous_variables,
                                         fixed_selection_params=self.fixed_selection_params,tol=tol,use_guess= use_guess, 
                                         verbose=verbose, print_steps= print_steps, maxiter=maxiter )
                self.params = solution
            else:
                self.params = imported_params
                        
            self.ll = ll_fun[self.model](self.params)
            self.ll_binary = ll_fun_binary[self.model](self.params)
            self.jacobian = jac_fun[self.model](self.params)
            self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
            self.aic = nF.AIC(self.ll,len(self.params))
            self.aic_binary = nF.AIC(self.ll_binary,len(self.params))
            self.bic = nF.BIC(self.ll,len(self.params),len(self.adjacency))

    

    def gen_ensemble(self,n_ensemble = 1000):
        """Generate an ensemble of -n_ensemble- networks according to the selected model
        
        :param n_ensemble: Number of wanted Graph realizations
        :type n_ensemble: int
        
        :return self.w_ensemble_matrix: N_obs X n_ensemble numpy matrix that collects all the ensemble adjacency matrices
        :rtype self.w_ensemble_matrix: np.ndarray
        """
        if self.model in self.continuous_models:
            w_mat_ensemble = eF.faster_ensemble_matrix_directed(params=self.params,Wij=self.adjacency,
                                                        model=self.model,
                                                        exogenous_variables=self.exogenous_variables,
                                                        selection_variables=self.selection_variables,
                                                        fixed_selection_params = self.fixed_selection_params,
                                                        n_ensemble=n_ensemble)
        elif self.model in self.discrete_models:
            w_mat_ensemble = eF.discrete_ensemble_matrix_directed(self.params,self.adjacency,self.model,self.exogenous_variables,
                                                                    self.selection_variables,self.fixed_selection_params,n_ensemble)
        else:
            raise TypeError("model not implemented!")        
        self.w_ensemble_matrix = w_mat_ensemble
        
    
    def classification_measures(self,n_ensemble = 1000, percentiles = (2.5,97.5),stats=[]):
        """Computes Measures for the quality of classification, listed in self.implemented_classifier_statistics.
        To be used after .solve()
        :param n_ensemble: Wanted number of ensemble graphs, default is 1000.
        :type n_ensemble: int
        
        :param percentiles: Explicit the percentiles used for the construction of the confidence interval, default is (2.5,97.5) for a 95 CI.
        :type percentiles: tuple
        :param stats: numpy array or list of classifier statistics in string format.
        :type stats: list of strings
        
        :return self.avg_*: Average value of * classifier statistic in the graph ensemble.
        :rtype self.avg_*: float
        
        :return self.std_*: Standard Deviation of * classifier statistic in the graph ensemble.
        :rtype self.std_*: float
        
        :return self.percentiles_*: percentiles of * classifier statistic in the graph ensemble, default is (2.5,97.5)
        :rtype self.percentiles_*: tuple
        
        :return self.array_*: Whole array of * classifier statistics measured on the graph ensemble.
        :rtype self.array_*: np.ndarray
        
        """
        if len(stats) == 0:
            stats = self.implemented_classifier_statistics
        top_mat = nF.pij_matrix_directed(self.params,self.model,self.adjacency,
        self.selection_variables,self.exogenous_variables,self.fixed_selection_params)
        self.model_binary_adjacency = top_mat

        if "TPR" in stats:
            self.avg_TPR, self.std_TPR, self.percentiles_TPR, self.array_TPR = nF.TPR_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        if "SPC" in stats:
            self.avg_SPC, self.std_SPC, self.percentiles_SPC, self.array_SPC = nF.SPC_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        if "PPV" in stats:
            self.avg_PPV, self.std_PPV, self.percentiles_PPV, self.array_PPV = nF.PPV_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        if "ACC" in stats:
            self.avg_ACC, self.std_ACC, self.percentiles_ACC, self.array_ACC = nF.ACC_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        if "BACC" in stats:
            self.avg_BACC, self.std_BACC, self.percentiles_BACC, self.array_BACC = nF.BACC_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        if "F1_score" in stats:
            self.avg_F1_score, self.std_F1_score, self.percentiles_F1_score, self.array_F1_score = nF.F1_score_ensemble(top_mat,self.binary_adjacency,n_ensemble,percentiles)
        
        
            
    def netstats_measures(self,percentiles = (2.5,97.5),stats=[]):
        """Computes available network statistics, 
        To be used after .solve() and .gen_ensemble(). Available attributes are .avg_*, .std_*, .percentiles_* and .array_* where * stands for the wanted statistics,
        avg_* is namely the ensemble average, std_* is the ensemble standard deviation, percentiles_* is a Tuple made by (percentile[0],percentile[1]) of the ensemble distribution,
        array_* is the whole ensemble distribution.
        
        :param percentiles: Explicit the percentiles used for the construction of the confidence interval, default is (2.5,97.5) for a 95 CI.
        :type percentiles: Tuple
        :param stats: numpy array or list of available network statistics. The wanted stats must be in the list -.implemented_network_statistics-
        :type stats: list
    
        """
        if len(stats) == 0:
            stats = self.implemented_network_statistics
        
        if "degree" in stats:
            self.avg_degree, self.std_degree, self.percentiles_degree, self.array_degree = nF.degree_ensemble(self.w_ensemble_matrix,percentiles)
        if "degree_in" in stats:
            self.avg_degree_in, self.std_degree_in, self.percentiles_degree_in, self.array_degree_in = nF.degree_in_ensemble(self.w_ensemble_matrix,percentiles)
        if "annd" in stats:
            self.avg_annd, self.std_annd, self.percentiles_annd, self.array_annd = nF.annd_ensemble(self.w_ensemble_matrix,percentiles)
        if "clust" in stats:
            self.avg_clust, self.std_clust, self.percentiles_clust, self.array_clust = nF.clust_ensemble(self.w_ensemble_matrix,percentiles)
        if "strength" in stats:
            self.avg_strength, self.std_strength, self.percentiles_strength, self.array_strength = nF.st_ensemble(self.w_ensemble_matrix,percentiles)
        if "strength_in" in stats:
            self.avg_strength, self.std_strength, self.percentiles_strength, self.array_strength = nF.st_in_ensemble(self.w_ensemble_matrix,percentiles)
        if "anns" in stats:
            self.avg_anns, self.std_anns, self.percentiles_anns, self.array_anns = nF.anns_ensemble(self.w_ensemble_matrix,percentiles)
        if "clust_w" in stats:
            self.avg_clust_w, self.std_clust_w, self.percentiles_clust_w, self.array_clust_w = nF.cw_ensemble(self.w_ensemble_matrix,percentiles)
        
    def reproduction_accuracy_s(self,percentiles = (2.5,97.5),stats=[]):
        """Computes RA_s, the percentage of nodes for which the network statistics are compatible with the graph ensemble according to a percentile confidence interval.
        To be used after .solve() and .gen_ensemble(). 
        
        :param percentiles: Explicit the percentiles used for the construction of the confidence interval, default is (2.5,97.5) for a 95 CI.
        :type percentiles: tuple
        :param stats: numpy array or list of network statistics for which it is possible to recover RA_s. The wanted stats must be in the list -.implemented_network_statistics-
        :type stats: list of strings
        
        :return self.RA_s: L-list where L is the number of statistics, the order follows the input stats array
        :rtype self.RA_s: list of float
        """
        if len(stats) == 0:
            stats = List(self.implemented_network_statistics)
        else:
            stats = List(stats)
        
        self.RA_s = nF.ensemble_coverage(self.w_ensemble_matrix,self.adjacency,percentiles,stats)
        
        
    def reproduction_accuracy_w(self,percentiles = (2.5,97.5)):
        """Computes RA_w, the percentage of couples for which the weights are compatible with the graph ensemble according to a percentile confidence interval.
        To be used after .solve() and .gen_ensemble(). 
        
        :param percentiles: Explicit the percentiles used for the construction of the confidence interval, default is (2.5,97.5) for a 95 CI.
        :type percentiles: Tuple
        
        :return self.RA_w: A number (0,1) that explicits RA_w
        :rtype self.RA_w: np.ndarray
        """
        self.RA_w = nF.weighted_coverage(self.w_ensemble_matrix,self.adjacency, percentiles)
        
        
            
                
        
        
