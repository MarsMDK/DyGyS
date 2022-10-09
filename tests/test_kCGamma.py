import pandas as pd
import numpy as np
from numba import jit
from numba.typed import List
import time
import sys
import unittest

sys.path.append("../src/")
import DyGyS as dyg


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

def extrapolate_df_baci(df_gravity):
        nodes_o = np.unique(df_gravity["iso3_o"].values)
        gdp_o = df_gravity["gdp_ppp_pwt_o"].values
        gdp_d = df_gravity["gdp_ppp_pwt_d"].values
        distcap = df_gravity["distcap"].values
        tradeflow_baci = df_gravity["tradeflow_baci"].fillna(0).values
        log_gdp_o = np.log(gdp_o/gdp_o.mean())
        log_gdp_d = np.log(gdp_d/gdp_d.mean())
        
        log_gdp = log_gdp_o + log_gdp_d
        log_distcap = np.log(distcap)
        
        exog_matrix = np.ones((len(log_gdp),3))
        exog_matrix[:,1] = log_gdp
        exog_matrix[:,2] = log_distcap
        return nodes_o, tradeflow_baci, exog_matrix

def extrapolate_df_baci_directed(df_gravity):
    nodes_o = np.unique(df_gravity["iso3_o"].values)
    gdp_o = df_gravity["gdp_ppp_pwt_o"].values
    gdp_d = df_gravity["gdp_ppp_pwt_d"].values
    distcap = df_gravity["distcap"].values
    tradeflow_baci = df_gravity["tradeflow_baci"].fillna(0).values
    log_gdp_o = np.log(gdp_o/gdp_o.mean())
    log_gdp_d = np.log(gdp_d/gdp_d.mean())
    
    log_distcap = np.log(distcap)
    
    exog_matrix = np.ones((len(log_gdp_o),4))
    exog_matrix[:,1] = log_gdp_o
    exog_matrix[:,2] = log_gdp_d
    exog_matrix[:,3] = log_distcap
    return nodes_o, tradeflow_baci, exog_matrix


def example_data_and_settings_undirected():
    df_gravity_2007 = pd.read_csv("Baci_Gravity_2007.csv")
    nodes_2007, tradeflow_2007, exogenous_variables_2007 = extrapolate_df_baci(df_gravity_2007)
    tradeflow_2007 /= 1000
    tradeflow_2007 = symmetrize(tradeflow_2007)
    return tradeflow_2007,exogenous_variables_2007

def example_data_and_settings_directed():
    df_gravity_2007 = pd.read_csv("Baci_Gravity_2007.csv")
    nodes_2007, tradeflow_2007, exogenous_variables_2007 = extrapolate_df_baci_directed(df_gravity_2007)
    tradeflow_2007 /= 1000
    return tradeflow_2007,exogenous_variables_2007

class test_kCGamma(unittest.TestCase):
    

    def test_undirected(self):
        trade, exog= example_data_and_settings_undirected()
        tic = time.time()
        G = dyg.UndirectedGraph(trade)
        G.solver(model="k-CGamma",exogenous_variables=exog,selection_variables=exog)
        toc_solve = time.time() - tic
        print('norm',G.norm)
        print('time to solve',toc_solve,'seconds')
        tic = time.time()
        G.gen_ensemble()
        toc_gen_ensemble = time.time()- tic
        print('time to gen ensemble',toc_gen_ensemble,'seconds')
        self.assertTrue(G.w_ensemble_matrix.shape[0] == len(trade), "test for correct first dimension of ensemble graph collection")
        self.assertTrue(G.w_ensemble_matrix.shape[1] == 1000, "test for correct second dimension of ensemble graph collection")
        self.assertTrue(np.all(G.w_ensemble_matrix >= 0.), "all matrices must be non-negative")
        tic = time.time()
        G.classification_measures()
        toc_cf_measures = time.time()- tic
        print('time to compute cf measures',toc_cf_measures,'seconds')
        self.assertTrue(isinstance(G.avg_TPR,float) , "avg TPR does not have the right type.") 
        self.assertTrue(isinstance(G.std_TPR,float) , "std TPR does not have the right type.")
        self.assertTrue(isinstance(G.percentiles_TPR,tuple), "percentiles TPR does not have the right type.")
        self.assertTrue(isinstance(G.array_TPR,np.ndarray), "array TPR does not have the right type.")
        tic = time.time()
        
        G.netstats_measures()
        toc_netstats = time.time()- tic
        print('time to compute net measures',toc_netstats,'seconds')
        self.assertTrue( isinstance(G.avg_degree,np.ndarray) and isinstance(G.std_degree,np.ndarray) and isinstance(G.percentiles_degree,np.ndarray) and (G.array_degree,np.ndarray), "avg, std, percentiles and array of degree-centrality do not have the right types.") 
        tic = time.time()
        G.reproduction_accuracy_s()
        toc_ra_s = time.time()- tic
        print('time to compute ra_s',toc_ra_s,'seconds')
        self.assertTrue( isinstance(G.RA_s,np.ndarray) ,"RA_s has not the correct type")
        tic = time.time()
        G.reproduction_accuracy_w()
        toc_ra_w = time.time()- tic
        print('time to compute ra_w',toc_ra_w,'seconds')
        self.assertTrue( isinstance(G.RA_w,np.ndarray), "RA_w has not the correct type" )

    def test_directed(self):
        trade, exog= example_data_and_settings_directed()
        tic = time.time()
        G = dyg.DirectedGraph(trade)
        G.solver(model="k-CGamma",exogenous_variables=exog,selection_variables=exog)
        toc_solve = time.time() - tic
        print('norm',G.norm)
        print('time to solve',toc_solve,'seconds')
        tic = time.time()
        G.gen_ensemble()
        toc_gen_ensemble = time.time()- tic
        print('time to gen ensemble',toc_gen_ensemble,'seconds')
        self.assertTrue(G.w_ensemble_matrix.shape[0] == len(trade), "test for correct first dimension of ensemble graph collection")
        self.assertTrue(G.w_ensemble_matrix.shape[1] == 1000, "test for correct second dimension of ensemble graph collection")
        self.assertTrue(np.all(G.w_ensemble_matrix >= 0.), "all matrices must be non-negative")
        tic = time.time()
        G.classification_measures()
        toc_cf_measures = time.time()- tic
        print('time to compute cf measures',toc_cf_measures,'seconds')
        self.assertTrue(isinstance(G.avg_TPR,float) , "avg TPR does not have the right type.") 
        self.assertTrue(isinstance(G.std_TPR,float) , "std TPR does not have the right type.")
        self.assertTrue(isinstance(G.percentiles_TPR,tuple), "percentiles TPR does not have the right type.")
        self.assertTrue(isinstance(G.array_TPR,np.ndarray), "array TPR does not have the right type.")
        tic = time.time()
        
        G.netstats_measures()
        toc_netstats = time.time()- tic
        print('time to compute net measures',toc_netstats,'seconds')
        self.assertTrue( isinstance(G.avg_degree,np.ndarray) and isinstance(G.std_degree,np.ndarray) and isinstance(G.percentiles_degree,np.ndarray) and (G.array_degree,np.ndarray), "avg, std, percentiles and array of degree-centrality do not have the right types.") 
        tic = time.time()
        G.reproduction_accuracy_s()
        toc_ra_s = time.time()- tic
        print('time to compute ra_s',toc_ra_s,'seconds')
        self.assertTrue( isinstance(G.RA_s,np.ndarray) ,"RA_s has not the correct type")
        tic = time.time()
        G.reproduction_accuracy_w()
        toc_ra_w = time.time()- tic
        print('time to compute ra_w',toc_ra_w,'seconds')
        self.assertTrue( isinstance(G.RA_w,np.ndarray), "RA_w has not the correct type" )

    
if __name__ == '__main__':
    unittest.main()

