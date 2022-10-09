import numpy as np
from numba import jit


def check_df(df_gravity):
    nodes_o = np.unique(df_gravity["iso3_o"].values)
    nodes_d = np.unique(df_gravity["iso3_d"].values)
    gdp_o = np.unique(df_gravity["gdp_ppp_pwt_o"].values)
    gdp_d = np.unique(df_gravity["gdp_ppp_pwt_d"].values)
    
    
    cond1 = np.all(nodes_o == nodes_d)
    cond2 = np.all(gdp_o == gdp_d)
     
    flag = False
    if cond1 and cond2:
        flag = True
    return flag

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

def extrapolate_df_baci_plot(df_gravity):
    nodes_o = np.unique(df_gravity["iso3_o"].values)
    gdp_o = df_gravity["gdp_ppp_pwt_o"].values
    gdp_d = df_gravity["gdp_ppp_pwt_d"].values
    distcap = df_gravity["distcap"].values
    tradeflow_baci = df_gravity["tradeflow_baci"].fillna(0).values
    log_trade = []
    log_var = []
    for i in range(len(nodes_o)):
        for j in range(len(nodes_o)):
            ij = i*len(nodes_o)+j 
            if tradeflow_baci[ij] != 0 and distcap[ij]!=0:
                log_trade.append(np.log(tradeflow_baci[ij]))
                log_var.append(np.log(gdp_o[ij]*gdp_d[ij]/distcap[ij]))
    
    return nodes_o, log_trade, log_var


def extrapolate_df_gled_plot(df_gravity):
    nodes_o = np.unique(df_gravity["acra"].values)
    gdp_o = df_gravity["gdpa"].values
    gdp_d = df_gravity["gdpb"].values
    distcap = df_gravity["kmdist"].values
    tradeflow_baci = df_gravity["trade_udd"].values.astype(np.float64)
    
    log_trade = []
    log_var = []
    for i in range(len(nodes_o)):
        for j in range(len(nodes_o)):
            ij = i*len(nodes_o)+j 
            if tradeflow_baci[ij] != 0 and distcap[ij]!=0:
                log_trade.append(np.log(tradeflow_baci[ij]))
                log_var.append(np.log(gdp_o[ij]*gdp_d[ij]/distcap[ij]))
    print(len(log_trade),len(log_var))
    
    return nodes_o, log_trade, log_var

def extrapolate_df_gled(df_gravity):
    nodes_o = np.unique(df_gravity["acra"].values)
    gdp_o = df_gravity["gdpa"].values
    gdp_d = df_gravity["gdpb"].values
    distcap = df_gravity["kmdist"].values
    tradeflow_baci = df_gravity["trade_udd"].values.astype(np.float64)
    log_gdp_o = np.log(gdp_o/gdp_o.mean())
    log_gdp_d = np.log(gdp_d/gdp_d.mean())
    
    log_gdp = log_gdp_o + log_gdp_d
    log_distcap = np.zeros(len(distcap))
    for i in range(len(log_distcap)):
        if distcap[i] != 0:
            log_distcap[i] = np.log(distcap[i])
    
    
    exog_matrix = np.ones((len(log_gdp),3))
    exog_matrix[:,1] = log_gdp
    exog_matrix[:,2] = log_distcap
    return nodes_o, tradeflow_baci, exog_matrix

def extrapolate_df_gled_mod(df_gravity):
    nodes_o = np.unique(df_gravity["acra"].values)
    gdp_o = df_gravity["gdpa"].values/df_gravity["gdpa"].values.mean()
    gdp_d = df_gravity["gdpb"].values/df_gravity["gdpb"].values.mean()
    distcap = df_gravity["kmdist"].values
    tradeflow_baci = df_gravity["trade_udd"].values.astype(np.float64)
    gdp_product = np.zeros(len(gdp_o))
    for i in range(len(distcap)):
        gdp_product[i] = gdp_o[i]*gdp_d[i]
            
    exog_matrix = np.ones((len(gdp_product),3))
    exog_matrix[:,1] = gdp_product
    exog_matrix[:,2] = distcap
    return nodes_o, tradeflow_baci, exog_matrix



@jit(nopython=True)
def symmetrize(input):
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
def binarize(input):
    dim = len(input)
    byn = np.zeros(dim)
    for i in range(dim):
        if input[i]> 0:
            byn[i] = 1
    return byn 
        

def obscured_adjacency_symmetric_positives(adjacency,obscuration_percentage = 0):
    n_countries = int(np.sqrt(len(adjacency)))
    indexes_w_positives = []
    count_w_positives = 0 
    for i in range(n_countries):
        for j in range(i+1,n_countries):
            ij = int(i*n_countries+j)
            if adjacency[ij] > 0:
                indexes_w_positives.append(ij)
                count_w_positives += 1

    n_obscured = int(count_w_positives*obscuration_percentage)

    chosen_indexes = np.random.choice(indexes_w_positives,n_obscured)
    adjacency_obscured = adjacency.copy()
    for i in range(n_countries):
        for j in range(i+1,n_countries):
            ij = int(i*n_countries+j)
            ji = int(j*n_countries+i)
            if ij in chosen_indexes:
                adjacency_obscured[ij] = 0
                adjacency_obscured[ji] = 0

    return adjacency_obscured, chosen_indexes


def obscured_adjacency_symmetric(adjacency,obscuration_percentage = 0):
    n_countries = int(np.sqrt(len(adjacency)))
    indexes_w = []
    count_w = 0 
    for i in range(n_countries):
        for j in range(i+1,n_countries):
            ij = int(i*n_countries+j)
            indexes_w.append(ij)
            count_w +=1
            
    n_obscured = int(count_w*obscuration_percentage)

    chosen_indexes = np.random.choice(indexes_w,n_obscured)
    adjacency_obscured = adjacency.copy()
    for i in range(n_countries):
        for j in range(i+1,n_countries):
            ij = int(i*n_countries+j)
            ji = int(j*n_countries+i)
            if ij in chosen_indexes:
                rand_1 = np.random.random()
                rand_2 = np.random.random()
                if rand_1 < rand_2:
                    adjacency_obscured[ij] = 0
                    adjacency_obscured[ji] = 0
                else:
                    adjacency_obscured[ij] = 1
                    adjacency_obscured[ji] = 1


    return adjacency_obscured, chosen_indexes

