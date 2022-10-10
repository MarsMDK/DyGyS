
DyGyS: DYadic GravitY regression models with Soft constraints
=====================================

DyGyS is a package developed on python3 for Maximum Entropy regression models with gravity specification for undirected and directed network data.

DyGyS provides a numerous amount of models, described in their undirected declination in articles [1](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033105#) and [2] and consisting of both econometric and statistical physics-inspired models.
The use of soft constraints enable the user to explicitly constrain network properties such as the number of links, the degree sequence (degree centrality for Undirected Networks and out-degree/in-degree centralities for directed networks), and the total weight (for a small number of viable models).

Furthermore it is not only possible to solve the model and extract the parameters, but also to generate the ensemble, compute a number of network statistics, compute model selection measures such as AIC and BIC, and quantify the reproduction accuracy of:
- Topology using measures as True Positive Rate, Specificity, Precision, Accuracy, Balanced Accuracy and F1score;
- Weights, measuring the fraction of weights inside the percentile CI extracted from the ensemble of graphs;
- Network Statistics, measuring the fraction of nodes for which the network statistics are inside the wanted percentile CI extracted from the ensemble of graphs.

To explore Maximum-Entropy modeling on networks, checkout [Maximum Entropy Hub](https://meh.imtlucca.it/).

When using the module for your scientific research please consider citing:
::
    @article{PhysRevResearch.4.033105,
      title = {Gravity models of networks: Integrating maximum-entropy and econometric approaches},
      author = {Di Vece, Marzio and Garlaschelli, Diego and Squartini, Tiziano},
      journal = {Phys. Rev. Research},
      volume = {4},
      issue = {3},
      pages = {033105},
      numpages = {19},
      year = {2022},
      month = {Aug},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.4.033105},
      url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.033105}
    }


and
::
    @article{PhysRevResearch.4.033105,
      title = {Gravity models of networks: Integrating maximum-entropy and econometric approaches},
      author = {Di Vece, Marzio and Garlaschelli, Diego and Squartini, Tiziano},
      journal = {Phys. Rev. Research},
      volume = {4},
      issue = {3},
      pages = {033105},
      numpages = {19},
      year = {2022},
      month = {Aug},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.4.033105},
      url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.033105}
    }


Currently Available Models
--------------------------

DyGyS contains models for network data with both continuous and discrete-valued semi-definite positive weights.
The available models for discrete count data are described in [1](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033105#) and consist of:
- **POIS** *Poisson Model*__ 
- **ZIP** *Zero-Inflated Poisson Model* __
- **NB2** *Negative Binomial Model* __
- **ZINB** *Zero-Inflated Negative Binomial Model* __
- **L-CGeom** *L-constrained Conditional Geometric Model*, noted as TSF in the paper. __
- **k-CGeom** *k-constrained Conditional Geometric Model*, noted as TS in the paper. __
- **L-IGeom** *L-constrained Integrated Geometric Model*, noted as H(1) in the paper. __
- **k-IGeom** *k-constrained Integrated Geometric Model*, noted as H(2) in the paper. __

The analogue models for continuous-valued data are described in [2](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033105#) and consist of:
- **L-CExp** *L-constrained Conditional Exponential Model*, the L-constrained variant of C-Exp in the paper.
- **k-CExp** *k-constrained Conditional Exponential Model*, noted as CExp in the paper.
- **L-IExp** *L-constrained Integrated Exponential Model*, the L-constrained variant of I-Exp in the paper.
- **k-IExp** *k-constrained Integrated Exponential Model*, noted as IExp in the paper.
- **L-CGamma** *L-constrained Conditional Gamma Model*, the L-constrained variant of C-Gamma in the paper.
- **k-CGamma** *k-constrained Conditional Gamma Model*, noted as CGamma in the paper.
- **L-CPareto** *L-constrained Conditional Pareto Model*, the L-constrained variant of C-Pareto in the paper.
- **k-CPareto** *k-constrained Conditional Pareto Model*, noted as CPareto in the paper.
- **L-CLognormal** *L-constrained Conditional Lognormal Model*, the L-constrained variant of C-Lognormal in the paper.
- **k-CLognormal** *k-constrained Conditional Lognormal Model*, noted as CLognormal in the paper.

Please refer to the papers for further details.

Installation
------------
DyGyS can be installed via pip. You can do it from your terminal
::
    $ pip install DyGyS

If you already installed the package and want to  upgrade it,
you can type from your terminal:

::
        $ pip install DyGyS --upgrade

Dependencies
---------------------
DyGyS uses the following dependencies:
- **scipy** for optimization and root solving;
- **numba** for fast computation of network statistics and criterion functions.
- **numba-scipy** for fast computation of special functions such as gammaincinv and erfinv.

They can be easily installed via pip typing
::
    $ pip install scipy
    $ pip install numba
    $ pip install numba-scipy


How-to Guidelines
------------
The module containes two classes, namely UndirectedGraph and DirectedGraph.
An Undirected Graph is defined as a network where weights are reciprocal, i.e., $$w_{ij} = w_{ji}$$ where $$w_{ij}$$ is the network weight from node $$i$$ to node $$j$$. 
If weights are not reciprocal, please use the DirectedGraph class.

Class Instance and Empirical Network Statistics
-----------

To inizialize an UndirectedGraph or DirectedGraph instance you can type:

::
    G = UndirectedGraph(adjacency=Wij)
    or
    G = DirectedGraph(adjacency=Wij)
where Wij is the weighted adjacency matrix in 1-D (dense) or 2-D numpy array format.

After initializing you can already explore core network statistics such as (out-)degree, in-degree, average neighbor degree, binary clustering coefficient, (out-)strength, in-strength, average neighbor strength and weighted clustering coefficient.
These are available using the respective codewords:

    G.degree, G.degree_in, G.annd, G.clust, G.strength, G.strength_in, G.anns, G.clust_w

Solving the models
------------
You can explore the currently available models using
::    
    G.implemented_models
use their names as described in this list not to incur in error messages.

In order to solve the models you need to define a *regressor matrix* $X_w$ of dimension $N_{obs} \times k$ where $N_{obs} = N^2$ is the number of observations (equivalent to the square of the number of nodes), and $k$ is the number of exogenous variables introduced in the Gravity Specification. 
For L-Constrained Conditional Models and Zero-Inflated models you ought to define also a regressor matrix $X_t$ for the first-stage (or topological) optimization and you can choose to fix some of the first-stage parameters.

When ready you can choose one of the aforementioned models and solve for their parameters using
::    
    G.solve(model= <chosen model>,exogenous_variables = X_w, selection_variables = X_t,
        fixed_selection_parameters = <chosen fixed selection parameters>)

Once you solved the model various other attributes become visible and measures dependent solely on criterion functions are computed. These include Loglikelihood, Jacobian, Infinite Jacobian Norm, AIC, Binary AIC and BIC, available using the codewords:
::
    G.ll, G.jacobian, G.norm, G.aic, G.aic_binary, G.bic

For further details on the .solve functions please see the documentation.



Generating the network ensemble 
----------------
Generating the network ensemble is very easy. It's enough to type:
::    
    G.gen_ensemble(n_ensemble=<wanted number of graphs>)
The graphs are produced using the "default_rng" method for discrete-valued models or using Inverse Transform Sampling for continuous-valued models.

This method returns
::
    G.w_ensemble_matrix
which is a $N_{obs} \times N_{ensemble}$ matrix which includes all of the $N_{ensemble}$ adjacency matrices in the ensemble.
Such method behaves well for networks up to $ N=200 $ for $10^{4}$ ensemble graphs, no test has been done for large networks where G.w_ensemble_matrix could be limited by RAM.


Computing relevant measures
----------------
Let's start by showing how to compute topology-related measures. 
You can type:
::    
    G.classification_measures(n_ensemble=<wanted number of graphs>,percentiles = (inf_p, sup_p), stats =[<list of wanted statistics>])
This method does not need G.w_ensemble_matrix so you can use it without generating the ensemble of weighted networks.
The statistics you can compute are listed in G.implemented_classifier_statistics and once you define the number of networks, the ensemble percentiles and statistics of interest, it returns
::
    G.avg_*, G.std_*, G.percentiles_*, G.array_*
where "avg" stands for ensemble average, "std" for ensemble standard deviation, "array" stands for the entire measures on each ensemble graph, "percentiles" is a tuple containing the inf_p-percentile (default 2.5) and sup_p-percentile (default 97.5) in the ensemble and * is the statistic of interest, written as in G.implemented_classifier_statistics.


To compute network statistics you can type:
::
    G.netstats_measures(percentiles=(inf_p, sup_p), stats = [<list of wanted statistics>])
This method needs the previous computation of G.w_ensemble_matrix.
It computes average, standard deviation, percentiles and ensemble arrays of the network statistics of interest which can be seen in G.implemented_network_statistics.
It returns:
::
    G.avg_*, G.std_*, G.percentiles_*, G.array_*

To compute the reproduction accuracy for the network statistics (introduced in [2]) you can type:
    
    G.reproduction_accuracy_s(percentiles=(inf_p,sup_p),stats=[])
This method needs the previous computation of G.w_ensemble_matrix.
It computes the fraction of nodes for which the network measure is inside a percentile CI extracted from the graph ensemble.
It returns
::    
    G.RA_s
i.e., a list of reproduction accuracies for each of the network statistics introduced via -stats- list arranged according to its order.

Finally, you can compute the reproduction accuracy for the weights (introduced in [2]) using:
::
    G.reproduction_accuracy_w(percentiles=(inf_p,sup_p))
This method needs the previous computation of G.w_ensemble_matrix.
It computes the fraction of empirical weights which fall inside the percentile CI interval given by the inf_p-percentile  and sup_p-percentile, extracted from the graph ensemble and it returns as the attribute 
::
    G.RA_w.


Credits
-----

*Author*:

[Marzio Di Vece](https://www.imtlucca.it/it/marzio.divece) (a.k.a. [MarsMDK](https://github.com/MarsMDK))

*Acknowledgments*:
The module was developed under the supervision of [Diego Garlaschelli](https://www.imtlucca.it/en/diego.garlaschelli) and [Tiziano Squartini](https://www.imtlucca.it/en/tiziano.squartini).
It was developed at [IMT School for Advanced Studies Lucca](https://www.imtlucca.it/en) and financed by the IMT research project PAI PROCOPE - "Prosociality, Cognition and Peer Effects".

