# Lswarm
Particle Swarm Optimization (PSO) + Sensitivity Analysis for parameter rankings in Slope Stability / Landslide analysis.

## The "PSO" folder
This is the main folder that contains the main PSO. 
The codes consists of 5 python files:
  1. deus_ex_machina.py: main routine
  2. sentinels.py: PSO computations routine (positions, velocities, etc)
  3. the_matrix.py: objective function calculation, linked to physics solver
  4. sentinel_config.py: PSO hyperparameters
  5. exos.py” auxiliaries such as printing and monitoring

Other auxiliary functions are in the Aux folder:
  Plot_PSOdb : to plot PSO objective function (and other parameters) per iteration
  export_params : to export parameters at certain iterations of PSO

## The "PSO Result Files" folder
Contains 5 result files. Four of them are used for the manuscript: pso_r_7i.db, pso_Qgauss_2.db, pso_Qcauchy_1.db, pso_QLevyII_1.db

## The "PDP and S2" folder
Contains Jupyter lab notebook for Partial Dependence Plot and S2 Sensitivity analysis (PDP_S2.ipynb). The image files in the folder are the exported files from the running the notebook.

## The "Data Diagnosis" folder
Contains the code for statistical diagnosis of the data in PSO result files (.db files as in "PSO Result Files").
---
# How to Run
The main codes is in the "PSO" folder. The code needs a physics solver (e.g., COMSOL) that can be called externally.
The main code workflow is deus_ex_machina < sentinels (< sentinel_config) < the_matrix
The PSO then produces SQLite database file that contains all particle's properties and objective function values for each iteration.
The PSO saves result each iteration in the database file and can resume iterations by reading the database file. If a database file already has results or partial results saved, it will continue from the last iteration.
