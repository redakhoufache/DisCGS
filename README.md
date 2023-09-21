# DisCGS

This code performs the distributed collapsed Gibbs sampler for Dirichlet Process Mixture Model inference.

### Building

The script build.sh is provided to build an executable jar containing all the dependencies. 
Use the following command to build it: 
```
/bin/bash build.sh
```
See src/pom.xml file for Scala and spark dependencies.

### Running 

In order to run the built jar use the following code:

```
scala -J-Xmx1024m target/DisDPMM_2.13-1.0-jar-with-dependencies.jar <dataset name> <number of workers> <number of runs>
```

Example of execution:

```
scala -J-Xmx1024m target/DisDPMM_2.13-1.0-jar-with-dependencies.jar EngyTime_4096_2_2 2 10
```
The above code will perform  10 runs on EngyTime dataset using 2 workers.

The datasets used in the paper are provided in dataset file.

The results are saved in results file.

### Analyse results

A jupyter notebook is provided to analyse the results.

Please go to results file and open the notebook AnalyseResults.ipynb and execute the cells to see the results.
