# Multi-output Surrogate Assisted Evolutionary Algorithm for Expensive Multi-Modal Optimization Problems 

 [Renzhi Chen]()\* [Ke Li]()\*,
[[Paper]]() [[Supplementary]]()

## Overview

This repository contains Python implementation of the algorithm Multi-output Surrogate Assisted Evolutionary Algorithm (MSAEA) for Expensive Multi-Modal Optimization problems.



## Code Structure

algorithms/ --- algorithms definitions

benchmarks/ --- multi-modal problem definitions

models/ -- surrogate models

multimodels/ -- multioutput surrogate models

utils/ -- supporting files



run.sh -- run the experiment in batch

experiment.py --- main execution file

stats.py -- result analysis file

visualization.py -- result visualization



## Requirements

- Python version: tested in Python 3.7.7
- Operating system: tested in Ubuntu 20.04



## Getting Started

### Basic usage

Run the main file with python with specified arguments:

```bash
python3.7 experiment.py --problems Ackley Rastrigin Griewank  --n-vars 3 5 8 10 --algorithms MSAEA BO cBO MAMPSO DREM DRNESO --seeds 1 2 3 4 5 6 7 8 9 10 11
```

### Parallel experiment

Run the script file with bash, for example:

```bash
./scripts/run.sh
```

The following variables should be set according to your experiment. 

```bash
n_process=8 
PY=python3.7
algos="MSAEA BO cBO MAMPSO DRNESO DREM"
seeds="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"
vars="3 5 8 10"
problems="Rastrigin Ackley Griewank"
```



### Statistic 

Run the main file with python with specified arguments:

```bash
python3.7 stats.py --problems Ackley Rastrigin Griewank  --n-vars 3 5 8 10 --algorithms MSAEA BO cBO MAMPSO DREM DRNESO --seeds 1 2 3 4 5 6 7 8 9 10 11
```

This script will generate A12 and Scott-knott in folder ./output/data/. This script also will generate the mean(std) value, accompanied with Wilcoxon signed-rank test.



## Result

The optimization results are saved in txt format. They are stored under the folder:

```
output/data/{problem}/x{n}/{algo}/{seed}/
```



## Citation

If you find our repository helpful to your research, please cite our paper:

```latex
@article{
}
```



