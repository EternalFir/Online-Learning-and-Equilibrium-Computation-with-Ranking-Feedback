# Online-Learning-and-Equilibrium-Computation-with-Ranking-Feedback

## Installation

```bash 
git clone git@github.com:EternalFir/Online-Learning-and-Equilibrium-Computation-with-Ranking-Feedback.git
cd ./Online-Learning-and-Equilibrium-Computation-with-Ranking-Feedback

conda create -n OLECRF python=3.8
conda activate OLECRF
pip install -r requirements.txt
```

## Run Experiments

#### Compile all C++ files

```bash
make
```

#### Generate work files and random utility variations

```bash
python ./distribute.py 0 0
```

#### Run all experiments

```bash
python ./distribute.py 0 1
```

#### Search best parameters and plot

```bash
python ./draw_search.py
```

Experiment results will be stored in ./results and figures generated will be stored in ./figures

