leap_rigs
---------

This repository contains code for running experiments on the LEAP rigs, including open and closed-loop optogenetic stimulation and realtime SLEAP inference.


# Installation

1. Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repo and create the environment:

```
git clone https://github.com/murthylab/leap_rigs
cd leap_rigs
conda env create -f environment.yml
```

This installs all dependencies including Motif, CUDA and SLEAP into an environment named `leap_rigs`.

To install to a different environment name:

```
conda env create -f environment.yml -n my_env
```

To update, just `git pull` to grab new changes.

**Note:** This will *not* install the scripts as a package, so you must be in this directory to run the experiment scripts.


# Usage
1. Activate the environment: `conda activate leap_rigs`
2. Run the experiment script you want to use: `python pilot_expt.py`


See [`pilot_expt.py`](https://github.com/murthylab/leap_rigs/blob/main/pilot_expt.py) for a test experiment setup.

See [`sandbox.py`](https://github.com/murthylab/leap_rigs/blob/main/sandbox.py) for ad-hoc usage.