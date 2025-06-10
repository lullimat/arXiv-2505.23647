# Higher-order Tuning of Interface Physics in Multiphase Lattice Boltzmann

**This repository matches the arXiv version of the paper** [https://arxiv.org/abs/2505.23647](https://arxiv.org/abs/2505.23647)

**The repository is being updated**

**Please, keep on pulling the updates, either with your git manager software of choice or by means of the command**
```bash
$ git pull
```

This project will be updated in order be kept functional for further updates to [**idea.deploy**](https://github.com/lullimat/idea.deploy).

Welcome to the repository related to the paper published on arXiv [https://arxiv.org/abs/2505.23647](https://arxiv.org/abs/2505.23647). In order to use this repository you should have cloned it from the parent project [idea.deploy](https://github.com/lullimat/idea.deploy) in the directory ./papers by means of the local python script idpy-papers.py and switch to 'devel' branch by means of `git checkout devel` command.

In order to open the present Jupyter notebook "LatticeMomentumBalance.ipynb" you should perform the following steps
- install the [idea.deploy](https://github.com/lullimat/idea.deploy) project, following the instructions in the README.md in the section "Installation"
- load the idea.deploy python virtual environment: if you installed the bash aliases for the [idea.deploy](https://github.com/lullimat/idea.deploy) project you can issue the command "idpy-load"
- launch locally the Jupyter server with "idpy-jupyter" or "idpy-jupyter-lab"
- copy and paste in your browser the url prompted in the terminal in order to open the Jupyter server interface
- click the file "MultiPhaseTolmanTuning.ipynb"
- wait for all the extension to be loaded: when the notebook is loaded you should see a "Table of Contents" on the left side and the different code cells in the "folded" mode with a small grey arrow on the left

In order to execute the code content of a cell, select it and enter the key combination "shift + enter".

As of today, after much testing Firefox, Google Chrome and VSCode have offered the most reliable experience in handling the notebook.

## Installation
```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/lullimat/arXiv-2505.23647/refs/heads/main/install.sh)"
```

## Dependencies

A part from the python dependencies a working "latex" environment needs to be installed on the system in order to reproduce the plots which contain latex symbols.

## Disk Requirement
The output of the fully execute notebook has a memory footprint of -- MB.

Monday June 10, 2025