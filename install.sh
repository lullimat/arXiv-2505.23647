# Script for the installation of idea.deploy python virtual environment
# Copyright (C) 2020-2025 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 10/6/2025

echo "Installing the idea.deploy environment for the repository of the paper:"
echo "'Higher-order Tuning of Interface Physics in Multiphase Lattice Boltzmann'"

bash -c "$(curl -fsSL https://raw.githubusercontent.com/lullimat/idea.deploy/refs/heads/master/idpy-bootstrap.sh)"

# source ${HOME}/.bashrc
# idpy-go
# idpy-load

# cd ./papers/
# python idpy-papers.py arXiv-2505.23647
# cd ./arXiv-2505.23647/
