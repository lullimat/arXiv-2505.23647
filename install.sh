# Script for the installation of idea.deploy python virtual environment
# Copyright (C) 2020-2025 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 10/6/2025

echo "Installing the idea.deploy environment for the repository of the paper:"
echo "'Higher-order Tuning of Interface Physics in Multiphase Lattice Boltzmann'"

bash -c "$(curl -fsSL https://raw.githubusercontent.com/lullimat/idea.deploy/refs/heads/master/idpy-bootstrap.sh)"

# Sourcing aliases
ALIAS_SOURCE="$(cat ~/.bashrc | grep idea.deploy | grep source)"
shopt -s expand_aliases
eval "${ALIAS_SOURCE}"

# source ${HOME}/.bashrc
# IDPY_GO="$(type -a idpy-go | cut -d '`' -f 2 | cut -d \' -f 1)"
# IDPY_LOAD="$(type -a idpy-load | cut -d '`' -f 2 | cut -d \' -f 1)"

# echo "IDPY_GO:" "${IDPY_GO}"
# echo "IDPY_LOAD:" "${IDPY_LOAD}"

idpy-go
idpy-load
cd ./papers/
echo ${PWD}

python idpy-papers.py --repo arXiv-2505.23647
cd ./arXiv-2505.23647/
echo
echo "The environemnt idea.deploy is ready and the paper repository has been succefully cloned!"
echo
echo "Please run 'cd ${PWD}' to go to the paper repository"
echo