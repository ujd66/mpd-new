#!/bin/bash

git submodule update --init --recursive --progress

eval "$(~/miniconda3/bin/conda shell.bash hook)"

CONDA_ENV_NAME="mpd-new"

# check if a conda environment with the same name already exists, if yes remove it
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
  echo "Removing existing conda environment: ${CONDA_ENV_NAME}"
  conda env remove -n "${CONDA_ENV_NAME}" --yes
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPS_DIR="${THIS_DIR}/deps"

conda update -n base conda

conda env create -f environment.yml

conda activate ${CONDA_ENV_NAME}

echo "-------> Installing pybullet_ompl"
cd "${DEPS_DIR}"/pybullet_ompl || exit 1
git clone https://github.com/ompl/ompl.git
cd ompl || exit 1
git checkout fca10b4bd4840856c7a9f50d1ee2688ba77e25aa
mkdir -p build/Release
cd build/Release || exit 1
cmake -DCMAKE_DISABLE_FIND_PACKAGE_pypy=ON ../.. -DPYTHON_EXEC=${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python
make -j 32 update_bindings  # This step takes a lot of time.
for _ in {1..5}; do make -j 32; done  # run multiple times to avoid errors
cd ${DEPS_DIR}/pybullet_ompl && pip install -e .

# ncurses is causing an error using the linux command watch, htop, ...
conda remove --force ncurses --yes

# Git
pre-commit install

cd "${THIS_DIR}" || exit 1
