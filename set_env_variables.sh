export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mpd-splines-public/lib
export CPATH=$HOME/miniconda3/envs/mpd-splines-public/include

# OMPL Python bindings path
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMPL_BUILD_DIR="${THIS_DIR}/deps/pybullet_ompl/ompl/build/Release"

# Add OMPL Python bindings to PYTHONPATH
export PYTHONPATH="${OMPL_BUILD_DIR}/lib:${THIS_DIR}/deps/pybullet_ompl/ompl/py-bindings:${PYTHONPATH}"

# Add OMPL shared library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${OMPL_BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
