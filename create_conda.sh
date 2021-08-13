#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p conda_env/ pip python=3.7
source activate conda_env/
conda install \
    pip \
    accelerate \
    Biopython \
    cython \
    dateutil \
    funcsigs \
    libgfortran \
    matplotlib \
    numexpr \
    numpy \
    pandas \
    ply \
    seaborn \
    scikit-learn \
    scipy \
    sympy \
    tornado

/home/data/nbc/misc-projects/Salo_PowerReplication/code/conda_env/bin/pip install pip -U

/home/data/nbc/misc-projects/Salo_PowerReplication/code/conda_env/bin/pip install \
    ipython \
    ipywidgets \
    jupyterlab \
    nibabel \
    nilearn \
    rapidtide==2.0.5 \
    tedana==0.0.11 \
    git+https://github.com/ME-ICA/godec.git@094129688775e0c07ad81e946974fa0ef72f34c4 \
    git+https://github.com/physiopy/peakdet.git@f6908e3cebf2fdc31ba73f2b3d3370bf7dfae89c \
    git+https://github.com/tsalo/phys2denoise.git@be8251db24b157c9a7717f3e2e41eca60ed23649

conda list > python_requirements.txt
