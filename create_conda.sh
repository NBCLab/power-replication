#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p conda_env/ pip python=3.7
source activate conda_env/
conda install \
    pip \
    Biopython \
    cython \
    funcsigs \
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
    accelerate \
    duecredit \
    ipython \
    ipywidgets \
    jupyterlab \
    nibabel \
    nilearn \
    git+https://github.com/tsalo/rapidtide.git@f86237d8867222d645dd53556cc4617ba4d7ebde \
    git+https://github.com/ME-ICA/tedana.git@3147e01bfbd2a058f41af73eb55debb3dea519b2 \
    git+https://github.com/ME-ICA/godec.git@094129688775e0c07ad81e946974fa0ef72f34c4 \
    git+https://github.com/tsalo/peakdet.git@0034b2c76669a227295501e0d7c7d1fd207259d1 \
    git+https://github.com/tsalo/phys2denoise.git@be8251db24b157c9a7717f3e2e41eca60ed23649

conda list > python_requirements.txt
