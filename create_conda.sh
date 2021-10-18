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
    ddmra \
    duecredit \
    ipython \
    ipywidgets \
    jupyterlab \
    nibabel \
    nilearn \
    nitransforms \
    git+https://github.com/tsalo/rapidtide.git@f86237d8867222d645dd53556cc4617ba4d7ebde \
    git+https://github.com/tsalo/tedana.git@099d508db7ad13164918ad48178638ae935fed52 \
    git+https://github.com/ME-ICA/godec.git@094129688775e0c07ad81e946974fa0ef72f34c4 \
    git+https://github.com/tsalo/peakdet.git@0034b2c76669a227295501e0d7c7d1fd207259d1 \
    git+https://github.com/tsalo/phys2denoise.git@10f9fb6d550dbbe5b9cc80c0cdaee346440e66fa

conda list > python_requirements.txt
