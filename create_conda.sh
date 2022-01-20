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
    git+https://github.com/tsalo/nilearn.git@e90e3f9f63474b63c56e8e7fcaa1ca604dd06156 \
    nitransforms \
    git+https://github.com/tsalo/rapidtide.git@f86237d8867222d645dd53556cc4617ba4d7ebde \
    git+https://github.com/tsalo/tedana.git@099d508db7ad13164918ad48178638ae935fed52 \
    git+https://github.com/ME-ICA/godec.git@fa95ac88854c79325afc18ed32d16d69a430d391 \
    git+https://github.com/tsalo/peakdet.git@0034b2c76669a227295501e0d7c7d1fd207259d1 \
    git+https://github.com/tsalo/phys2denoise.git@c6d2499b84c786be440bd23e1a727bbbb8e3d97d \
    git+https://github.com/tsalo/ddmra.git@e8e559625fe3ee2b5ca02662092314083e73d9ba

conda list > python_requirements.txt
