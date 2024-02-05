FROM condaforge/mambaforge:latest

LABEL maintainer=""
LABEL version="0.0.1"

# make tmp accessible 
RUN chmod -R 777 /tmp

# Set the timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install utils and dependencies for pyinterp
RUN apt update && apt install -y \
 tmux \
 htop \
 nano \
 zip \
 git \
 cmake \
 g++ \
 libblas-dev \
 libgsl-dev \
 libeigen3-dev\
 libgtest-dev \
 python3-numpy && \
 echo 'set -g mouse on' > ~/.tmux.conf

# Install packages with mamba
RUN mamba install -y -c conda-forge \
 python=3.10 \
 'boost>=1.79' \
 pyinterp \
 matplotlib \
 cartopy \
 hvplot \
 arviz \
 metpy \
 'pandas>=2' \
 'xarray>=2023' \
 zarr \
 dask \
 netCDF4 \
 bottleneck \
 scipy \
 xrft \
 numpy_groupies \
 'xesmf>=0.7.0' \
 pint-xarray \
 gcm_filters \
 pytest \
 dvc \
 tqdm \
 brotlipy \
 cmocean \
 jupyter-book \
 ghp-import

# Install pip packages
RUN pip install --upgrade pip && \
    pip install \
 git+https://github.com/jejjohnson/ocn-tools.git \
 hydra-core \
 pyrootutils \
 loguru \
 xrpatcher \
 autoroot \
 einops \
 corner \
 ipykernel \
 deepsensor \
 wandb \
 torchgeo \
 black \
 isort \
 flake8

# Create a Python 3 Jupyter kernel
RUN python -m ipykernel install --user --name=oceanbenchKernel


ENV workdir /home/user
# Add the local module directory to the Python path
ENV PYTHONPATH "${PYTHONPATH}:/home/user/oceanbench"
WORKDIR ${workdir}
CMD bash