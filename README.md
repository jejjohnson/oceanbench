# OceanBench: Sea Surface Height Edition

[**About**](#about) 
| [**Tutorials**](#tutorials)
| [**Quickstart**](#quickstart)
| [**Installation**](#installation)

![pyver](https://img.shields.io/badge/python-3.9%203.10%203.11_-red)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/notebooks/pytorch_integration.ipynb)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jejjohnson/oceanbench)
[![JupyterBook][jbook-badge]][jbook-link]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


## About<a id="about"></a>

**OceanBench** is a unifying framework that provides standardized processing steps that comply with domain-expert standards. 
It is designed with a flexible and pedagogical abstraction to do:
1. provides plug-and-play data and pre-configured pipelines for ML researchers to benchmark their models w.r.t. ML and domain-related baselines
2. provides a transparent and configurable framework for researchers to customize and extend the pipeline for their tasks.

It is lightweight in terms of the core functionality.
We keep the code base simple and focus more on how the user can combine each piece.
We adopt a strict functional style because it is easier to maintain and combine sequential transformations.


There are five features we would like to highlight about OceanBench:
1. Data availability and version control with [DVC](https://dvc.org/).
2. An agnostic suite of geoprocessing tools for [xarray](https://docs.xarray.dev/en/stable/) datasets that were aggregated from different sources
3. [Hydra](https://github.com/facebookresearch/hydra) integration to pipe sequential transformations
4. [xrpatcher](https://github.com/jejjohnson/xrpatcher/tree/main) - A flexible multi-dimensional array generator from xarray datasets that are compatible with common deep learning (DL) frameworks
5. A [JupyterBook](https://jejjohnson.github.io/oceanbench/content/overview.html) that offers library tutorials and demonstrates use-cases.
In the following section, we highlight these components in more detail.


## Tutorials<a id="tutorials"></a> [![JupyterBook][jbook-badge]][jbook-link]

We have a fully fledged [Jupyter-Book]() available to showcase how OceanBench can be used in practice.
There are some *quickstart* tutorials and there are also some more detailed tutorials which highlight some of the intricacies of OceanBench.
Some highlighted tutorials are listed in the next section.

## Quickstart<a id="quickstart"></a>

**Data Registry** [![Open In GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/quentinf00/oceanbench-data-registry)

We have a open data registry located at the [oceanbench-data-registry](https://github.com/quentinf00/oceanbench-data-registry) GitHub repository.
You can find some more meta-data about the available datasets as well as how to download them yourself.

**Machine Learning Datasets** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://jejjohnson.github.io/oceanbench/content/getting_started/TaskToPatcher.html)

We have a set of tasks related to sea surface height interpolation and they come readily integrated into a digestable ML-ready format.
We use our custom [xrpatcher](https://github.com/jejjohnson/xrpatcher) package to pipe `xarray` data structures to PyTorch datasets/dataloaders.
For ML researchers who want to see how they can get started quickly, look at our *Task-to-Patcher* demo available.
For more information about the datasets, see the [oceanbench-data-registry](https://github.com/quentinf00/oceanbench-data-registry).

**LeaderBoard** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://jejjohnson.github.io/oceanbench/content/getting_started/Leaderboards.html)

OceanBench can be used to generate the leaderboard for our different interpolation challenges.
To generate the leaderboards for different tasks with the available data we have, look at our *LeaderBoard* demo.

**Machine Learning Example** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://jejjohnson.github.io/oceanbench/content/getting_started/ocean_bench_4dvarnet.html)

Currently, the most successful algorithm for the SSH challenges is a Bi-Level Optimization algorithm (4DVarNet). 
To see a reproducible end-to-end example for how a SOTA method was used in conjunction with OceanBench, see our *End-to-End* demo.

## Installation<a id="installation"></a>

### `conda` (RECOMMENDED)

We use conda/mamba as our package manager. To install from the provided environment files
run the following command.

```bash
git clone https://github.com/jejjohnson/oceanbench.git
cd oceanbench
mamba env create -n environments/linux.yaml
```

#### Jupyter 
if you want to add the oceanbench conda environment as a jupyter kernel, you need to set the ESMF environment variable:

```
conda activate oceanbench
mamba install ipykernel -y 
python -m ipykernel install --user --name=oceanbench --env ESMFMKFILE "$ESMFMKFILE"
```

### `pip`

We can directly install it via pip from the.

```bash
pip install "git+https://github.com/jejjohnson/oceanbench.git"
```

**Note**: There are some known dependency issues related to `pyinterp` and `xesmf`. 
You may need to manually install some of the dependencies before installing oceanbench via pip.
See the [pyinterp](https://pangeo-pyinterp.readthedocs.io/en/latest/setup/pip.html) and [xesmf](https://xesmf.readthedocs.io/en/latest/installation.html) packages for more information.

### `poetry`

For developers who want all of the dependencies via pip, we can use poetry to install the package.


```bash
git clone https://github.com/jejjohnson/oceanbench.git
cd oceanbench
conda create -n oceanbench python=3.10 poetry
poetry install
```

## Acknowledgements

We would like to acknowledge the [Ocean-Data-Challenge Group](https://ocean-data-challenges.github.io/) for all of their work for providing open source data and a tutorial of metrics for SSH interpolation.



[jbook-badge]: https://raw.githubusercontent.com/executablebooks/jupyter-book/master/docs/images/badge.svg
[jbook-link]: https://jejjohnson.github.io/oceanbench/content/overview.html
