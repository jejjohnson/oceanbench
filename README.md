# OceanBench: Sea Surface Height Edition

Primary Devs:
* J. Emmanuel Johnson
* Quentin Febvre

Other Devs:
* Maxime B
* Sammy Metref
* Maxime B
* Ronan Fablet

---
## About

This repo will provide the full preprocessing and postprocessing pipeline for sea surface height interpolation.
It builds upon the SSH interpolation data challenges already provide across several repos.
This will prove the data-driving learning community with a suite of tools to make the modeling
framework easier.

---

## Package Elements


### Data Storage


We use `AWS` as our primary storage (for now) with `dvc` to track the data changes.

### Configurations

We use `hydra` for all of our configuration definitions.

### Geopreprocessing

We use `xarray` and various other packages to handle the geo-processing (e.g. slicing, etc)

### Storage

We use `.zarr` files (with an xarray backend) to store all datasets for training.


### Datasets, DataLoaders, DataModules

We use `pytorch-lightning` to handle our datasets, dataloaders and data modules.


---
## Tutorials


* How to download the data from AWS (**TODO**)
* How to load the data from `dvc` (**TODO**)

---

## Installation Guide

We use conda/mamba as our package manager. To install from the provided environment files
run the following command.

```bash
mamba env create -n environments/linux.yaml
```

**Note**: we also have a `macos.yaml` file for MACOS users as well.