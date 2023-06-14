# OceanBench: Sea Surface Height Editions
## Overall structure

![Schema](Oceanbench.png)

The **Oceanbench** project aims to facilitate the development and evaluation of ML methods applied to ocean observation data.
For this purpose it provides:

@ https://github.com/quentinf00/oceanbench-data-registry.git
- a data registry at  with versioned and open access ocean data

@ https://github.com/jejjohnson/oceanbench.git
- preconfigured **SSH interpolation task** configurations with
  - preconfigured data validation and preprocessing pipelines
- A **xarray dataarray patcher** allowing to
  - slice a large ocean state into ML ingestible items
  - reconstructing the full state from models predictions
- **leaderboards** with preconfigured post-processing and evaluation pipelines

## Installation
In order to install the project:

- clone https://github.com/jejjohnson/oceanbench.git
- create conda environment
```

conda create -n oceanbench
conda activate oceanbench
cd oceanbench
mamba env update -f environment/linux.yaml
```


## Data Download
Follow the installation procedure above

- clone https://github.com/quentinf00/oceanbench-data-registry.git
- Download files:
```
cd oceanbench-data-registry

dvc pull -R osse_natl60/grid/natl* # osse tasks data (nadir, nadirswot, nadir_sst)
dvc pull -R ose/coord/gf* # ose task data

dvc pull -R results/ose_gf/* # ose task results
dvc pull -R results/osse_gf_nadir/* # osse_gf_nadir task results
dvc pull -R results/osse_gf_nadirswot/* # osse_gf_nadirswot task results
dvc pull -R results/osse_gf_nadir_sst/* # osse_gf_nadir_sst task results

```




## Getting Started


