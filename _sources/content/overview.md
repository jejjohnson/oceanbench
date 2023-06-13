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
