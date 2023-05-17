# Data Challenge 2021a - Sea Surface Height OSE

This is the experiment repository for the 2021 OSE data challenge over the GulfStream using real observations from altimetry tracks.

---
## Preprocessing

**ALONGTRACK**

Here is an example with an alongtrack preprocessing pipeline

```bash
# ALONGTRACK | SUBSET DOMAIN | 1 day | 1/5 degree resolution
python main.py stage=preprocess
# ALONGTRACK | FULL DOMAIN | 12 hours | 1/20 degree resolution
python main.py stage=preprocess domain=all
```

**GRIDDED**

Here is an example with a gridded preprocessing pipeline

```bash
# GRIDDED | SUBSET DOMAIN | 1 day | 1/5 degree resolution
python main.py stage=preprocess grid=duacs
# GRIDDED | FULL DOMAIN | 12 hours | 1/20 degree resolution
python main.py stage=preprocess domain=all grid=natl60
```

---
## Results 

This uses a custom `.yaml` file where we can run the evaluation script.

```bash
python main.py stage=evaluation results=duacs ++overwrite_results=True
```

## Current LeaderBoard 

This provides the current leaderboard with these specific metrics for all of the evaluation datasets.

```bash
bash scripts/metrics.sh
```


## Results


#### BASELINE

*Optimal Interpolation*

```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_BASELINE.nc"
```

#### DUACS

*Optimal Interpolation* with optimized covariance matrices

```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_DUACS.nc"
```

#### MIOST


```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_MIOST.nc"
```

#### DYMOST


```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_DYMOST.nc"
```

#### BFN QG


```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_BFN.nc"
```


#### 4DVarNet

```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_4dvarNet.nc"
```


```bash
wget --user johnsonj@univ-grenoble-alpes.fr  --password tySIa6 --directory-prefix=/gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test/results "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_4dvarNet_2022.nc"
```