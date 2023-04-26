

# Results

## Example Results 

This uses a custom `.yaml` file where we can run the evaluation script.

```bash
python main.py stage=evaluation results=duacs_swot ++overwrite_results=True postprocess=relative_vorticity
```

```bash
python main.py stage=evaluation results=nerf_ffn_nadir ++overwrite_results=True postprocess=sea_surface_height ++csv_name=results_nerf
```

## Current LeaderBoard 

This provides the current leaderboard with these specific metrics for all of the evaluation datasets.

```bash
bash scripts/metrics.sh
```

## Previous Results

Below we have some of the previous results


### DUACS


**NADIR 4**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_DUACS_en_j1_tpn_g2.nc
```

**SWOT 1 + NADIR 5**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_DUACS_swot_en_j1_tpn_g2.nc
```


---

### MIOST

**NADIR 4**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_MIOST_en_j1_tpn_g2.nc
```

**SWOT 1 + NADIR 5**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_MIOST_swot_en_j1_tpn_g2.nc
```


---

### DYMOST


**NADIR 4**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_DYMOST_Dynamic_en_j1_tpn_g2.nc
```

**SWOT 1 + NADIR 5**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_DYMOST_Dynamic_swot_en_j1_tpn_g2.nc
```

---

### BFN

This is the *Backwards-Forward Nudging with Quasi-Geostrophic Equations*

**NADIR 4**


```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_BFN_Steady_State_QG1L_en_j1_tpn_g2.nc
```

**SWOT 1 + NADIR 5**


```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_BFN_Steady_State_QG1L_swot_en_j1_tpn_g2.nc
```

---

#### 4DVarNet


**NADIR 4**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_4DVarNet_v2022_nadir_GF_GF.nc
```


**SWOT 1 + NADIR 5**

```bash
wget -nc https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_mapping/2020a_SSH_mapping_NATL60_4DVarNet_v2022_nadirswot_GF_GF.nc
```