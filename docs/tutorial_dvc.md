# Dataset Download

> In this tutorial, we will look at which datasets we will use for the development stage of OceanBench.
> We will use `dvc` as a tool to handle all of the data in the AWS bucket.


---
## Setup Environment


First, we need to setup the environment using `dvc`. 
This step is only needed once.
In the `$HOME` directory, create a folder called `.aws` which will allow the functions to point to the amazon credentials
and buckets.

```bash
mkdir $HOME/.aws
```

In the `.aws` directory, create a `credentials` file with the `toml` format to store the necessary information about the bucket.

```yaml
[default]
access-key=XXXXXXXXX
secret-key=YYYYYYYYY
```

Again in the `.aws` directory, create a `config` file to with the `toml` format to  store the necessary information about

```yaml
[default]
region = us-east-1
```


If necessary, give the correct permissions

```bash
sudo chmod 600 credentials config
```


---
## Download Dataset Repo


There is already a repo which has access to all of the data we will be using. 
After we have setup our environment, we simply need to download the repo and then we can start downloading the data.
Download the `CIA-Oceanix/sla-data-registry` repo to see all of the available files.

Using `gh` 

```bash
gh repo clone CIA-Oceanix/sla-data-registry
```

or using `https`

```bash
git clone https://github.com/CIA-Oceanix/sla-data-registry.git
```


---
## Usage

There are two important commands when using this service: 1) listing the files available and 2) pulling any changes.
For the purposes of this project, we will only need these two commands (for now). 
Later we will add a workflow to push new changes.

**List Files**

We can see all of the files and subsequent directories using the `dvc list` command.

```bash
dvc list .
```


**Pull Changes**

We can pull the latest changes with the following command.

```bash
dvc pull NATL60/NATL/ref_new
```


---
## Important Files

Here are the important files and their directories. For the primila


For the preliminary work, we are working with simulations in an OSSE experiment. 
These simulations come from the NEMO model which cover the entire North Atlantic.
We also have two sets of pseudo-observations observations which were generated from the NEMO model. 
Below, we have some tables with the exact datasets we will be working with.
To see a full table for all of the datasets, [see here](https://jejjohnson.notion.site/OceanBench-Database-465eb05c436d416b98d4987c478c426c)

---
### Gridded Data

We have gridded data which are datasets that have are already defined on a regular constant grid. Example:

`data = u(lat x lon x time x variable)`

We have these from the NEMO model and we also have some pseudo-observations which are already "pre-gridded".
Lastly, we have some predictions from the Optimal Interpolation scheme from DUACS.

Below is a summary table of the important file paths.

#### North Atlantic

|  Data Origin   |  Data Type   |     Region      | Data Structure |                           Path                            |   Variables   | FileType |
|:--------------:|:------------:|:---------------:|:--------------:|:---------------------------------------------------------:|:-------------:|:--------:| 
|      NEMO      |  Simulation  | North Atlantic  |    Gridded     | ``NATL60/NATL/data_new/NATL60-CJM165_NATL_*_y2013.1y.nc`` | `ssh`, `sst`, `mld` | `netcdf` |
| SWOT-Simulator |    NADIR     | North Atlantic  |    Gridded     |        `NATL60/NATL/data_new/dataset_nadir_0d.nc`         |     `ssh`      | `netcdf`  |
| SWOT-Simulator |     SWOT     | North Atlantic  |    Gridded     |          `NATL60/NATL/data_new/dataset_swot.nc`           |     `ssh`      | `netcdf`  |
| SWOT-Simulator | NADIR + SWOT | North Atlantic  |    Gridded     |      `NATL60/NATL/data_new/dataset_nadir_0d_swot.nc`      |     `ssh`      | `netcdf`  |


#### GulfStream

We can use the GULFSTREAM region as toy some datasets to play around with if the NA is too big.
Below are the main files we will work with.


|      Data Origin      |  Data Type   | Data Structure |                           Path                            |   Variables   | FileType |
|:---------------------:|:------------:|:--------------:|:---------------------------------------------------------:|:-------------:|:--------:| 
|         NEMO          |  Simulation   |    Gridded     | ``NATL60/NATL/data_new/NATL60-CJM165_NATL_*_y2013.1y.nc``  | ssh, sst, mld | `netcdf` |
| SWOT-Simulator (NEMO) |    NADIR      |    Gridded     |   `NATL60/NATL/data_new/dataset_nadir_0d.nc`    |     `ssh`      | `netcdf`  |
| SWOT-Simulator (NEMO) |     SWOT      |    Gridded     |     `NATL60/NATL/data_new/dataset_swot.nc`      |     `ssh`      | `netcdf`  |
| SWOT-Simulator (NEMO) | NADIR + SWOT  |    Gridded     | `NATL60/NATL/data_new/dataset_nadir_0d_swot.nc` |     `ssh`      | `netcdf`  |
|         DUACS         |     SWOT      |    Gridded     |     `NATL60/NATL/data_new/dataset_swot.nc`      |     `ssh`      | `netcdf`  |
|         DUACS         | NADIR + SWOT  |    Gridded     | `NATL60/NATL/data_new/dataset_nadir_0d_swot.nc` |     `ssh`      | `netcdf`  |


---
### AlongTrack Data

We also have alongtrack data which are pseudo-observations on the satellite. 
These are irregular and not in the gridded format as above. 
It can be thought of as a vector with the coordinates and variables of interest.

`data = (lat, lon, time, swath, variable)`

|      Data Origin      |     Data Type      | Data Structure |                Path                 |   Variables   | FileType |
|:---------------------:|:------------------:|:--------------:|:-----------------------------------:|:-------------:|:--------:| 
| SWOT-Simulator (NEMO) |       NADIR        |   AlongTrack   |      `sensor_zarr/zarr/nadir/`      |     `ssh`      |  `zarr`  |
| SWOT-Simulator (NEMO) |        SWOT        |    AlongTrack     |       `sensor_zarr/zarr/swot`       |     `ssh`      | `zarr` |
| SWOT-Simulator (NEMO) |    SWOT (error)    |    AlongTrack     |     `sensor_zarr/zarr/new_swot`     |     `ssh`      | `zarr` |
| SWOT-Simulator (NEMO) | SWOT (error param) |    AlongTrack     | `sensor_zarr/zarr/new_swot_with_1d` |     `ssh`      | `zarr` |


