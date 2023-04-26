# grab arguments
dir=$3
username=$1
password=$2

# make train directory
train_dir=$dir/train
echo train_dir

# make ref directory
ref_dir=$dir/ref

# make test directory
test_dir=$dir/test

# make results directory
result_dir=$dir/results

# GULFSTREAM (SARAL/Altika obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc'

# GULFSTREAM (Jason-3 obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc'

# GULFSTREAM (Sentinel-3A obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc'


# GULFSTREAM (Jason-2 obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc'


# GULFSTREAM (Jason-2n obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc'

# # GULFSTREAM (Cryosat-2 obs)
# wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc'

# GULFSTREAM (Haiyang-2 obs)
wget --user $username --password $password --directory-prefix=$train_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc'


# download reference map data
wget --user $username --password $password --directory-prefix=$ref_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/mdt.nc'

# download test data
wget --user $username --password $password --directory-prefix=$test_dir 'https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
