VARIABLES=(
    "sea_surface_height"
)

# VARIABLES=(
#     "sea_surface_height"
#     #"kinetic_energy"
#     #"relative_vorticity"
#     #"strain"
# )

# ####################################
# DUACS
# ####################################
echo "Starting DUACS Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting DUACS | NADIR | $i Results..."
   python main.py stage=evaluation results=duacs_nadir postprocess=$i
   echo "Starting BFN QG | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=duacs_swot postprocess=$i
done

# ####################################
# BFN-QG
# ####################################
echo "Starting BFN QG Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting BFN QG | NADIR | $i Results..."
   python main.py stage=evaluation results=bfnqg_nadir postprocess=$i
   echo "Starting BFN QG | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=bfnqg_swot postprocess=$i
done


# ####################################
# MIOST
# ####################################
echo "Starting MIOST Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MIOST| NADIR | $i Results..."
   python main.py stage=evaluation results=miost_nadir postprocess=$i
   echo "Starting MIOST | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=miost_swot postprocess=$i
done

# ####################################
# DYMOST
# ####################################

echo "Starting DYMOST Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting DYMOST| NADIR | $i Results..."
   python main.py stage=evaluation results=dymost_nadir postprocess=$i
   echo "Starting DYMOST | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=dymost_swot postprocess=$i
done


# ####################################
# 4DVARNET
# ####################################


echo "Starting 4DVARNet Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting 4DVARNet | NADIR | $i Results..."
   python main.py stage=evaluation results=4dvarnet_nadir postprocess=$i
   echo "Starting 4DVARNet | SWOT | $i Results..."
   python main.py stage=evaluation results=4dvarnet_swot postprocess=$i
done

# ####################################
# NERF (MLP)
# ####################################

echo "Starting MLP Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MLP | NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_mlp_nadir postprocess=$i
   echo "Starting MLP | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_mlp_swot postprocess=$i
done



# ####################################
# NERF (FFN)
# ####################################

echo "Starting FFN Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting FFN | NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_ffn_nadir postprocess=$i
   echo "Starting FFN | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_ffn_swot postprocess=$i
done


# ####################################
# NERF (SIREN)
# ####################################

echo "Starting SIREN Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting SIREN | NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_siren_nadir postprocess=$i
   echo "Starting SIREN | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_siren_swot postprocess=$i
done
