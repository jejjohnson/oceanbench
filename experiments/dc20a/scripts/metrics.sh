VARIABLES=(
    "sea_surface_height"
    "kinetic_energy"
    "relative_vorticity"
    "strain"
)

# ####################################
# DUACS
# ####################################
echo "Starting DUACS | NADIR | SSH Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting DUACS | NADIR | $i Results..."
   python main.py stage=evaluation results=duacs_nadir postprocess=$i
done

echo "Starting DUACS | SWOT-NADIR Results..."

for i in $${VARIABLES[@]}
do
   echo "Starting BFN QG | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=duacs_swot postprocess=$i
done

# ####################################
# BFN-QG
# ####################################
echo "Starting BFN QG | NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting BFN QG | NADIR | $i Results..."
   python main.py stage=evaluation results=bfnqg_nadir postprocess=$i
done

echo "Starting BFN QG | SWOT-NADIR Results..."

for i in ${VARIABLES[@]}
do
   echo "Starting BFN QG | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=bfnqg_swot postprocess=$i
done

# ####################################
# MIOST
# ####################################
echo "Starting MIOST | NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MIOST| NADIR | $i Results..."
   python main.py stage=evaluation results=miost_nadir postprocess=$i
done

echo "Starting MIOST | SWOT-NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MIOST | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=miost_swot postprocess=$i
done

# ####################################
# DYMOST
# ####################################

echo "Starting DYMOST | NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MIOST| NADIR | $i Results..."
   python main.py stage=evaluation results=dymost_nadir postprocess=$i
done

echo "Starting DYMOST | SWOT-NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting MIOST | SWOT-NADIR | $i Results..."
   python main.py stage=evaluation results=dymost_swot postprocess=$i
done

# ####################################
# 4DVARNET
# ####################################


echo "Starting 4DVARNet | NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting 4DVARNet | NADIR | $i Results..."
   python main.py stage=evaluation results=4dvarnet_nadir postprocess=$i
done

echo "Starting 4DVARNet | SWOT-NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting 4DVARNet | NADIR | $i Results..."
   python main.py stage=evaluation results=4dvarnet_swot postprocess=$i
done


# ####################################
# NERF (FFN)
# ####################################

echo "Starting FFN | NADIR Results..."
for i in ${VARIABLES[@]}
do
   echo "Starting FFN | NADIR | $i Results..."
   python main.py stage=evaluation results=nerf_ffn_nadir postprocess=$i
done