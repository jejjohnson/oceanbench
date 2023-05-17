
# DUACS
echo "Starting DUACS Results..."
python main.py stage=evaluation results=duacs ++overwrite_results=True

# BFN-QG
echo "Starting BFN QG Results..."
python main.py stage=evaluation results=bfnqg

# MIOST
echo "Starting MIOST Results..."
python main.py stage=evaluation results=miost

# DYMOST
echo "Starting DYMOST Results..."
python main.py stage=evaluation results=dymost

# 4DVARNET
echo "Starting 4DVARNet Results..."
python main.py stage=evaluation results=4dvarnet

# # NERF (FFN)
# echo "Starting FFN | NADIR Results..."
# python main.py stage=evaluation results=nerf_ffn_nadir

# # NERF (MLP)

# # NERF (SIREN)