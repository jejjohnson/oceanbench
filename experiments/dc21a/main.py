import autoroot
import hydra
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
import preprocess
import evaluation


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    if cfg.stage == "preprocess":
        logger.info(f"Starting preprocessing stage...!")
        preprocess.main(cfg)
        
    elif cfg.stage == "evaluation":
        logger.info(f"Starting evaluation stage...!")
        evaluation.main(cfg)
        
    elif cfg.stage == "viz":
        logger.info(f"Starting visualization stage...!")
        NotImplementedError()
    
    else:
        raise ValueError(f"Unrecognized stage: {cfg.stage}")
    

if __name__ == "__main__":
    main()