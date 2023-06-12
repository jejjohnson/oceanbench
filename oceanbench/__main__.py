import hydra
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

METRICS_SUMMARY = []

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg):
    lib = hydra.utils.instantiate(cfg.outputs)

    logger.debug(f"Building and preprocessing eval dataset...")
    ds = lib.build_eval_ds()

    logger.debug(f"metrics summary...")
    metrics_summary = {}
    for name, value in lib.summary.items():

        logger.debug(f"Computing metric {name}...")
        metrics_summary[name] = value(ds) if callable(value) else value

    s = pd.Series(metrics_summary)
    logger.info(s.to_frame(name='Leaderboard').T.to_markdown())
    METRICS_SUMMARY.append(s)



if __name__ == "__main__":
    main()
    print(pd.concat(METRICS_SUMMARY, axis=1).T.to_markdown())
