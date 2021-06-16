from utils.config import cfg


def get_algo(split):
    """
    Get the right trainer.

    Parameter
    ---------
    split : int
        Split number.
    """
    if cfg.TRAIN.MODEL == "likelihood":
        from .likelihoodtrainer import LikelihoodTrainer
        algo = LikelihoodTrainer(split)
    elif cfg.TRAIN.MODEL == "emd":
        from .emd_trainer import EMDTrainer
        algo = EMDTrainer(split)
    else:
        raise NotImplementedError()

    return algo
