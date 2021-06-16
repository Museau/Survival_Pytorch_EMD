import numpy as np
import yaml

from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True


# Dataset options
__C.DATA = edict()
__C.DATA.DEATH_AT_CENSOR_TIME = False
__C.DATA.NO_CENSORED_DATA = False
__C.DATA.ADD_CENS = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b,
    clobbering the options in b whenever they are also specified in a.

    Parameters
    ----------
    a : dict
        Config dictionary a.
    b : dict
        Config dictionary b.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            # raise KeyError('{} is not a valid config key'.format(k))
            b[k] = v

        else:
            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(f"Type mismatch ({type(b[k])} vs. {type(v)}) for config key: {k}")

            # recursively merge dicts
            if type(v) is edict:
                try:
                    _merge_a_into_b(a[k], b[k])
                except Exception:
                    print(f"Error under config key: {k}")
                    raise
            else:
                b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.

    Parameters
    ----------
    filename : str
        Path to filename.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.Loader))

    _merge_a_into_b(yaml_cfg, __C)
