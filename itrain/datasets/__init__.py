# flake8: noqa

from .dataset_manager import (
    CacheMode,
    ColumnConfig,
    DatasetManager,
    GlueManager,
    HellaswagManager,
    RaceManager,
    SuperGlueManager,
)
from .squad import SquadV1Manager, SquadV2Manager


DATASET_MANAGER_CLASSES = {
    "glue": GlueManager,
    "super_glue": SuperGlueManager,
    "hellaswag": HellaswagManager,
    "race": RaceManager,
    "squad": SquadV1Manager,
    "squad_v2": SquadV2Manager
}
