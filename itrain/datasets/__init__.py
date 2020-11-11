# flake8: noqa

from .dataset_manager import (
    CacheMode,
    ColumnConfig,
    DatasetManager,
    DatasetManagerBase,
    GlueManager,
    SimpleClassificationManager,
    SuperGlueManager,
)
from .multiple_choice import HellaswagManager, RaceManager
from .squad import SquadV1Manager, SquadV2Manager


DATASET_MANAGER_CLASSES = {
    "imdb": SimpleClassificationManager,
    "rotten_tomatoes": SimpleClassificationManager,
    "glue": GlueManager,
    "super_glue": SuperGlueManager,
    "hellaswag": HellaswagManager,
    "race": RaceManager,
    "squad": SquadV1Manager,
    "squad_v2": SquadV2Manager
}
