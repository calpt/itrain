# flake8: noqa

from .dataset_manager import (
    ANLIManager,
    CacheMode,
    ColumnConfig,
    DatasetManager,
    DatasetManagerBase,
    EmotionDatasetManager,
    GlueManager,
    SimpleClassificationManager,
    SNLIManager,
    SuperGlueManager,
)
from .multiple_choice import ARTManager, HellaswagManager, QuailManager, RaceManager
from .qa import SquadV1Manager, SquadV2Manager


DATASET_MANAGER_CLASSES = {
    "imdb": SimpleClassificationManager,
    "rotten_tomatoes": SimpleClassificationManager,
    "emo": SimpleClassificationManager,
    "yelp_polarity": SimpleClassificationManager,
    "snli": SNLIManager,
    "anli": ANLIManager,
    "emotion": EmotionDatasetManager,
    "glue": GlueManager,
    "super_glue": SuperGlueManager,
    "hellaswag": HellaswagManager,
    "race": RaceManager,
    "quail": QuailManager,
    "art": ARTManager,
    "squad": SquadV1Manager,
    "squad_v2": SquadV2Manager
}
