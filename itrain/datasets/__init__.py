# flake8: noqa

from .classification import (
    ANLIManager,
    ClassificationDatasetManager,
    EmotionDatasetManager,
    GlueManager,
    SciTailManager,
    SNLIManager,
    SuperGlueManager,
)
from .dataset_manager import CacheMode, ColumnConfig, DatasetManager, DatasetManagerBase
from .multiple_choice import ARTManager, HellaswagManager, MultipleChoiceDatasetManager, QuailManager, RaceManager
from .qa import (
    ComQAManager,
    CQManager,
    CWQManager,
    DROPManager,
    DuoRCParaphraseManager,
    DuoRCSelfManager,
    HotpotQAManager,
    NewsQAManager,
    QADatasetManager,
    SearchQAManager,
    SquadV1Manager,
    SquadV2Manager,
    TriviaQAManager,
    WikiHopManager,
)


DATASET_MANAGER_CLASSES = {
    "imdb": ClassificationDatasetManager,
    "rotten_tomatoes": ClassificationDatasetManager,
    "emo": ClassificationDatasetManager,
    "yelp_polarity": ClassificationDatasetManager,
    "snli": SNLIManager,
    "scitail": SciTailManager,
    "anli": ANLIManager,
    "emotion": EmotionDatasetManager,
    "glue": GlueManager,
    "super_glue": SuperGlueManager,
    "hellaswag": HellaswagManager,
    "race": RaceManager,
    "quail": QuailManager,
    "art": ARTManager,
    "squad": SquadV1Manager,
    "squad_v2": SquadV2Manager,
    "drop": DROPManager,
    "wikihop": WikiHopManager,
    "hotpotqa": HotpotQAManager,
    "triviaqa": TriviaQAManager,
    "comqa": ComQAManager,
    "cq": CQManager,
    "cwq": CWQManager,
    "newsqa": NewsQAManager,
    "searchqa": SearchQAManager,
    "duorc_p": DuoRCParaphraseManager,
    "duorc_s": DuoRCSelfManager,
}
