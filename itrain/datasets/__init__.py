# flake8: noqa

from .classification import (
    ANLIManager,
    ClassificationDatasetManager,
    EmotionDatasetManager,
    GlueManager,
    SciTailManager,
    SuperGlueManager,
    WikiQAManager,
)
from .dataset_manager import CacheMode, ColumnConfig, DatasetManager, DatasetManagerBase
from .multiple_choice import (
    ARTManager,
    CommonsenseQAManager,
    COPAManager,
    CosmosQAManager,
    HellaswagManager,
    MultipleChoiceDatasetManager,
    QuailManager,
    QuartzManager,
    RaceManager,
    SocialIQAManager,
    SWAGManager,
    WinograndeManager,
)
from .qa import (
    ComQAManager,
    CQManager,
    DROPManager,
    DuoRCParaphraseManager,
    DuoRCSelfManager,
    HotpotQAManager,
    NewsQAManager,
    QADatasetManager,
    QuorefManager,
    SquadV1Manager,
    SquadV2Manager,
    WikiHopManager,
)
from .tagging import FCEErrorDetectionManager, TaggingDatasetManager


DATASET_MANAGER_CLASSES = {
    "imdb": ClassificationDatasetManager,
    "rotten_tomatoes": ClassificationDatasetManager,
    "emo": ClassificationDatasetManager,
    "yelp_polarity": ClassificationDatasetManager,
    "scicite": ClassificationDatasetManager,
    "trec": ClassificationDatasetManager,
    "snli": ClassificationDatasetManager,
    "eraser_multi_rc": ClassificationDatasetManager,
    "sick": ClassificationDatasetManager,
    "wiki_qa": WikiQAManager,
    "scitail": SciTailManager,
    "anli": ANLIManager,
    "emotion": EmotionDatasetManager,
    "glue": GlueManager,
    "super_glue": SuperGlueManager,
    "copa": COPAManager,
    "hellaswag": HellaswagManager,
    "swag": SWAGManager,
    "race": RaceManager,
    "quail": QuailManager,
    "art": ARTManager,
    "social_i_qa": SocialIQAManager,
    "cosmos_qa": CosmosQAManager,
    "commonsense_qa": CommonsenseQAManager,
    "quartz": QuartzManager,
    "winogrande": WinograndeManager,
    "squad": SquadV1Manager,
    "squad_v2": SquadV2Manager,
    "drop": DROPManager,
    "wikihop": WikiHopManager,
    "hotpotqa": HotpotQAManager,
    # "triviaqa": TriviaQAManager,
    "comqa": ComQAManager,
    "cq": CQManager,
    # "cwq": CWQManager,
    "newsqa": NewsQAManager,
    # "searchqa": SearchQAManager,
    "duorc_p": DuoRCParaphraseManager,
    "duorc_s": DuoRCSelfManager,
    "quoref": QuorefManager,
    "conll2000": TaggingDatasetManager,
    "conll2003": TaggingDatasetManager,
    "ud_pos": TaggingDatasetManager,
    "ud_deprel": TaggingDatasetManager,
    "fce_error_detection": FCEErrorDetectionManager,
    "pmb_sem_tagging": TaggingDatasetManager,
    "wnut_17": TaggingDatasetManager,
    "mit_movie": TaggingDatasetManager,
}
