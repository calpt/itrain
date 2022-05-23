# flake8: noqa

from .classification import ANLIManager, ClassificationDatasetManager, SciTailManager, WikiQAManager
from .dataset_manager import CacheMode, ColumnConfig, DatasetManager, DatasetManagerBase
from .dependency_parsing import UniversalDependenciesManager
from .glue import GlueManager
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
from .qa import QADatasetManager
from .super_glue import SuperGlueManager
from .tagging import FCEErrorDetectionManager, TaggingDatasetManager


DATASET_MANAGER_CLASSES = {
    "imdb": ClassificationDatasetManager,
    "rotten_tomatoes": ClassificationDatasetManager,
    "emo": ClassificationDatasetManager,
    "emotion": ClassificationDatasetManager,
    "yelp_polarity": ClassificationDatasetManager,
    "scicite": ClassificationDatasetManager,
    "trec": ClassificationDatasetManager,
    "snli": ClassificationDatasetManager,
    "eraser_multi_rc": ClassificationDatasetManager,
    "sick": ClassificationDatasetManager,
    "ag_news": ClassificationDatasetManager,
    "wiki_qa": WikiQAManager,
    "scitail": SciTailManager,
    "anli": ANLIManager,
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
    "squad": QADatasetManager,
    "squad_v2": QADatasetManager,
    "drop": QADatasetManager,
    "wikihop": QADatasetManager,
    "hotpotqa": QADatasetManager,
    # "triviaqa": QADatasetManager,
    "comqa": QADatasetManager,
    "cq": QADatasetManager,
    # "cwq": QADatasetManager,
    "newsqa": QADatasetManager,
    # "searchqa": QADatasetManager,
    "duorc_p": QADatasetManager,
    "duorc_s": QADatasetManager,
    "quoref": QADatasetManager,
    "conll2000": TaggingDatasetManager,
    "conll2003": TaggingDatasetManager,
    "ud_pos": TaggingDatasetManager,
    "ud_deprel": TaggingDatasetManager,
    "fce_error_detection": FCEErrorDetectionManager,
    "pmb_sem_tagging": TaggingDatasetManager,
    "wnut_17": TaggingDatasetManager,
    "mit_movie": TaggingDatasetManager,
    "ud_parsing": UniversalDependenciesManager,
}
