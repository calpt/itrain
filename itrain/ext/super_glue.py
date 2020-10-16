# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The SuperGLUE benchmark metric."""

import datasets
from sklearn.metrics import f1_score, matthews_corrcoef


_CITATION = """\
@article{wang2019superglue,
  title={SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
  author={Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R},
  journal={arXiv preprint arXiv:1905.00537},
  year={2019}
}
"""

_DESCRIPTION = """\
SuperGLUE (https://super.gluebenchmark.com/) is a new benchmark styled after
GLUE with a new set of more difficult language understanding tasks, improved
resources, and a new public leaderboard.
"""

_KWARGS_DESCRIPTION = """
Compute SuperGLUE evaluation metric associated to each SuperGLUE dataset.
Args:
    predictions: list of predictions to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
    "accuracy": Accuracy
    "f1": F1 score
    "pearson": Pearson Correlation
    "spearmanr": Spearman Correlation
    "matthews_correlation": Matthew Correlation
"""


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, f1_avg="binary"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
    return {
        "accuracy": acc,
        "f1": f1,
    }


class SuperGlue(datasets.Metric):
    def _info(self):
        if self.config_name not in [
            "boolq",
            "cb",
            "copa",
            "multirc",
            "record",
            "rte",
            "wic",
            "wsc",
            "wsc.fixed",
            "axb",
            "axg",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc", "wsc.fixed", "axb", "axg",]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64" if self.config_name != "stsb" else "float32"),
                    "references": datasets.Value("int64" if self.config_name != "stsb" else "float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        # TODO missing multirc & record
        if self.config_name == "axb":
            return {"matthews_correlation": matthews_corrcoef(references, predictions)}
        elif self.config_name == "cb":
            return acc_and_f1(predictions, references, f1_avg="macro")
        # elif self.config_name == "record":
        #     return acc_and_f1(predictions, references)
        elif self.config_name in ["copa", "rte", "wic", "wsc", "wsc.fixed", "boolq", "axg"]:
            return {"accuracy": simple_accuracy(predictions, references)}
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc", "wsc.fixed", "axb", "axg",]'
            )