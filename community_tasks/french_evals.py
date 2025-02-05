# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

This module implements tasks for the french specific datasets
See : https://huggingface.co/fr-gouv-coordination-ia
"""

import random
from typing import Callable

import numpy as np
from aenum import extend_enum

import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.extended.ifeval.main import ifeval_metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


# custom Metric
class PrefixSuffixExactMatch:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = True,
    ):
        """An exact match class.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if pred.startswith(gold):  # prefix em
            return 1
        if pred.endswith(gold):  # suffix em
            return 1
        return 1 if gold == pred else 0  # strict em


prefixsuffix_quasi_exact_match = SampleLevelMetric(
    metric_name="psqem",
    sample_level_fn=PrefixSuffixExactMatch(
        normalize_gold=helm_normalizer,
        normalize_pred=helm_normalizer,
        strip_strings=True,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


# qpqa-fr prompt function
def prompt_gpqa_fr(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Réponse incorrecte 1"], line["Réponse incorrecte 2"], line["Réponse incorrecte 3"]]
    choices.insert(gold_index, line["Réponse correcte"])

    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"

    query = f"Question: {line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# BAC-fr prompt function
def prompt_bac_fr(line, task_name: str = None):
    prompt = f"Enoncé: {line['enonce']}\n"
    if line["instruction"] is not None:
        prompt += f"{line['instruction']}\n"
    prompt += "Réponse: "
    if line["choix"] is not None:  # Multichoice evaluation
        # prompt += "\n".join([f"{LETTER_INDICES[ix]}.{choix}" for ix, choix in enumerate(line["choix"])])
        return Doc(
            task_name=task_name,
            query=prompt,
            choices=as_list(line["choix"]),
            gold_index=as_list(line["choix"]).index(line["choix correct"]),
            instruction="",
        )
    else:
        return Doc(task_name=task_name, query=prompt, choices=[line["reponse"]], gold_index=0, instruction="")


# IFEVal-fr task


ifeval_fr_task = LightevalTaskConfig(
    name="ifeval-fr",
    prompt_function=prompt_ifeval_fr,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="fr-gouv-coordination-ia/IFEval-fr",
    hf_subset="default",
    metric=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",  # select your metric in Metrics
)

# GPQA-fr task
gpqa_fr_task = LightevalTaskConfig(
    name="gpqa-fr",
    suite=["community"],
    prompt_function=prompt_gpqa_fr,
    hf_repo="fr-gouv-coordination-ia/gpqa-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# BAC-fr task
bac_fr_task = LightevalTaskConfig(
    name="bac-fr",
    suite=["community"],
    prompt_function=prompt_bac_fr,
    hf_repo="fr-gouv-coordination-ia/bac-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    metric=[Metrics.quasi_exact_match, prefixsuffix_quasi_exact_match],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [ifeval_fr_task, gpqa_fr_task, bac_fr_task]
