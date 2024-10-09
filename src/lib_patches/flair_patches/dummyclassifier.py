import random
from collections import defaultdict
from typing import Literal

import flair.nn
import torch
import torch.nn
from flair.data import Dictionary, Sentence
from flair.embeddings import Embeddings
from loguru import logger


class DummyTextClassifier(
    flair.nn.DefaultClassifier[Sentence, Sentence]):
    def __init__(
            self,
            baseline_type: Literal["majority", "distribution", "uniform"],
            classifier_type: Literal["TextClassifier", "TextPairClassifier", "TextTripleClassifier", "SequenceTagger"],
            label_type: str = "predicted",
            multi_label: bool = False,
    ):
        super().__init__(embeddings=Embeddings(), label_dictionary=Dictionary(), final_embedding_size=1)
        logger.info(f"Initializing {baseline_type} DummyTextClassifier")
        assert baseline_type in [
            "majority",
            "distribution",
            "uniform",
        ], "BaselineClassifier only works as a majority, distribution and uniform baseline"
        assert classifier_type in [
            "TextClassifier",
            "TextPairClassifier",
            "TextTripleClassifier",
            "SequenceTagger",
        ], "BaselineClassifier only works with TextClassifier, TextPairClassifier, TextTripleClassifier and SequenceTagger"

        self.baseline_type = baseline_type
        self.classifier_type = classifier_type
        self.label_type = label_type
        self.label_distribution = defaultdict(int)
        assert not multi_label, "BaselineClassifier does not support multi_label"
        self.multi_label = multi_label

    def forward_loss(self, sentences: list[Sentence]) -> tuple[torch.Tensor, int]:
        for s in sentences:
            for l in s.get_labels():
                self.label_distribution[l.value] += 1
        return (torch.tensor(0.0, requires_grad=True), len(sentences))

    def predict(self, sentences, label_name, *args, **kwargs):
        for s in sentences:
            to_be_labelled = []
            if "classifier" in self.classifier_type.lower():
                to_be_labelled.append(s)
            elif "tagger" in self.classifier_type.lower():
                to_be_labelled.extend([token for token in s])

            for dings in to_be_labelled:
                dings.remove_labels(label_name)
                if self.baseline_type == "majority":
                    dings.add_label(
                        typename=label_name,
                        value=max(self.label_distribution, key=self.label_distribution.get),
                        score=1.0,
                    )
                elif self.baseline_type == "uniform":
                    dings.add_label(
                        typename=label_name,
                        value=random.choice(list(self.label_distribution.keys())),
                        score=1.0 / len(self.label_distribution),
                    )
                elif self.baseline_type == "distribution":
                    label = random.choices(
                        list(self.label_distribution.keys()), weights=self.label_distribution.values()
                    )[0]
                    dings.add_label(
                        typename=label_name,
                        value=label,
                        score=self.label_distribution[label] / sum(self.label_distribution.values()),
                    )
                else:
                    raise NotImplementedError
        return torch.tensor(0.0, requires_grad=True), len(sentences)

    def label_type(self):
        return self._label_type

    def _get_data_points_from_sentence(self, sentence: Sentence) -> list[Sentence]:
        return [sentence]

    def _get_embedding_for_data_point(*args, **kwargs):
        return torch.tensor(0.0, requires_grad=True)
