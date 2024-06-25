import typing
from pathlib import Path
from typing import List, Union

import flair
import flair.embeddings
import flair.nn
import torch
from flair.data import DT, DT2, Corpus, DataPoint, FlairDataset, Sentence, _iter_dataset
from flair.datasets.base import find_train_dev_test_files

DT3 = typing.TypeVar("DT3", bound=DataPoint)


class DataTriple(DataPoint, typing.Generic[DT, DT2, DT3]):
    def __init__(self, first: DT, second: DT2, third: DT3):
        super().__init__()
        self.first = first
        self.second = second
        self.third = third

    def to(self, device: str, pin_memory: bool = False):
        self.first.to(device, pin_memory)
        self.second.to(device, pin_memory)
        self.third.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):
        self.first.clear_embeddings(embedding_names)
        self.second.clear_embeddings(embedding_names)
        self.third.clear_embeddings(embedding_names)

    @property
    def embedding(self):
        return torch.cat([self.first.embedding, self.second.embedding, self.third.embedding])

    def __len__(self):
        return len(self.first) + len(self.second) + len(self.third)

    @property
    def unlabeled_identifier(self):
        return f"DataPair: '{self.first.unlabeled_identifier}' + '{self.second.unlabeled_identifier}' + '{self.third.unlabeled_identifier}'"

    @property
    def start_position(self) -> int:
        return self.first.start_position

    @property
    def end_position(self) -> int:
        return self.first.end_position

    @property
    def text(self):
        return self.first.text + " || " + self.second.text + "||" + self.third.text


class DataTripleCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        columns: List[int] = [0, 1, 2, 3],
        train_file=None,
        test_file=None,
        dev_file=None,
        use_tokenizer: bool = True,
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        in_memory: bool = True,
        label_type: str = None,
        autofind_splits=True,
        sample_missing_splits: bool = True,
        skip_first_line: bool = False,
        separator: str = "\t",
        encoding: str = "utf-8",
    ):
        """
        Corpus for tasks involving triplets of sentences or paragraphs. The data files are expected to be in column format where each line has columns
        for the first sentence/paragraph, the second sentence/paragraph, the third sentence/paragraph, and the labels, respectively.
        The columns must be separated by a given separator (default: '\t').

        :param data_folder: base folder with the task data
        :param columns: List that indicates the columns for the first sentence (first entry in the list),
                        the second sentence (second entry), the third sentence (third entry), and label (last entry).
                        default = [0,1,2,3]
        :param train_file: the name of the train file
        :param test_file: the name of the test file, if None, dev data is sampled from train (if sample_missing_splits is true)
        :param dev_file: the name of the dev file, if None, dev data is sampled from train (if sample_missing_splits is true)
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param in_memory: If True, data will be saved in list of flair.data.DataTriple objects, otherwise we use lists with simple strings which need less space
        :param label_type: Name of the label of the data triples
        :param autofind_splits: If True, train/test/dev files will be automatically identified in the given data_folder
        :param sample_missing_splits: If True, a missing train/test/dev file will be sampled from the available data
        :param skip_first_line: If True, the first line of data files will be ignored
        :param separator: Separator between columns in data files
        :param encoding: Encoding of data files

        :return: a Corpus with annotated train, dev, and test data
        """

        # find train, dev, and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder,
            dev_file,
            test_file,
            train_file,
            autofind_splits=autofind_splits,
        )

        # create DataTripleDataset for train, test, and dev files, if they are given

        train = (
            DataTripleDataset(
                train_file,
                columns=columns,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                label_type=label_type,
                skip_first_line=skip_first_line,
                separator=separator,
                encoding=encoding,
            )
            if train_file is not None
            else None
        )

        test = (
            DataTripleDataset(
                test_file,
                columns=columns,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                label_type=label_type,
                skip_first_line=skip_first_line,
                separator=separator,
                encoding=encoding,
            )
            if test_file is not None
            else None
        )

        dev = (
            DataTripleDataset(
                dev_file,
                columns=columns,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                label_type=label_type,
                skip_first_line=skip_first_line,
                separator=separator,
                encoding=encoding,
            )
            if dev_file is not None
            else None
        )

        super(DataTripleCorpus, self).__init__(
            train,
            dev,
            test,
            sample_missing_splits=sample_missing_splits,
            name=str(data_folder),
        )


TextTriple = DataTriple[Sentence, Sentence, Sentence]


class DataTripleDataset(FlairDataset):
    def __init__(
        self,
        path_to_data: Union[str, Path],
        columns: List[int] = [0, 1, 2, 3],
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        use_tokenizer=True,
        in_memory: bool = True,
        label_type: str = None,
        skip_first_line: bool = False,
        separator: str = "\t",
        encoding: str = "utf-8",
        label: bool = True,
    ):
        """
        Creates a Dataset for triplets of sentences/paragraphs. The file needs to be in a column format,
        where each line has columns for the first sentence/paragraph, the second sentence/paragraph,
        the third sentence/paragraph, and the label separated by e.g. '\t'.
        For each data triplet, we create a flair.data.DataTriple object.

        :param path_to_data: path to the data file
        :param columns: list of integers that indicate the respective columns. The first entry is the column
        for the first sentence, the second for the second sentence, the third for the third sentence,
        and the fourth for the label. Default [0, 1, 2, 3]
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param use_tokenizer: Whether or not to use the in-built tokenizer
        :param in_memory: If True, data will be saved in a list of flair.data.DataTriple objects, otherwise we use lists with simple strings which need less space
        :param label_type: Name of the label of the data triples
        :param skip_first_line: If True, the first line of the data file will be ignored
        :param separator: Separator between columns in the data file
        :param encoding: Encoding of the data file
        :param label: If False, the dataset expects unlabeled data
        """

        path_to_data = Path(path_to_data)

        # stop if the file does not exist
        assert path_to_data.exists()

        self.in_memory = in_memory

        self.use_tokenizer = use_tokenizer

        self.max_tokens_per_doc = max_tokens_per_doc

        self.label = label

        assert label_type is not None
        self.label_type = label_type

        self.total_data_count: int = 0

        if self.in_memory:
            self.data_triples: List[DataTriple] = []
        else:
            self.first_elements: List[str] = []
            self.second_elements: List[str] = []
            self.third_elements: List[str] = []
            self.labels: List[typing.Optional[str]] = []

        with open(str(path_to_data), encoding=encoding) as source_file:
            source_line = source_file.readline()

            if skip_first_line:
                source_line = source_file.readline()

            while source_line:
                source_line_list = source_line.strip().split(separator)

                first_element = source_line_list[columns[0]]
                second_element = source_line_list[columns[1]]
                third_element = source_line_list[columns[2]]

                if self.label:
                    triple_label: typing.Optional[str] = source_line_list[columns[3]]
                else:
                    triple_label = None

                if max_chars_per_doc > 0:
                    first_element = first_element[:max_chars_per_doc]
                    second_element = second_element[:max_chars_per_doc]
                    third_element = third_element[:max_chars_per_doc]

                if self.in_memory:
                    data_triple = self._make_data_triple(first_element, second_element, third_element, triple_label)
                    self.data_triples.append(data_triple)
                else:
                    self.first_elements.append(first_element)
                    self.second_elements.append(second_element)
                    self.third_elements.append(third_element)
                    if self.label:
                        self.labels.append(triple_label)

                self.total_data_count += 1

                source_line = source_file.readline()

    # create a DataTriple object from strings
    def _make_data_triple(self, first_element: str, second_element: str, third_element: str, label: str = None):
        first_sentence = Sentence(first_element, use_tokenizer=self.use_tokenizer)
        second_sentence = Sentence(second_element, use_tokenizer=self.use_tokenizer)
        third_sentence = Sentence(third_element, use_tokenizer=self.use_tokenizer)

        if self.max_tokens_per_doc > 0:
            first_sentence.tokens = first_sentence.tokens[: self.max_tokens_per_doc]
            second_sentence.tokens = second_sentence.tokens[: self.max_tokens_per_doc]
            third_sentence.tokens = third_sentence.tokens[: self.max_tokens_per_doc]

        data_triple = TextTriple(first_sentence, second_sentence, third_sentence)

        if label:
            data_triple.add_label(typename=self.label_type, value=label)

        return data_triple

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_data_count

    # if in_memory is True we return a DataTriple, otherwise we create one from the lists of strings
    def __getitem__(self, index: int = 0) -> DataTriple:
        if self.in_memory:
            return self.data_triples[index]
        elif self.label:
            return self._make_data_triple(
                self.first_elements[index],
                self.second_elements[index],
                self.third_elements[index],
                self.labels[index],
            )
        else:
            return self._make_data_triple(
                self.first_elements[index], self.second_elements[index], self.third_elements[index]
            )


class TextTripleClassifier(flair.nn.DefaultClassifier[TextTriple, TextTriple]):
    def __init__(
        self,
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        embed_separately: bool = False,
        **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """
        super().__init__(
            **classifierargs,
            embeddings=embeddings,
            final_embedding_size=3 * embeddings.embedding_length if embed_separately else embeddings.embedding_length,
            should_embed_sentence=False,
        )

        self._label_type = label_type

        self.embed_separately = embed_separately

        if not self.embed_separately:
            # set separator to concatenate three sentences
            self.sep = " "
            if isinstance(
                self.embeddings,
                flair.embeddings.document.TransformerDocumentEmbeddings,
            ):
                if self.embeddings.tokenizer.sep_token:
                    self.sep = " " + str(self.embeddings.tokenizer.sep_token) + " "
                else:
                    self.sep = " [SEP] "

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def _get_data_points_from_sentence(self, sentence: TextTriple) -> List[TextTriple]:
        return [sentence]

    def _get_embedding_for_data_point(self, prediction_data_point: TextTriple) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        if self.embed_separately:
            self.embeddings.embed(
                [prediction_data_point.first, prediction_data_point.second, prediction_data_point.third]
            )
            return torch.cat(
                [
                    prediction_data_point.first.get_embedding(embedding_names),
                    prediction_data_point.second.get_embedding(embedding_names),
                    prediction_data_point.third.get_embedding(embedding_names),
                ],
                0,
            )
        else:
            concatenated_sentence = Sentence(
                prediction_data_point.first.to_tokenized_string()
                + self.sep
                + prediction_data_point.second.to_tokenized_string()
                + self.sep
                + prediction_data_point.third.to_tokenized_string(),
                use_tokenizer=False,
            )
            self.embeddings.embed(concatenated_sentence)
            return concatenated_sentence.get_embedding(embedding_names)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "embed_separately": self.embed_separately,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("document_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            embed_separately=state.get("embed_separately"),
            **kwargs,
        )

    def get_used_tokens(self, corpus: Corpus) -> typing.Iterable[List[str]]:
        for sentence_triple in _iter_dataset(corpus.get_all_sentences()):
            yield [t.text for t in sentence_triple.first]
            yield [t.text for t in sentence_triple.second]
            yield [t.text for t in sentence_triple.third]
