import random
import typing as tp
from .predllm_utils import _get_string
import pandas as pd
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
# from _get_string
import numpy as np


class PredLLMDataset(Dataset):
    def set_args(self, numerical_features):
        self.numerical_features = numerical_features
    
    def get_ds_size(self, len):
        self.len = len

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        if key < self.len:
            # permute only feature variables
            all_column_idx = list(range(row.num_columns))
            shuffle_idx = all_column_idx[:-1]
            random.shuffle(shuffle_idx)
            # keep target variable at the end
            target_idx = all_column_idx[-1]

            shuffled_text = ", ".join(
                [_get_string(self.numerical_features,
                             row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                 for i in shuffle_idx]
            )
            shuffled_text += ", {} {}".format(row.column_names[target_idx],
                                                 str(row.columns[target_idx].to_pylist()[0]).strip())
        else:
            # fix both feature and target variables
            shuffle_idx = list(range(row.num_columns))

            shuffled_text = ", ".join(
                [_get_string(self.numerical_features,
                             row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                 for i in shuffle_idx]
            )
        # print("shuffled_text: {}".format(shuffled_text))
        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


### TODO
class MyDataset(Dataset):
    def __init__(self, tokenizer):
        self.mydata = []
        self.length = 0
        self.idx = 0
        self.reverse_idx = {}
        self.subtractor = {}
        self.tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def add_dataframe(self, df: pd.DataFrame):
        self.subtractor[self.idx] = self.length
        for i in range(self.length, self.length + df.shape[0]):
            self.reverse_idx[i] = self.idx
        self.length += df.shape[0]
        self.idx += 1
        pred_ds = PredLLMDataset.from_pandas(df)
        pred_ds.set_args(numerical_features=numerical_features)
        pred_ds.get_ds_size(len(df))
        pred_ds.set_tokenizer(self.tokenizer)
        self.mydata.append(pred_ds)

    def __len__(self):
        return self.length

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        idx = self.reverse_idx[key]
        tokenized_text = self.mydata[idx]._getitem(key - self.subtractor[idx])
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


###


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
