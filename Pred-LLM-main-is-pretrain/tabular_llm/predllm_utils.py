import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


# def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
#     """ Converts the sentences back to tabular data

#     Args:
#         text: List of the tabular data in text form
#         df_gen: Pandas DataFrame where the tabular data is appended

#     Returns:
#         Pandas DataFrame with the tabular data from the text appended
#     """
#     columns = df_gen.columns.to_list()
        
#     # Convert text to tabular data
#     for t in text:
#         features = t.split(",")
#         td = dict.fromkeys(columns)
#         print(f'td:{td}')
        
#         # Transform all features back to tabular data
#         for f in features:
#             values = f.strip().split(" ")
#             if values[0] in columns and not td[values[0]]:
#                 try:
#                     td[values[0]] = [values[1]]
#                 except IndexError:
#                     #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
#                     pass
                
#         df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
#     return df_gen

def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    
    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = {col: None for col in columns}  # Khởi tạo với None để xử lý cột thiếu
        
        # Transform all features back to tabular data
        seen_keys = set()
        for f in features:
            values = f.strip().split(" ", 1)  # Chia làm 2 phần: key và value
            if len(values) == 2 and values[0] in columns and values[0] not in seen_keys:
                td[values[0]] = values[1]
                seen_keys.add(values[0])  # Đảm bảo mỗi key chỉ được sử dụng một lần
        
        # Thêm dữ liệu vào DataFrame
        df_gen = pd.concat([df_gen, pd.DataFrame([td])], ignore_index=True, axis=0)
    
    return df_gen


def _encode_row_partial(row, shuffle=True):
    """Function that takes a row and converts all columns into the text representation that are not NaN."""
    num_cols = len(row.index)
    if not shuffle:
        idx_list = np.arange(num_cols)
    else:
        idx_list = np.random.permutation(num_cols)

    lists = ", ".join(
        sum(
            [
                [f"{row.index[i]} {row[row.index[i]]}"]
                if not pd.isna(row[row.index[i]])
                else []
                for i in idx_list
            ],
            [],
        )
    )
    return lists

