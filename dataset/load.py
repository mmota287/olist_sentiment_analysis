"""Data Loader"""

from dataset.types import ClearDataFnType
from numpy.core.records import array
import pandas as pd
from toolz import curry
pd.options.mode.chained_assignment = None  # default='warn'


def load_from_file(path: str, clear_fn: ClearDataFnType) -> pd.DataFrame:
    """Load Dataset from file

    Args:
        path (str): File path
        clear_fn (ClearDataFnType): Clear function

    Returns:
        pd.DataFrame: Dataframe loaded
    """
    df = pd.read_csv(path)

    if clear_fn is None:
        return df

    return clear_fn(df)

@curry
def clear_data_fn(df: pd.DataFrame, column_text: str, column_label: str,
        bins: array = [0, 2, 5], classes: array = [0,1]) -> pd.DataFrame:
    """
    Clear dataframe loaded

    Args:
        df (pd.DataFrame): Dataframe loaded
        column_text (str): Column text
        column_label (str): Column label
        bins (array, optional): Column score. Defaults to [].
        classes (array, optional): Sentiments classe. Defaults to [0,1].

    Returns:
        pd.DataFrame: Dataframe cleaned
    """
    df = df.dropna(subset=[column_text])
    df['label'] = pd.cut(df[column_label], bins=bins, labels=classes)
    df = df.rename(columns={column_text: 'text'})
    df = df[['text','label']]
    return df