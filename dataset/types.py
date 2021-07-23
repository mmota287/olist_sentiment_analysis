from typing import Callable, Tuple
import pandas as pd
import tensorflow as tf
import numpy as np

ClearDataFnType = Callable[[pd.DataFrame], pd.DataFrame]
SplitDatasetFnType = Callable[[tf.data.Dataset], Tuple[tf.data.Dataset, tf.data.Dataset]]
TransformToSeqFnType = Callable[[np.array], tf.data.Dataset]
