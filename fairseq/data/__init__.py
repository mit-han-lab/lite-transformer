# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .concat_dataset import ConcatDataset
from .language_pair_dataset import LanguagePairDataset
from .append_token_dataset import AppendTokenDataset
from .strip_token_dataset import StripTokenDataset
from .truncate_dataset import TruncateDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'BaseWrapperDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'StripTokenDataset',
    'TruncateDataset',
    'AppendTokenDataset'
]
