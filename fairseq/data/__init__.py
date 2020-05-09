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
from .id_dataset import IdDataset
from .monolingual_dataset import MonolingualDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .numel_dataset import NumelDataset
from .pad_dataset import LeftPadDataset, PadDataset, RightPadDataset
from .prepend_token_dataset import PrependTokenDataset
from .strip_token_dataset import StripTokenDataset
from .transform_eos_dataset import TransformEosDataset
from .truncate_dataset import TruncateDataset
from .token_block_dataset import TokenBlockDataset
from .lm_context_window_dataset import LMContextWindowDataset

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
    'IdDataset',
    'MonolingualDataset',
    'NestedDictionaryDataset',
    'NumelDataset',
    'LeftPadDataset',
    'PadDatase',
    'RightPadDataset',
    'NestedDictionaryDataset',
    'PrependTokenDataset',
    'TransformEosDataset',
    'TokenBlockDataset',
    'LMContextWindowDataset'
]
