#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Iterator, Union

import torch
import numpy as np
from torch.utils.data import Sampler


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
    


class ProportionalBatchSampler(Sampler):
    def __init__(self, episode_data_index, batch_size, my_episode_max=50, my_fraction=0.5):
        """
        episode_data_index: dict with 'from', 'to' keys listing per-episode frame ranges
        batch_size: int, total batch size
        my_episode_max: int, episodes [0, my_episode_max) are "my task"
        my_fraction: float, fraction of each batch from "my task"
        """
        # Create lists of indices per group
        self.my_indices = []
        self.other_indices = []
        for episode_idx, (start, end) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            indices = list(range(start.item(), end.item()))
            if episode_idx < my_episode_max:
                self.my_indices.extend(indices)
            else:
                self.other_indices.extend(indices)
        self.batch_size = batch_size
        self.my_batch_size = int(batch_size * my_fraction)
        self.other_batch_size = batch_size - self.my_batch_size

    def __iter__(self):
        # Shuffle at epoch start
        my_idx = np.random.permutation(self.my_indices)
        other_idx = np.random.permutation(self.other_indices)
        my_ptr = 0
        other_ptr = 0

        while my_ptr + self.my_batch_size <= len(my_idx) and other_ptr + self.other_batch_size <= len(other_idx):
            my_batch = my_idx[my_ptr:my_ptr + self.my_batch_size]
            other_batch = other_idx[other_ptr:other_ptr + self.other_batch_size]
            batch = np.concatenate([my_batch, other_batch])
            np.random.shuffle(batch)
            yield batch.tolist()
            my_ptr += self.my_batch_size
            other_ptr += self.other_batch_size

    def __len__(self):
        return min(
            len(self.my_indices) // self.my_batch_size,
            len(self.other_indices) // self.other_batch_size
        )
