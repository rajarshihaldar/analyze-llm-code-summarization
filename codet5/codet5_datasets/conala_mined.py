"""Dataset for loading raw JSON data for the automaticallymined data CoNaLa,
the Code/Natural Language Challenge.
"""

import json
import os

import numpy as np

from codet5_datasets import BaseDataset


class CoNaLaMined(BaseDataset):
    """CoNaLa dataset."""

    def _load(self, examples=-1):
        """Load a dataset into memory."""

        fname = 'conala-mined.jsonl'
        data = []
        count = 0
        with open(os.path.join(self.root, fname)) as f:
            for line in f:
                x = json.loads(line)
                if len(x['snippet']) > 0 and len(x['snippet']) > 0:
                    data.append((x['snippet'], x['intent']))
                    count += 1
                if count == examples:
                    break

        self.data = np.array(data, dtype=np.object)
