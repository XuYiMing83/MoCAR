import os
import pickle
import zlib

from torch_geometric.data import Dataset, HeteroData


class AVDataset(Dataset):
    """Dataset for compressed AV2 HeteroData samples.

    Expected layout:
        root/processed/train.dat
        root/processed/val.dat
        root/processed/test.dat

    For compatibility with older local preprocessing outputs, this also accepts:
        root/processed/train/*.pkl
        root/processed/val/*.pkl
        root/processed/test/*.pkl
    """

    def __init__(self, root, processed="MoCAR_data", split="train", transform=None):
        super().__init__(root=root, transform=transform)
        self.root = root
        self.processed = processed
        self.split = split
        self._dat_path = os.path.join(root, processed, f"{split}.dat")
        self._processed_dir = os.path.join(root, processed, split)
        if os.path.isfile(self._dat_path):
            with open(self._dat_path, "rb") as f:
                self._items = pickle.load(f)
            self._processed_file_names = []
        else:
            self._items = None
            self._processed_file_names = sorted(
                name for name in os.listdir(self._processed_dir) if name.endswith(".pkl")
            )

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def len(self):
        if self._items is not None:
            return len(self._items)
        return len(self._processed_file_names)

    def get(self, idx):
        if self._items is not None:
            return HeteroData(pickle.loads(zlib.decompress(self._items[idx])))
        with open(os.path.join(self._processed_dir, self._processed_file_names[idx]), "rb") as f:
            return HeteroData(pickle.loads(zlib.decompress(pickle.load(f))))
