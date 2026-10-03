import os
import pickle
import zlib

from torch_geometric.data import Dataset, HeteroData


class AVDataset(Dataset):
    """Dataset for compressed AV2 HeteroData lists.

    Expected layout:
        root/processed/train.dat
        root/processed/val.dat
        root/processed/test.dat

    Each .dat file contains a list of zlib-compressed pickled dictionaries.
    """

    def __init__(self, root, processed="MoCAR_data", split="train", transform=None):
        super().__init__(root=root, transform=transform)
        self.root = root
        self.processed = processed
        self.split = split

        if split == "dex":
            self.ex_list = []
            for part in ["train", "val", "test"]:
                self.ex_list.extend(self._load_split(part))
        else:
            self.ex_list = self._load_split(split)

    def _load_split(self, split):
        path = os.path.join(self.root, self.processed, f"{split}.dat")
        with open(path, "rb") as f:
            return pickle.load(f)

    def len(self):
        return len(self.ex_list)

    def get(self, idx):
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return HeteroData(instance)
