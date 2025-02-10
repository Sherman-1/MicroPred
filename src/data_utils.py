from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight

import numpy as np

from pathlib import Path

CLASSES = [
    
    "molten",
    "globular",
    "bitopic",
    "polytopic",
    "disprot",
    "iORFs",
    "random_uniprot",
    "scer_20_100",
    "PFAL_CDS_20_100",
    "ATHA_CDS_20_100",
    "MMUS_CDS_20_100",
    "HSAP_CDS_20_100",
    "DMEL_CDS_20_100",
    "CELE_CDS_20_100",
    "OSAT_CDS_20_100",
    "TREMBL_MICROPROTEINS"
]

CLASS_TO_INT = dict(zip(CLASSES, range(len(CLASSES)))) 
INT_TO_CLASS = dict(zip(range(len(CLASSES)), CLASSES))


def create_dataset_from_sequences(tokenizer, seqs, labels, max_seq_length=1024):
    tokenized = tokenizer(seqs, max_length=max_seq_length, padding="max_length", truncation=True)
    tokenized["labels"] = labels
    return HFDataset.from_dict(tokenized)


class SeqProtT5Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset  

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        item = {key: torch.tensor(val) for key, val in sample.items()}
        item["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
        return item
    
class EmbedProtT5Dataset(Dataset):

    """
    >>> ds = EmbedProtT5Dataset(ptfile="datasets/test_data.pt")

    single
    >>> embed, class_id, name = ds[0]
    >>> embed.shape # Shortest sequence of the test set is 23
    torch.Size([23, 1024])
    >>> class_id.shape
    torch.Size([5])
    >>> name
    'EPGN_HUMAN_A_1_elong_first'

    batch
    >>> dl = DataLoader(ds, batch_size = 3, collate_fn = padding_collate)
    >>> first_batch = next(iter(dl))
    >>> embeds, class_ids, masks, names = first_batch
    >>> embeds.shape # 84 Longest seq of the first batch ( shortest sequences of the set )
    torch.Size([3, 84, 1024])
    >>> class_ids.shape # Batch of 3, 5 classes in the set
    torch.Size([3, 5])
    >>> len(names)
    3
    >>> masks.shape # 3 masks, each mask is relative to the length of the longest seq of the batch
    torch.Size([3, 84])
    """
    
    def __init__(self, ptfile, num_classes):
        
        self.data = torch.load(f=ptfile, weights_only=False)
        self.names = list(self.data.keys())
        self.num_classes = num_classes

        targets = [self.data[name]["class_type"] for name in self.names]

        if not all(isinstance(target, type(targets[0])) for target in targets):
            raise ValueError("Targets contain mixed datatypes.")

        if type(targets[0]) == str:
            self.targets = [ CLASS_TO_INT[t] for t in targets ]
        elif type(targets[0]) == int:
            self.targets = targets
        else:
            print("Wtf")

        assert num_classes == len(set(self.targets)), "Error : more classes in dataset than given in num_classes"

        self.class_weights = torch.tensor(self._get_class_weights(), dtype=torch.float32)

        assert len(self.targets) == len(self.names) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.names[idx]
        embedding = self.data[name]["embedding"]
        class_type = self.data[name]["class_type"]

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        if not isinstance(class_type, int):
            target = CLASS_TO_INT[class_type]
        else:
            target = class_type

        target = torch.tensor(target, dtype=torch.long)  
        one_hot_target = torch.nn.functional.one_hot(target, self.num_classes).float()

        return embedding, target, one_hot_target, name
    
    def _get_class_weights(self):

        return compute_class_weight(y = self.targets, 
                                    class_weight = "balanced", 
                                    classes = np.sort(np.unique(self.targets)) # Sort to match CLASS_TO_INT
                                )
        
def padding_collate(batch):
    """
    >>> batch = [(torch.rand(30, 1024), torch.tensor([1,0,0]), "oui"), (torch.rand(56, 1024), torch.tensor([0,0,1]), "non")]
    >>> [e[0].shape for e in batch]
    [torch.Size([30, 1024]), torch.Size([56, 1024])]
    >>> embeddings, class_id, mask, names = padding_collate(batch)
    >>> embeddings.shape
    torch.Size([2, 56, 1024])
    >>> class_id.shape
    torch.Size([2, 3])
    >>> mask.shape 
    torch.Size([2, 56])
    >>> mask.sum() == 26
    tensor(True)
    >>> names
    ['oui', 'non']
    """

    embeddings = [e[0] for e in batch]
    class_id = torch.stack([e[1] for e in batch])
    one_hot = torch.stack([e[2] for e in batch])
    names = [e[3] for e in batch]

    maxlen = max([len(e) for e in embeddings])
    mask = []
    
    for e in embeddings:
        p = torch.zeros(len(e), dtype = bool)
        p = F.pad(p, (0, maxlen-len(e)), value = True)
        mask.append(p)
    mask = torch.stack(mask)

    embeddings = torch.stack([F.pad(e, (0, 0, 0, maxlen-len(e))) for e in embeddings], dim=0)

    return embeddings, class_id, one_hot, mask, names

def get_dataloaders(base_train_path = "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset", 
                    base_val_path = "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset", 
                    num_classes = 5, 
                    batch_size_train=512, 
                    batch_size_val=512,
                    collate_fn = None,
                    embed_type = None):


    """
    >>> import io
    >>> fake_data = {
    ...     "EPGN_HUMAN_A_1": {"embedding": torch.rand(23, 1024), "class_type": "bitopic"},
    ...     "XYZ_HUMAN_B_2": {"embedding": torch.rand(50, 1024), "class_type": "polytopic"},
    ...     "LMN_MOUSE_C_3": {"embedding": torch.rand(64, 1024), "class_type": "molten"},
    ...     "LMN_MOUSE_C_4": {"embedding": torch.rand(44, 1024), "class_type": "globular"},
    ...     "LMN_MOUSE_C_5": {"embedding": torch.rand(99, 1024), "class_type": "disprot"},
    ...     "LMN_MOUSE_C_6": {"embedding": torch.rand(100, 1024), "class_type": "disprot"}
    ... }
    >>> fake_data = {k: v for k, v in sorted(fake_data.items(), key=lambda item: item[1]["embedding"].shape[0])}
    >>> byte_stream = io.BytesIO()
    >>> torch.save(fake_data, byte_stream)
    >>> _ = byte_stream.seek(0)
    >>> ds = EmbedProtT5Dataset(byte_stream, num_classes=5)
    >>> embed, class_id, one_hot, name = ds[0]
    >>> embed.shape
    torch.Size([23, 1024])
    >>> class_id
    tensor(2)
    >>> name
    'EPGN_HUMAN_A_1'

    >>> dl = DataLoader(ds, batch_size = 2, collate_fn = padding_collate)
    >>> embeds, class_id, one_hot, mask, name = next(iter(dl))
    >>> one_hot.shape # 5 classes, 2 points
    torch.Size([2, 5])
    >>> mask.shape
    torch.Size([2, 44])
    """



    assert embed_type in ["residue","protein"], "Per protein or per residue embedding needed as input"

    print(f" Preparing data ... ")

    train_path = Path(base_train_path) / f"trainset_{embed_type}_embeddings.pt"
    val_path = Path(base_train_path) / f"testset_{embed_type}_embeddings.pt"

    if not (train_path.exists() and val_path.exists()):

        raise FileNotFoundError("Training or Val dataset does not exist")


    print("         Loading datasets ")

    train_ds = EmbedProtT5Dataset(ptfile=train_path, num_classes=num_classes)
    val_ds = EmbedProtT5Dataset(ptfile=val_path, num_classes=num_classes)

    print("         Defining dataloaders")

    train_dl = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, collate_fn = collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, collate_fn = collate_fn)

    try: 
        next(iter(train_dl))

    except Exception as e:
        print(f"Error while defining DataLoader :")
        print(e)
    return train_dl, val_dl, train_ds.class_weights


if __name__ == "__main__":

    import argparse
    import doctest
    import sys

    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                func = getattr(sys.modules[__name__], f)   

                doctest.run_docstring_examples(
                    func,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )

        sys.exit()