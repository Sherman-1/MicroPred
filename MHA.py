
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Nov 29 16:53:41 2024

import os

import numpy as np
import torch
import torch.nn as nn



class ScaledDotProductAttention(nn.Module):
    """
    See: https://einops.rocks/pytorch-examples.html
    """

    def __init__(self, temperature, attn_dropout=0.1, activation=nn.Softmax(dim=2)):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.activation = activation

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttentionBouGui(nn.Module):

    """
    See: https://einops.rocks/pytorch-examples.html
    
    >>> bs = 4
    >>> d_model = 128
    >>> d_k = 16
    >>> d_v = 32
    >>> nq = 100
    >>> nv = 78
    >>> nhead = 8
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v)
    >>> q = torch.rand(bs, nq, d_model)
    >>> v = torch.rand(bs, nv, d_model)
    >>> k = torch.clone(v)
    >>> out, attn =  mha(q, k, v)
    >>> out.shape
    torch.Size([4, 100, 128])
    >>> attn.shape
    torch.Size([4, 100, 78])

    To compare when key_padding_mask is not None
    Put infinite values in k:
    >>> k[bs-1, nv-1] = torch.inf
    >>> k[bs-2, nv-2] = torch.inf
    >>> out, attn =  mha(q, k, v)

    Consequently the output contain nan at those positions
    >>> torch.isnan(out[bs-1, nv-1]).all()
    tensor(True)
    >>> torch.isnan(out[bs-2, nv-2]).all()
    tensor(True)

    and not at the other:
    >>> torch.isnan(out[bs-3, nv-3]).any()
    tensor(False)

    As well as the attention matrix (attn)
    >>> torch.isnan(attn[bs-1, :, nv-1]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-2, :, nv-2]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-3, :, nv-3]).any()
    tensor(False)

    Define a mask
    >>> key_padding_mask = torch.zeros(bs, nv, dtype=bool)
    >>> key_padding_mask[bs-1, nv-1] = True
    >>> key_padding_mask[bs-2, nv-2] = True
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> out.shape
    torch.Size([4, 100, 128])

    The output doesn't contain nan anymore as the infinite values are masked:
    >>> torch.isnan(out[bs-1, nv-1]).any()
    tensor(False)
    >>> torch.isnan(out[bs-2, nv-2]).any()
    tensor(False)

    The attn matrix contain 0 at the masked positions
    >>> attn.shape
    torch.Size([4, 100, 78])
    >>> (attn[bs-1, :, nv-1] == 0).all()
    tensor(True)
    >>> (attn[bs-2, :, nv-2] == 0).all()
    tensor(True)

    The attn matrix is softmaxed
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(True)

    The user can define another activation function (or identity)
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0, attn_activation=lambda x: x, attn_temperature=1.0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(False)

    User defined output dimension d_out:
    >>> q.shape
    torch.Size([4, 100, 128])
    >>> k.shape
    torch.Size([4, 78, 128])
    >>> v.shape
    torch.Size([4, 78, 128])
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, d_out=64)
    >>> out, attn = mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> out.shape, attn.shape
    (torch.Size([4, 100, 64]), torch.Size([4, 100, 78]))
    """

    def __init__(self, n_head:int, d_model:int, 
                 d_k:int=None, d_v:int=None, d_out:int=None, dropout:float=0.1, attn_dropout:float=0.1, 
                 attn_activation:callable=nn.Softmax(dim=2), attn_temperature=None, skip_connection=True):
        super().__init__()

        # Essaie 131224
        if not d_k : d_k = d_model
        if not d_v : d_v = d_model 

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.skip_connection = skip_connection

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if attn_temperature is None:
            attn_temperature = np.power(d_k, 0.5)
        self.attention = ScaledDotProductAttention(temperature=attn_temperature, attn_dropout=attn_dropout, activation=attn_activation)

        if d_out is None:
            d_out = d_model
        elif d_out != d_model:
            self.skip_connection = False
        self.layer_norm = nn.LayerNorm(d_out)
        self.fc = nn.Linear(n_head * d_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, key_padding_mask=None, average_attn_weights=True):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Order for n_head and batch size: (n_head, sz_b, ...)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        if key_padding_mask is not None:  #(sz_b, len_k)
            key_padding_mask = torch.stack([key_padding_mask,]*n_head, dim=0).reshape(sz_b*n_head, 1, len_k) * torch.ones(sz_b*n_head, len_q, 1, dtype=torch.bool, device = q.device)
            if mask is not None:
                mask = mask + key_padding_mask
            else:
                mask = key_padding_mask

        output, attn = self.attention(q, k, v, mask=mask)
        # >>> output.shape
        # torch.Size([sz_b, len_q, d_v])
        # >>> attn.shape
        # torch.Size([sz_b*n_head, len_q, len_k])
        if average_attn_weights:
            attn = attn.view(n_head, sz_b, len_q, len_k)
            attn = attn.mean(dim=0)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        if self.skip_connection:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
        return output, attn

class MHAP(nn.Module):
    """
    MultiheadAttention Pooling (mhap)
    >>> nres = 35
    >>> ne = 1024
    >>> sb = 5
    >>> prottrans_embeddings = torch.rand(sb, nres, ne)
    >>> prottrans_embeddings.shape
    torch.Size([5, 35, 1024])
    >>> mhap = MHAP(embed_dim=ne)

    >>> attn_output, attn_output_weights = mhap(prottrans_embeddings)
    >>> attn_output.shape
    torch.Size([5, 1024])

    batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> attn_output, attn_output_weights = mhap(batch)
    >>> attn_output.shape
    torch.Size([64, 1024])

    d_out:
    >>> mhap = MHAP(embed_dim=1024, d_out=128)
    >>> attn_output, attn_output_weights = mhap(batch)
    >>> attn_output.shape
    torch.Size([64, 128])
    """
    def __init__(self, embed_dim:int, d_out:int=None, n_head:int=8, dropout:float=0.1, 
                 attn_dropout:float = 0.1):
        super(MHAP, self).__init__()

        if d_out is None: 
            d_out = embed_dim

        self.attention = MultiHeadAttentionBouGui(d_model=embed_dim, 
                                               n_head=n_head,  
                                               dropout=dropout, 
                                               attn_dropout=attn_dropout,
                                               d_out = d_out,
                                               d_v = d_out,
                                               d_k = d_out)
                                               
    def forward(self, embeddings, masks = None):
        
        Qi = embeddings.mean(axis=-2, keepdim=True) # Suppress mean pool for self attention
        attn_output, attn_output_weights = self.attention(q=Qi, k=embeddings, v=embeddings, key_padding_mask = masks)
        return torch.atleast_2d(torch.squeeze(attn_output)), attn_output_weights

class MLP(nn.Module):

    def __init__(self, input_shape, output_shape, dropout_rate=0.2):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            
            nn.Linear(input_shape, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(16, output_shape),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.model(x)

class MHAP_MLP(nn.Module):
    """
    forward batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> mhap_mlp = MHAP_MLP(input_embed_dim = 1024, output_embed_dim = 128, num_classes = 5)
    >>> mlp_output = mhap_mlp.forward(batch, mask = None)
    >>> mlp_output.shape
    torch.Size([64, 5])
    """
    def __init__(self, input_embed_dim: int, num_classes: int, output_embed_dim:int=None):
        super(MHAP_MLP, self).__init__()
        self.mhap = MHAP(embed_dim=input_embed_dim, d_out = output_embed_dim)
        self.mlp = MLP(input_shape=output_embed_dim, output_shape=num_classes)

    def forward(self, embeddings, mask):
        attn_output, _ = self.mhap(embeddings, mask)  
        mlp_output = self.mlp(attn_output)             
        return mlp_output

class ProtTransDataset(Dataset):

    """
    >>> ds = ProtTransDataset(ptfile="datasets/test_data.pt")

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
    
    def __init__(self, ptfile, num_classes = 5, return_one_hot = True):
        
        self.data = torch.load(f=ptfile, weights_only=False)
        self.names = list(self.data.keys())
        self.targets = [self.data[name]["class_type"] for name in self.names]
        self.num_classes = num_classes
        self.return_one_hot = return_one_hot


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        name = self.names[idx]
        embedding = self.data[name]["embedding"]
        class_type = self.data[name]["class_type"]

        class_id = CLASS_TO_INT[class_type]
        
        if self.return_one_hot:
            one_hot = torch.nn.functional.one_hot(torch.tensor(class_id), self.num_classes).float()
            return embedding, one_hot, name
        else:
            return embedding, torch.tensor(class_id), name

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
    names = [e[2] for e in batch]

    maxlen = max([len(e) for e in embeddings])
    mask = []
    
    for e in embeddings:
        p = torch.zeros(len(e), dtype = bool)
        p = F.pad(p, (0, maxlen-len(e)), value = True)
        mask.append(p)
    mask = torch.stack(mask)

    embeddings = torch.stack([F.pad(e, (0, 0, 0, maxlen-len(e))) for e in embeddings], dim=0)

    return embeddings, class_id, mask, names

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
                f = getattr(sys.modules[__name__], f)   

                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )

        sys.exit()
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    epochs = 10
    lr = 0.001
    batch_size = 512

    # def __init__(self, embed_dim:int, d_out:int=None, n_head:int=8, dropout:float=0.1, 
    #            attn_dropout:float = 0.1):

    data = ProtTransDataset(ptfile="datasets/train_data.pt")
    dataloader = DataLoader(data, collate_fn=padding_collate, batch_size=batch_size, shuffle=True)
    model = MHAP_MLP(input_embed_dim=1024, output_embed_dim=128, num_classes=5).to(device)

    data_val = ProtTransDataset(ptfile="datasets/test_data.pt")
    dataloader_val = DataLoader(data_val, collate_fn=padding_collate, batch_size=batch_size, shuffle=True)


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    losses = list()
    val_losses = list()
    val_iter = iter(dataloader_val)

    for epoch in tqdm(range(epochs), desc="Epochs"):

        loss_tmp = list()
        for i, batch in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):

            model.train()    
            embeddings, class_id, mask, ids = batch

            embeddings = embeddings.to(device)
            mask = mask.to(device)
            class_id = class_id.to(device)
            
            optimizer.zero_grad()

            output = model(embeddings, mask)
            loss = criterion(output, class_id)
            loss.backward()
            
            optimizer.step()

            with torch.no_grad():

                model.eval()
                try:
                    batch_val = next(val_iter)

                # Go through iter once
                except StopIteration:

                    val_iter = iter(dataloader_val)
                    batch_val = next(val_iter)

                embed_val, class_id_val, mask_val, ids_val = batch_val
                embed_val = embed_val.to(device)
                class_id_val = class_id_val.to(device)
                mask_val = mask_val.to(device)
                output_val = model(embed_val, mask_val)
                loss_val = criterion(output_val, class_id_val)
                val_losses.append(loss_val.item())

                print(f"{epoch=}")
                print(f"step={i}")
                print(f"{loss=:.4g}")
                print(f"{loss_val=:.4g}")
                print("--")

            losses.append(loss.item())

               
        
        torch.save(model.state_dict(), f"models/101224_MHAP_MLP_BCE_reduce_embeds_epoch_{epoch}.pt")


    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(val_losses, label = "Validation loss")
    plt.plot(losses, label = "Loss")
    plt.legend()
    plt.savefig("101224_MHAP_MLP_BCE_reduce_embeds_losses.png")


    
