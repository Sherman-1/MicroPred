
import random
import numpy as np 
import polars as pl 

from datasets import Dataset

from transformers import T5Tokenizer

data = pl.scan_csv("../splits/three_vs_rest.csv").with_columns(pl.when(pl.col("target") >= 1).then(pl.lit(1)).otherwise(pl.lit(0)).alias("label"))

train_data = data.filter(pl.col("set") == "train").select(["sequence","label"]).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String)
).collect()

test_data = data.filter(pl.col("set") == "test").select(["sequence","label"]).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String)
).collect()

def create_dataset(tokenizer,seqs,labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset


tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

train_dataset = create_dataset(
    tokenizer,
    train_data[["sequence"]].to_series().to_list(),
    train_data[["label"]].to_series().to_list()
)

test_dataset = create_dataset(
    tokenizer,
    test_data[["sequence"]].to_series().to_list(),
    test_data[["label"]].to_series().to_list()
)
