from generate_pred_data import get_embeddings, embeddings_to_dataframe, return_final_df
from bin.descriptors.sequence_descriptors import process_data as get_descriptors, process_data_pool as get_descriptors_pool
from pathlib import Path

from Bio import SeqIO 

import numpy as np



paths = Path("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/shuffled_fastas").glob("*.fa*")

DEVICE = "cuda"


def main():

    for fasta in paths:

        category = fasta.stem

        print(f"Processing {category} ... ")

        records = list(SeqIO.parse(fasta, "fasta"))

        records.sort(key = lambda x: len(x.seq), reverse=False)

        seq_dict = { record.id : str(record.seq) for record in records }

        print("     Computing sequence descriptors ... : ")
        seq_descriptors = get_descriptors_pool(records = records, category = category, num_workers=40)

        print("     Computing embeddings ... ")
        seq_embeddings = get_embeddings(device = DEVICE, seqs = seq_dict, per_protein=True, per_residue=True, sec_struct = False, max_batch=300)
        seq_embeddings = embeddings_to_dataframe(seq_embeddings)

        final = return_final_df(seq_descriptors, seq_embeddings, seq_dict)

        print("     Writing to parquet ... ")

        final.write_parquet(f"../shuffled_data/{category}.parquet", compression="lz4")

        print(f"Written to ../shuffled_data/{category}.parquet")
        print(f"Done processing {category}, dataset is {final.shape[0]} entries long. ")


if __name__ == "__main__": 


    main()