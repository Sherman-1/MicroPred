from Bio import SeqIO 
import polars as pl 
from sklearn.model_selection import train_test_split
from bin.descriptors.sequence_descriptors import process_data as get_descriptors
print("Importing embeddings dependencies")
from generate_training_embeddings import get_embeddings
print("Done ")
import torch
import warnings



CLASSES = [
    
    "molten",
    "globular",
    "bitopic",
    "polytopic",
    "disprot"
]

CLASS_TO_INT = dict(zip(CLASSES, range(len(CLASSES)))) 


def results_to_dataframe(results: dict) -> pl.DataFrame:
    """
    converts dict returned by get_embeddings to a Polars DF
    suited for parquet storage
    """

    protein_ids = set(results.get("protein_embs", {}).keys())
    residue_ids = set(results.get("residue_embs", {}).keys())

    assert len(protein_ids) == len(residue_ids), "IDs don't match between per-residue and per-protein embeddings"

    all_ids = protein_ids.union(residue_ids)

    rows = []
    for identifier in all_ids:

        protein_array = results.get("protein_embs", {}).get(identifier)
        protein_list = protein_array.tolist() if protein_array is not None else None

        # Get the residue embeddings (2D) if they exist, converting to a list of lists
        residue_array = results.get("residue_embs", {}).get(identifier)
        residue_list = residue_array.tolist() if residue_array is not None else None

        rows.append({
            "id": identifier,
            "protein_emb": protein_list,
            "residue_emb": residue_list
        })

    return pl.DataFrame(rows)



fastas = {

    "bitopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/bitopic_representatives.fasta",
    "polytopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/polytopic_representatives.fasta",
    "disprot"  : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/disprot_representatives.fasta", 
    "molten" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/Small_full.fasta",
    "globular" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/S3_full_subset.fasta"

}


if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = "cuda"
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    warnings.warn("Warning: Embeddings inference running on CPU!", RuntimeWarning)

final_df = None

for category, fasta in fastas.items(): 

    records = list(SeqIO.parse(fasta, "fasta"))

    seq_dict = { record.id : str(record.seq) for record in records }

    print("Computing sequence descriptors ... : ")

    seq_descriptors = get_descriptors(records = records, category = CLASS_TO_INT[category])
    print("Computing embeddings ... ")
    seq_embeddings = get_embeddings(device = device, seqs = seq_dict, per_protein=True, per_residue=True, sec_struct = False, max_batch=300)
    seq_embeddings = results_to_dataframe(seq_embeddings)

    spaced_seqs = { id : " ".join(list(seq)) for id, seq in seq_dict.items() }

    seq_descriptors = (
        seq_descriptors
        .with_columns(
            descriptors = pl.concat_list(pl.exclude(["id","category"])),
            descriptor_names = seq_descriptors.select(pl.exclude(["id","category"])).columns
        )
        .select(["id","descriptors","category","descriptor_names"])
        .join(seq_embeddings, on = "id", how = "inner")
        .join(
            pl.DataFrame({
                "id" : spaced_seqs.keys(), "sequence" : spaced_seqs.values()
            }), on = "id", how = "inner"
        )
    )

    if final_df is None:
        final_df = seq_descriptors
    else:
        final_df = final_df.vstack(seq_descriptors)