from Bio import SeqIO 
import polars as pl 
from sklearn.model_selection import train_test_split


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


fastas = {

    "bitopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/bitopic_representatives.fasta",
    "polytopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/polytopic_representatives.fasta",
    "disprot"  : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/disprot_representatives.fasta", 
    "molten" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/Small_full.fasta",
    "globular" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas/S3_full_subset.fasta"

}

dfs = list()

for category, fasta in fastas.items(): 

    seqs = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))

    seqs = { key : str(seq.seq) for key,seq in seqs.items() }

    dfs.append(pl.DataFrame({
        "sequence" : seqs.values(),
        "id" : seqs.keys(),
        "category" : [category] * len(seqs)
    }))


df = pl.concat(dfs).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String),
    category = pl.col("category").replace_strict(CLASS_TO_INT)
)

df_train, df_test = train_test_split(df, test_size = 0.4, stratify=df[["category"]])


df_train.write_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train_sequences.csv")
df_test.write_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/test_sequences")

if __name__ == "__main__": 

    pass

