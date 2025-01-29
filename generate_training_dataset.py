from Bio import SeqIO 
import polars as pl 

fastas = {

    "bitopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/bitopic_representatives.fasta",
    "polytopic" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/polytopic_representatives.fasta",
    "disprot"  : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/disprot_representatives.fasta", 
    "molten" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/Small_full.fasta",
    "globular" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/S3_full_subset.fasta"

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

data = pl.concat(dfs).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String)
)




