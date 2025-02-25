from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import re
import torch

from glob import glob

from Bio import SeqIO

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

pipeline = TokenClassificationPipeline(
    model=AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert_bfd_ss3"),
    tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_ss3", skip_special_tokens=True),
    device=device
)

def get_ss(records):

    """
    One-letter secondary structure code is nearly the same as used  in
	DSSP [2] (see Frishman and Argos [1] for details):

	   H	    Alpha helix
	   G	    3-10 helix
	   I	    PI-helix
	   E	    Extended conformation
	   B or	b   Isolated bridge
	   T	    Turn
	   C	    Coil (none of the above)
    """

    
    FOLDED = "HGIEBb"

    cleaned = list()
    for record in records:

        seq = " ".join(list(record.seq))
        cleaned.append(re.sub(r"[UZOB]", "X", seq))

    
    results = pipeline(cleaned, batch_size = 1024)

    folded_residues_list = list()
    for result_for_one_protein in results: 

        folded_residues = sum(1 for res in result_for_one_protein if res['entity'] in FOLDED) / len(result_for_one_protein)
        folded_residues_list.append(folded_residues)

    ids = [record.id for record in records]

    return dict(zip(ids, folded_residues_list))

if __name__ == "__main__":

    import argparse
    import polars as pl 

    parser = argparse.ArgumentParser(description='Predict secondary structure of proteins')

    parser.add_argument('--input', type=str, help='Input file in FASTA format')
    parser.add_argument('--output', type=str, help='Output file in parquet format')

    args = parser.parse_args()

    records = list(SeqIO.parse(args.input, "fasta"))

    results = get_ss(records)

    pl.DataFrame({"id" : results.keys(), "SS" : results.values() }).write_parquet(f"{args.output}.parquet")

    







