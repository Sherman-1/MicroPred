from pTrans_embeddings import main as pTrans_embeddings
from sequence_descriptors import process_data

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from pathlib import Path
import os
import glob


def search_main_directory():

    """
    Iteratively searches for the peptide directory and returns its absolute path.
    """

    global main_directory
    main_directory = None
    for i in range(3): 
        backward = "../" * i
        main_directory = Path(f"{backward}Peptides").resolve()
        if main_directory.exists():
            break

    if main_directory is None:
        raise FileNotFoundError("Peptide directory not found")
    
    print(f"Working on the main directory : {main_directory}")
    
    return main_directory

def main():

    search_main_directory()

    writting_dir = main_directory / "database" / "descriptors"
    writting_dir.mkdir(exist_ok = True)

    print("Bonjour")

    files = ["S3_compound.fasta", "Small_compound.fasta"]

    for file in files:

        fasta = main_directory / file

        print(f"Working on {fasta}")
        
        records = list(SeqIO.parse(fasta, "fasta"))

        category = Path(fasta).stem

        seq_based_descriptors = process_data(records, category) 

        #embeddings = pTrans_embeddings(records)

        seq_based_descriptors.write_parquet(f"{category}.parquet")

        

if __name__ == "__main__":

    main()