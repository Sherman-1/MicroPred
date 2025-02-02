from OPM import main as OPM
from SCOPe import main as SCOPe
from DisProt import main as DisProt
from random_sequence import generate_random_protein_sequences

from MMseqs import MMseqs2
from Bio import SeqIO

from pathlib import Path, PosixPath
import random
import os
import logging
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

COVS = [0.3, 0.5, 0.7]
IDENS = [0.3, 0.5, 0.7]

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
    
def create_database_directory(cleanup = False):

    """Creates the database directory and removes any existing contents."""

    search_main_directory()
    
    global database
    global representatives_path
    database = main_directory / "database"
    representatives_path = database / "representatives" 

    database.mkdir(exist_ok=True)
    
    if cleanup:
        for path in database.iterdir():
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                logging.error(f"Error deleting {path}: {e}")

    representatives_path.mkdir(exist_ok=True)

def setup_environment(clean_database : bool):

    """Sets up the random seed, mmseqs API, logging, and database directory."""

    random.seed(42)

    global mmseqs 
    mmseqs = MMseqs2(threads = max(1, os.cpu_count() - 4), cleanup=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    create_database_directory(cleanup=clean_database)

    tmpdir = Path("tmp")
    tmpdir.mkdir(parents = True, exist_ok = True)

def process_representatives(category, fasta_file):

    """
    Processes the representatives of the sequences in the fasta file, for different coverages and identities.
    For each iteration, stores the number of representatives and write their ids to a file.
    """

    data = {}

    for cov in COVS:

        data[cov] = {}

        for iden in IDENS:

            print(f"Processing category {category} with coverage {cov} and identity {iden}")

            data[cov][iden] = 0

            representatives = mmseqs.fasta2representativeseq(fasta_file=fasta_file, cov=cov, iden=iden, cov_mode=0)
            ids = list(representatives.keys())

            with open(representatives_path / f"{category}_cov_{cov}_iden_{iden}.txt", "w") as f:
                f.write("\n".join(ids))

            data[cov][iden] = len(ids)

    return data
            
def main():

    setup_environment(clean_database=True)

    print("------------------- Generating dataset -------------------")
    print("======== DisProt =========")
    DisProt_sequences = DisProt()
    print("======== OPM ========= ")
    OPM_sequences = OPM()
    print("======== SCOPe =========")
    SCOPe_sequences = SCOPe()
    print("======== Random sequences =========")
    random_sequences = generate_random_protein_sequences(1000, 20, 100)

    full_dataset =  { **DisProt_sequences }

    stats = {}
    
    print(" ------------------- Processing representatives ------------------- ")

    for category, sequences in full_dataset.items():

        print(f" ========= Processing category {category} ========= ")

        fasta_path = database / "full" 
        fasta_path.mkdir(parents = True, exist_ok = True)

        fasta_file = fasta_path / f"{category}.fasta"
        SeqIO.write(sequences, fasta_file, "fasta")

        data = process_representatives(category, fasta_file)

        stats[category] = data

if __name__ == "__main__":

    main()