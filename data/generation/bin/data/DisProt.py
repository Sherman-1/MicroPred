#!/usr/bin/env python3

from pathlib import Path 
import shutil
import requests
import logging
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from io import StringIO
import random
from MMseqs import MMseqs2


COVS = [0.3, 0.5, 0.7]
IDENS = [0.3, 0.5, 0.7]
MIN_LENGTH = 20
MAX_LENGTH = 100


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
    
def create_database_directory():

    """Creates the database directory and removes any existing contents."""

    search_main_directory()
    
    global database
    database = main_directory / "database"

    database.mkdir(exist_ok=True)
    for path in database.iterdir():
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as e:
            logging.error(f"Error deleting file {path}: {e}")


def get_random_length(length_distribution):

    """
    Pick a random length from the length_distribution that is greater than or equal to min_length.
    """
    return random.choice(length_distribution)

def cut_sequence(sequence, length_distribution, min_length):
    """
    Cut a given sequence into pieces based on the length_distribution and min_length.
    """
    cut_lengths = []
    remaining_length = len(sequence)

    # Select random lengths until the remaining_length is less than min_length
    while remaining_length >= min_length:
        cut_length = get_random_length(length_distribution)
        if remaining_length - cut_length < min_length:
            break
        cut_lengths.append(cut_length)
        remaining_length -= cut_length

    # Cut the sequence based on the selected cut_lengths
    cuts = []
    start = 0
    for cut_length in cut_lengths:
        cuts.append(sequence[start : start + cut_length])
        start += cut_length

    return cuts

def cut_fasta(fasta : SeqRecord, length_distribution, min_length, max_length):

    """
    Cut sequences in the input_fasta based on length_distribution and min_length.
    """

    records = []
    for record in fasta:
        cut_seqs = cut_sequence(record.seq, length_distribution, min_length)
        for i, cut_seq in enumerate(cut_seqs):
            records.append(SeqRecord(Seq(cut_seq), id=f"{record.id}_{i}", description=""))

    return [ seq for seq in records if min_length <= len(seq.seq) <= max_length]
            
def download_disprot_fasta(link):

    """
    Download the DisProt FASTA file and save it to the output_file.
    """
    
    response = requests.get(link)

    if response.status_code == 200:
        fasta_content = response.text
        fasta_io = StringIO(fasta_content)
        fasta_iterable = SeqIO.parse(fasta_io, "fasta")
    else:
        print("Failed to retrieve the file.")

    return fasta_iterable

def main():

    search_main_directory()


    disprot_link = "https://disprot.org/api/search?release=2024_06&show_ambiguous=false&show_obsolete=false&format=fasta&namespace=all&get_consensus=false"

    iORFs_distribution_handle = main_directory / "input/iORFs.csv"
    with open(iORFs_distribution_handle, "r") as f:
        iORFs_distribution = [int(line.strip()) for line in f.readlines()]

    min_length = MIN_LENGTH
    max_length = MAX_LENGTH

    disprot_iterable = download_disprot_fasta(disprot_link)

    records = cut_fasta(disprot_iterable, iORFs_distribution, min_length, max_length)

    return {"disprot": records}

    

    
if __name__ == "__main__":


    print(main())