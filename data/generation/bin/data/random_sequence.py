#!/usr/bin/env python3

import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def generate_random_protein_sequences(n, min_length, max_length) -> list[str]:

    """
    Generates random protein sequences based on two modalities:
        - Random sequences with uniform distribution for all AAs
        - Random sequences with AA proportions taken from databanks
    """


    # AA composition for mean ( median ? ) protein from https://web.expasy.org/protscale/pscale/A.A.Swiss-Prot.html
    amino_acid_composition = {

        "A": 8.25,  # Ala
        "R": 5.53,  # Arg
        "N": 4.06,  # Asn
        "D": 5.45,  # Asp
        "C": 1.37,  # Cys
        "Q": 3.93,  # Gln
        "E": 6.75,  # Glu
        "G": 7.07,  # Gly
        "H": 2.27,  # His
        "I": 5.96,  # Ile
        "L": 9.66,  # Leu
        "K": 5.84,  # Lys
        "M": 2.42,  # Met
        "F": 3.86,  # Phe
        "P": 4.70,  # Pro
        "S": 6.56,  # Ser
        "T": 5.34,  # Thr
        "W": 1.08,  # Trp
        "Y": 2.92,  # Tyr
        "V": 6.87   # Val
    }

    sequences = {}
    aa = "ACDEFGHIKLMNPQRSTVWY"

    # Uniform 
    buffer = []
    for i in range(n):
        length = random.randint(min_length, max_length)
        sequence = "".join(random.choices(aa, k=length))
        record = SeqRecord(Seq(sequence), id=f"random_uniform_{i}", description="Random uniform sequence")
        buffer.append(record)
    sequences["random_uniform"] = buffer

    # Proportional
    buffer = []
    for i in range(n):
        length = random.randint(min_length, max_length)
        sequence = "".join(random.choices(list(amino_acid_composition.keys()), weights=list(amino_acid_composition.values()), k=length))
        record = SeqRecord(Seq(sequence), id=f"random_proportional_{i}", description="Random proportional sequence")
        buffer.append(record)
    sequences["random_proportional"] = buffer

    return sequences


def main():

    return generate_random_protein_sequences(10, 50, 100)

if __name__ == "__main__":

    print(main())


    
    
    