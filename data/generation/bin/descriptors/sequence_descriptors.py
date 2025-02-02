#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/07/2024 

@author : Simon HERMAN

Based on Christos Papadopoulos work
"""

print("============== Importing dependencies ============= ")

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


import numpy as np

from pathlib import Path

import tool_functions
import iupred_funcs


from pyHCA.core.annotateHCA import _annotation_aminoacids
from pyHCA.core.classHCA import compute_disstat

import multiprocessing

import argparse
import polars as pl  

from tqdm import tqdm


# ------------------------ #
#   AA index descriptors   #
# ------------------------ #

AA = { "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V" }


main_directory = None
for i in range(10): 
    backward = "../" * i
    main_directory = Path(f"{backward}Peptides").resolve()
    if main_directory.exists():
        print(main_directory)
        break



def setup_aa_indexes():

    aa_indexes_names = []

    global aa_indexes
    aa_indexes = {}

    # Read the aa_indexes_names from the first file
    with open(main_directory / "bin" / "descriptors" / "support_data" / "selected_aaindex1_reformated_58_Van_Westen.dms", "r") as f:
        for line in f:
            if line.startswith("#"):
                aa = line.split()[1:]
                continue
            aa_indexes_names.append(line.split()[0].strip())

    # Read the aa_indexes from the second file
    with open(main_directory / "bin" / "descriptors" / "support_data" / "AA_index_scaled.txt", "r") as f:
        for line in f:
            aa = line.split()[0].strip()
            for i, aaj in enumerate(line.split()[1:]):
                if aa_indexes_names[i] not in aa_indexes:
                    aa_indexes[aa_indexes_names[i]] = {}
                aa_indexes[aa_indexes_names[i]][aa] = float(aaj)
    



def calculate_PDT(seq , aa_indexes , lamda):

    PDT = {}
    lamda = 2
    for aaj in aa_indexes:
        Ds = []
        for x in range(len(seq) - lamda):
            aa1 = seq[x]
            aa2 = seq[x + lamda]
            D = np.square(tool_functions.I(aa=aa1, aaj=aaj, aa_indexes=aa_indexes) - tool_functions.I(aa=aa2, aaj=aaj, aa_indexes=aa_indexes))
            Ds.append(D)

        PDT[aaj] = str(round(np.sum(Ds) / (len(seq) - lamda), 4))
        
    return(PDT)


def compute_sequence_metrics(record : SeqRecord):

    seq = str(record.seq)
    name = record.id

    try:

        # Calculate the mean of 58 aa index descriptors
        mean_aa_index = tool_functions.calculate_the_mean(seq, aa_indexes)

        # ==================================================================== #
        # Calculate the PDTn of 58 aa index descriptors
        #PDT2 = calculate_PDT(seq =seq , aa_indexes = aa_indexes, lamda =2)
        #PDT3 = calculate_PDT(seq =seq , aa_indexes = aa_indexes, lamda =3)
        #PDT4 = calculate_PDT(seq =seq , aa_indexes = aa_indexes, lamda =4)
        #PDT5 = calculate_PDT(seq =seq , aa_indexes = aa_indexes, lamda =5)
        #PDT8 = calculate_PDT(seq =seq , aa_indexes = aa_indexes, lamda =8)

        # ------ Protein Sequence Based Descriptors:   ----------------------------------- #

        hydrophobic_portion = tool_functions.hydrophobic_perc(seq)
        iupred_score   = iupred_funcs.iupred(seq=seq, mode="short")[0]
        iupred_portion = tool_functions.calculate_proportion_of_seq_disordered(iupred_score)
        iupred_mean    = round(sum(iupred_score) / len(iupred_score), 3)
        anchor_score   = iupred_funcs.anchor2(seq=seq,iupred_scores=iupred_score)
        anchor_portion = tool_functions.calculate_proportion_of_seq_disordered(anchor_score)
        score, pvalue = compute_disstat(0, len(seq), _annotation_aminoacids(seq=seq, method="domain", verbose=False)["cluster"])
        HCA_score = round(score, 2)
        
        aa_frequency = { aa : seq.count(aa) / len(seq) for aa in AA }

        #hca = HCA(seq=seq, querynames=name)
        #bc, bc_SS = tool_functions.get_hca_barcode(hca=hca, orf=name)

        return { 

                "id" : name,
                **mean_aa_index,
                "hydrophobic_portion" : hydrophobic_portion,
                "iupred_portion" : iupred_portion,
                "iupred_mean" : iupred_mean,
                "anchor_portion" : anchor_portion,
                "HCA_score" : HCA_score,
                **aa_frequency,
                "seq_len" : len(seq)
                #"bc" : bc,
                #"bc_SS" : bc_SS

            }
    
    except Exception as e:
        print(f"Exception while parsing seq {id} : {e}")
        return None


def pool_process(records, num_processes):
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        print(f"Processing {len(records)} sequences using {num_processes} processes")
        results = pool.map(compute_sequence_metrics, records)

    return [ res for res in results if res is not None ]


def validate_records(records) -> list: 

    AA = { "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V" }

    valid_records = []

    for record in records: 

        if len(record.seq) == 0 or set(str(record.seq)) <= AA:
            valid_records.append(record)

        else:

            print(f"Invalid record : {record.seq}")

    return valid_records


def read_fasta( fasta_path ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","").replace("*","") # drop gaps and cast to upper-case

    bad_keys = list()

    for key in sequences:

        if not set(sequences[key]).issubset(set(AA)):
            bad_keys.append(key)

    for key in bad_keys:

        del sequences[key]
                
    return [ SeqRecord(Seq(seq), id=uniprot_id) for uniprot_id, seq in sequences.items() ]

def process_data(records, category): 

    setup_aa_indexes()
    #results = pool_process(records, num_processes=8)
    
    results = []
    for record in tqdm(records):

        res = compute_sequence_metrics(record)
        if res is not None:
            results.append(res)

    columns = [
        "id",
        "seq_len",
        "hydrophobic_portion",
        "iupred_portion",
        "iupred_mean",
        "anchor_portion",
        "HCA_score",
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "ARGP820103",
        "BHAR880101",
        "CHAM810101",
        "CHAM820101",
        "CHAM830101",
        "CHAM830107",
        "CHAM830108",
        "CHOP780201",
        "CHOP780202",
        "CHOP780203",
        "CIDH920105",
        "FASG760101",
        "FAUJ880102",
        "FAUJ880103",
        "FAUJ880104",
        "FAUJ880105",
        "FAUJ880106",
        "FAUJ880109",
        "FAUJ880110",
        "FAUJ880111",
        "FAUJ880112",
        "FAUJ880113",
        "GRAR740102",
        "JANJ780102",
        "JANJ780103",
        "JOND920102",
        "JUNJ780101",
        "KLEP840101",
        "KRIW790101",
        "KYTJ820101",
        "LEVM760102",
        "LEVM760103",
        "LEVM760104",
        "LEVM760105",
        "LEVM760106",
        "LEVM760107",
        "NISK800101",
        "NISK860101",
        "PONP800101",
        "RACS770103",
        "RADA880108",
        "ROSG850101",
        "ROSG850102",
        "ROSM880102",
        "WARP780101",
        "WOLR810101",
        "VINM940101",
        "TAKK010101",
        "MONM990201",
        "KOEP990101",
        "KOEP990102",
        "MITS020101",
        "COSI940101",
        "PONP930101",
        "ZHOH040102",
        "ZHOH040103",
        "BAEK050101",
        "CASG920101",
        "category"
    ]


    special_columns = {

        "id": pl.Utf8,
        "category": pl.Utf8,
        "seq_len": pl.Int32,
    }

    # Only 3 columns are not fp32
    schema = {col: special_columns.get(col, pl.Float32) for col in columns}
        
    df = pl.DataFrame(results, schema=schema)

    df = df.with_columns(category = pl.lit(category)).select(columns)
    
    return df

if __name__ == "__main__": 

    import argparse 
    from Bio import SeqIO

    print("============== Parsing arguments ============= ")

    parser = argparse.ArgumentParser(description="Compute sequence descriptors")

    parser.add_argument("--input", type=str, help="Input fasta file")
    parser.add_argument("--output", type=str, help="Output CSV file")

    args = parser.parse_args()

    print("============== Reading data =============")

    records = read_fasta(args.input)
    records = validate_records(records)

    print(f"Found {len(records)} valid sequences")

    print("============== Processing data ============= ")
    
    df = process_data(records, category="test")

    df.write_csv(f"{args.output}.tsv", include_header=True, separator="\t")
    




