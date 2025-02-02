#!/usr/bin/env python3

import polars as pl 
import multiprocessing as mp 
from MMseqs import MMseqs2
from pathlib import Path, PosixPath
from Bio import SeqIO, SeqRecord, Seq
from tqdm import tqdm
import os
import shutil
import random
from glob import glob
import logging

from peripheral import peripheral
from transmembrane import transmembrane

MIN_LENGTH = 20
MAX_LENGTH = 100
MIN_SEGMENT_LENGTH = 10
INNER_MARGIN = 0
CLOSE_MARGIN = 5
MARGIN = 20
GAPS = 1
IORF_CSV = "input/iORFs.csv"
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
  

def get_paths():

    """
    ## type_id correspondance : 

    #   1 : Multitopic transmembrane
    #   2 : Monotopic / Peripherals
    #   3 : Peptides 

    ## classtype_id correspondance :

    ###### Multitopic transmembrane ######
    #  1  : Alpha-helical polytopic 
    #  11 : Bitopic proteins 
    #  2  : Beta barrel 

    ###### Monotopic ######
    #  4  : All alpha 
    #  3  : All beta
    #  5  : Alpha / beta
    #  6  : Alpha + beta

    ###### Peptides #######
    #  7  : Alpha-helical peptides
    #  8  : Beta-helical peptides
    #  9  : Beta hairpins 
    #  10 : Non-regular peptides



    Returns paths for each category of peptides/proteins that are present in the OPM and Membranome databases.

    """

    metadata = (

        pl.read_csv("input/proteins-2024-05-07.csv", separator = ",", infer_schema_length = 20000)
            .with_columns(
                pl.concat_str([
                        pl.lit("input/OPM"),
                        pl.concat_str([pl.col("pdbid"), pl.lit(".pdb")], separator = "")           
                    ], separator = "/",
                ).alias("pdb_path")
            )
    )

    ## FILTERS ## 

    # Transmembranes

    # Annotated bitopic proteins, put threshold of 20 to avoid some false positives
    bitopic_proteins = ((pl.col("classtype_id") == 11) & (pl.col("thickness") >= 20)) 
    # All peptides that are crossing the membrane ( > 20 ), regardless of their folding type
    bitopic_peptides = ((pl.col("type_id") == 3) & (pl.col("thickness") >= 20)) 
    # Multitopic proteins to be cut 
    polytopic_proteins = (pl.col("classtype_id") == 1) & (pl.col("thickness") >= 20)

    # Peripherals

    # Some "peripheral" proteins are anchored to the mb through lipidation, empirical threshold of 12 to remove them
    peripheral_proteins = (pl.col("type_id") == 2) & (pl.col("thickness") <= 12) 
    # All peptides that are not beta-helical or non-regular and that are not crossing the membrane
    peripheral_peptides = ((pl.col("type_id") == 3) & (pl.col("thickness") < 20) & (pl.col("tilt") <= 80)) 
    horizontal_peripheral_peptides = ((pl.col("type_id") == 3) & (pl.col("thickness") < 20) & (pl.col("tilt") > 80))
    # Mis-annotated bitopic proteins
    misannotated_proteins = ((pl.col("classtype_id") == 11) & (pl.col("thickness") < 20))


    ## CREATING PATHS ##

    membranome = [ main_directory / path for path in list(glob("input/membranome/*.pdb")) ]
    bitopic_proteins = [ main_directory / path for path in metadata.filter(bitopic_proteins)["pdb_path"].to_list() ]
    bitopic_peptides = [ main_directory / path for path in metadata.filter(bitopic_peptides)["pdb_path"].to_list() ]
    polytopics = [ main_directory / path for path in metadata.filter(polytopic_proteins)["pdb_path"].to_list() ]

    tm_paths = {
        
        "bitopic" : membranome + bitopic_proteins + bitopic_peptides,
        "polytopic" : polytopics

    }

    peripheral_proteins = [ main_directory / path for path in metadata.filter(peripheral_proteins)["pdb_path"].to_list() ]
    peripheral_peptides = [ main_directory / path for path in metadata.filter(peripheral_peptides)["pdb_path"].to_list() ]
    misannotated_proteins = [ main_directory / path for path in metadata.filter(misannotated_proteins)["pdb_path"].to_list() ]
    horizontal_peptides = [ main_directory / path for path in metadata.filter(horizontal_peripheral_peptides)["pdb_path"].to_list() ]

    peripheral_paths = {

        "peripheral" : peripheral_proteins + peripheral_peptides + misannotated_proteins,
        "horizontal" : horizontal_peptides

    }

    return tm_paths, peripheral_paths

def try_tm(args : list):

    try:
        return transmembrane(*args)
    except Exception as e:
        print(e)
        return None
    
def try_peripheral(args : list):

    try: 
        return peripheral(*args)
    except Exception as e:
        print(e)
        return None
    
def pretty_res(results):

    sequences = [ seq for res in results if res is not None for seq in res["sequences"] ]
    structures = {

        k2: v2
        for res in results if res is not None
        for _dict in res["structures"].values()
        for k2, v2 in _dict.items()
        
    }

    return sequences, structures

def write_pdb(structure: dict, id: str, writing_dir: PosixPath) -> int:

    """Writes the PDB file from the list of lines in the structure dictionary."""

    lines = []
    flag = False
    try:
        for res_number, atom_dict in structure.items():

            
            if not isinstance(atom_dict, dict):
                print(id)
            for atom_number, atom_line in atom_dict.items():
                lines.append(atom_line)
                
        if lines:
            file_name = f"{id}.pdb"
            file_path = writing_dir / file_name
            with open(file_path, "w") as output:
                output.write("".join(lines))  # Lines are already \n terminated
        else:
            logging.error(f"Empty structure for ID: {id}")
            return e
    except Exception as e:
        logging.error(f"Error writing PDB for ID: {id}, Error: {e}")
        return e

    return 0  

def validate_pdb_file(pdb_file : str) -> bool:

    try:
        if os.path.getsize(pdb_file) != 0:
            with open(pdb_file, "r") as f:
                return True
            
    except FileNotFoundError:
        logging.error(f"File not found: {pdb_file}")
    except IOError as e:
        logging.error(f"IO error reading {pdb_file}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error reading {pdb_file}: {e}")
    return False

def write_pdbs(writing_dir, common_keys, structures) -> list:

    written_pdbs = []
    for key in common_keys:
        returned = write_pdb(structures[key], key, writing_dir)
        flag = validate_pdb_file(f"{writing_dir}/{key}.pdb")

        if returned == 0 and flag:
            written_pdbs.append(key)
        else:
            print(returned)

    return written_pdbs



def write_fasta_sequences(fasta_path, category, cov, id, common_keys, representatives):

    seq_to_write = []
    for key in common_keys:
        if validate_pdb_file(f"{fasta_path}/../pdb/{key}.pdb"):
            if key in representatives:
                seq_to_write.append(representatives[key])
            else:
                logging.error(f"Key {key} not found in representatives")

    SeqIO.write(seq_to_write, f"{fasta_path}/{category}_cov_{cov}_iden_{id}.fasta", "fasta")

def process_pool(paths, process_function):

    if process_function == try_tm:

        args = [(path, None, MARGIN, INNER_MARGIN, MIN_LENGTH, MAX_LENGTH, GAPS, IORF_CSV, None, False) for path in paths]
    
    elif process_function == try_peripheral:

        args = [(path, CLOSE_MARGIN, MARGIN, MIN_LENGTH, MAX_LENGTH, MIN_SEGMENT_LENGTH, IORF_CSV, None, GAPS, False) for path in paths]

    else:
        raise ValueError("Unknown function used in pool")

    with mp.Pool(max(1, mp.cpu_count() - 4)) as pool:
        results = pool.map(process_function, args)

    return [ res for res in results if res is not None ]

def process_datasets(datasets, process_function):

    pbar = tqdm(datasets.items(), leave=False, total=len(datasets))

    result = {}
    for category, paths in pbar:

        pbar.set_description(f"Processing {category}")
        results = process_pool(paths, process_function)
        sequences, structures = pretty_res(results)

        print(f"Found {len(sequences)} sequences for {category}")
        print(f"Found {len(structures)} structures for {category}")

        writing_dir = main_directory / "database/OPM" / category
        writing_dir.mkdir(parents=True, exist_ok=True)

        written_pdbs_keys = write_pdbs(writing_dir, structures.keys(), structures)

        print(f"Written {len(written_pdbs_keys)} PDB files for {category}")

        good_sequences = [ seq for seq in sequences if seq.id in written_pdbs_keys ]

        print(f"Found {len(good_sequences)} sequences for {category}")

        result[category] = [ seq for seq in good_sequences if MIN_LENGTH <= len(seq) <= MAX_LENGTH ]

    pbar.close()

    return result

def main():

    search_main_directory()

    transmembranes, peripherals = get_paths()

    print("Processing Transmembranes ...")
    tms = process_datasets(transmembranes, try_tm)

    print("Processing peripherals ...")
    periphs = process_datasets(peripherals, try_peripheral)

    return {**tms, **periphs}

 
if __name__ == '__main__':

    search_main_directory()

    transmembranes, peripherals = get_paths()

    print("Processing Transmembranes ...")
    tms = process_datasets(transmembranes, try_tm)

    print("Processing peripherals ...")
    periphs = process_datasets(peripherals, try_peripheral)

    assert type(tms) == dict and type(periphs) == dict

    records = list()
    for category, records in tms.items():
        
        SeqIO.write(records, f"test/{category}.fasta", "fasta")

    for category, records in periphs.items():

        SeqIO.write(records, f"test/{category}.fasta", "fasta")


        

    
        

    


    
        
