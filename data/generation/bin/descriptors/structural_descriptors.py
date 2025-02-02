import numpy as np
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import polars as pl
import glob
import multiprocessing
import warnings
import os
import subprocess
import tempfile

from HullRadV9 import compute_metrics as hullrad
from hydromoment import analyze_sequence
from pyStride import stride

import argparse 

def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


PDBParser.get_structure = suppress_warnings(PDBParser.get_structure)
parser = PDBParser(QUIET=True)
ppb=PPBuilder()


AA = { "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V" }

def extract_matrix(structure):
    """
    Returns distance matrix of a PDB file.
    """

    try:
        full_barycentres = []
        for model in structure:
            for chain in model:     
                coords = []
                for residue in chain:
                    if residue.id[0] == ' ':  # Only consider standard residues (ATOM records)
                        # Compute the barycentre of the residue
                        # Only consider atoms that are part of the side chain
                        barycentre = np.mean([atom.get_coord() for atom in residue.get_unpacked_list()], axis=0)
                        coords.append(barycentre)
                full_barycentres.extend(coords)
            # Only compute the first model
            break

        return squareform(pdist(full_barycentres))
        
    except Exception as e:
        print(f"Error processing {structure.id}: {e}")
        return None


def describe_matrix(distance_matrix, k=5):

    """
    Returns a set of descriptive statistics of a distance matrix.
    """

    n = distance_matrix.shape[0]

    frobenius = np.linalg.norm(distance_matrix, 'fro')
    
    eigvals, _ = np.linalg.eigh(distance_matrix)
    eigens = { f"eig_{i}": eigvals[i] for i in range(k) }

    triu_indices = np.triu_indices_from(distance_matrix, k=1)  # Upper triangular indices
    distances = distance_matrix[triu_indices]

    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    q25_distance = np.percentile(distances, 25)
    q75_distance = np.percentile(distances, 75)

    return {

        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "std_distance": std_distance,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "q25_distance": q25_distance,
        "q75_distance": q75_distance,
        "frobenius": frobenius,
        **eigens
    }

def _compute(pdb_path):

    basename = os.path.basename(pdb_path).split(".")[0]

    try:
        
        structure = parser.get_structure("protein", pdb_path)
        distance_matrix = extract_matrix(structure)
        
    except Exception as e:
        print(e)
        return None

    seq = []
    for pp in ppb.build_peptides(structure):
        seq.extend(pp.get_sequence())
        
    if len(seq) == 0:
        print("Empty sequence")
        print(seq)
        return None
    
    if (set(seq) - AA) != set():
        print(f"Invalid amino acids: {set(seq) - AA} for sequence \n{"".join(seq)}")
        return None
        
    aa_composition = { aa : seq.count(aa) / len(seq) for aa in AA }

    chem = analyze_sequence(sequence = seq, window = -1)

    chem_features = {

        "av_h" : np.mean([d[6] for d in chem]),
        "av_uH" : np.mean([d[7] for d in chem]),
        "d" : np.mean([d[8] for d in chem]),
        "n_tot_pol" : np.mean([d[9] for d in chem]),
        "n_tot_apol" : np.mean([d[10] for d in chem]),
        "n_charged" : np.mean([d[11] for d in chem]),
        "n_aromatic" : np.mean([d[12] for d in chem])

    }

    geometric_features = describe_matrix(distance_matrix)

    phys_features = hullrad(pdb_path)
    
    second_struct_percent = stride(pdb_path, output_type = "percentage")
    
    if phys_features is None or second_struct_percent is None or distance_matrix is None or geometric_features is None:
        return None

    else: 


        return { **{"id": basename, "second_struct_percent": second_struct_percent}, 
                **geometric_features, **phys_features, **chem_features, **aa_composition}


def pool_process(pdb_paths, num_processes):
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        print(f"Processing {len(pdb_paths)} PDB files using {num_processes} processes")
        results = pool.map(_compute, pdb_paths)

    return [ res for res in results if res is not None ]


def structural_descriptors(pdb_paths, num_processes, category = None):

    if category is None: 

        category = "PLACEHOLDER"
    
    results = pool_process(pdb_paths, num_processes)

    return pl.DataFrame(results).with_columns(category = pl.lit(category))

if __name__ == "__main__":

    import argparse 

    parser = argparse.ArgumentParser(description="Compute structural descriptors from PDB files.")
    parser.add_argument("--pdb_dir", type=str, help="Path to the directory containing PDB files.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use.")
    parser.add_argument("--category", type=str, help="Category of the PDB files.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")

    args = parser.parse_args()

    pdb_paths = glob.glob(f"{args.pdb_dir}/*.pdb")
    df = structural_descriptors(pdb_paths, args.num_processes, args.category)

    df.write_parquet(f"{args.output_dir}/{args.category}.parquet")




    












