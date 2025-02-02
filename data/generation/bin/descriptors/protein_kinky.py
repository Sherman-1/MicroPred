import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

import polars as pl

import argparse
import pathlib
import glob

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


PARSER = PDBParser(QUIET=True)
PPB=PPBuilder()


def calculate_curvature(points, window_size=5):
    curvature = []
    for i in range(len(points) - window_size):

        segment = points[i:i+window_size]

        pca = PCA(n_components=2)
        pca.fit(segment)

        eigenvalues = pca.explained_variance_
        curvature.append(eigenvalues[1] / eigenvalues[0])  

    return np.array(curvature)

def pca_analysis(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.explained_variance_ratio_

def get_points_from_pdb(pdb_path):

    structure = PARSER.get_structure("protein", pdb_path)
    try:
        full_barycentres = []
        for model in structure:
            for chain in model:     
                coords = []
                for residue in chain:
                    if residue.id[0] == ' ':  # Only consider standard residues (ATOM records)
                        
                        barycentre = np.mean([atom.get_coord() for atom in residue.get_unpacked_list()], axis=0)
                        coords.append(barycentre)
                full_barycentres.extend(coords)
            # Only compute the first model
            break

        return np.array(full_barycentres)
        
    except Exception as e:
        print(f"Error processing {structure.id}: {e}")
        return None
    

if __name__ == "__main__": 



    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type = str, required = True,
                        help = "PDB directory or file")
    
    args = parser.parse_args()

    if pathlib.Path(args.input).is_dir():

        pdb_files = [ pathlib.Path(path) for path in glob.glob(f"{args.input}/*.pdb") ]

    else:

        pdb_files = [ pathlib.Path(args.input) ]

    results = {'id' : list(), 'mean_curvature' : list(), 'std_curvature' : list() }
    for pdbpath in tqdm(pdb_files, desc  = "Processing PDB files"):

        pdb_id = pdbpath.stem

        points = get_points_from_pdb(pdbpath)
        if points is None:
            continue

        try:
            curvatures = calculate_curvature(points)
            mean_curvature = curvatures.mean()
            std_curvature = curvatures.std()

            results['id'].append(pdb_id)
            results['mean_curvature'].append(mean_curvature)
            results['std_curvature'].append(std_curvature)

        except Exception as e:
            continue

        
        

    df = pl.DataFrame(results)

    plt.figure(figsize = (10,10))
    sns.kdeplot(data = df, x = 'mean_curvature')
    plt.savefig('mean_curvature.png')
    plt.close()

    plt.figure(figsize = (10,10))
    sns.kdeplot(data = df, x = 'std_curvature')
    plt.savefig('std_curvature.png')
    plt.close() 


    


