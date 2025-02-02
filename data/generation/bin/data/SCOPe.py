import requests
import tarfile
from io import BytesIO
from io import StringIO
import os
import shutil
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path, PosixPath
import logging

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
    
def create_database_directory(cleanup = False):

    """Creates the database directory and removes any existing contents."""

    search_main_directory()
    
    global database
    database = main_directory / "database"

    database.mkdir(exist_ok=True)
    
    if cleanup:
        for path in database.iterdir():
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                logging.error(f"Error deleting file {path}: {e}")
    
def create_directories(cat_path : PosixPath, cov, id):

    """Creates the directories for the category, coverage, and identity."""

    param_paths = cat_path / Path(f"cov_{cov}_iden_{id}")
    param_paths.mkdir(exist_ok=True)
    pdb_path = param_paths / Path("pdb")
    pdb_path.mkdir(exist_ok=True)
    fasta_path = param_paths / Path("fasta")
    fasta_path.mkdir(exist_ok=True)

    return pdb_path, fasta_path

def download_and_extract_tarball(url):
    

    response = requests.get(url)
    tar = tarfile.open(fileobj=BytesIO(response.content), mode="r:gz")
    extracted_files = {}
    for member in tar.getmembers():
        if member.isfile():
            file_content = tar.extractfile(member).read().decode('utf-8')
            extracted_files[member.name] = file_content
    
    return extracted_files

def write_pdbstyle_data_to_disk(pdbstyle_data, output_dir : PosixPath):

    for name, content in pdbstyle_data.items():
        file_path = output_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('w') as file:
            file.write(content)
    
    print(f"All files have been written to {output_dir}")

def download_pdbstyle(write = False):

    pdbstyle_tree = []
    
    try:
        for i in tqdm(range(1, 9)):
    
            print(f"Processing pdbstyle-2.08-{i}")
            url = f"https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.08-{i}.tgz"
            extracted_files = download_and_extract_tarball(url)
            pdbstyle_tree.extend([name for name in extracted_files.keys() if name.endswith('.ent')])

            pdb_output_dir = database / f"pdbs/SCOPe/pdbstyle-2.08-{i}"
            pdb_output_dir.mkdir(parents=True, exist_ok=True)

            if write:
                write_pdbstyle_data_to_disk(extracted_files, output_dir=pdb_output_dir)

    
    except requests.exceptions.HTTPError as e:
        print(f"Error downloading file : {e}")
    except Exception as e:
        print(f"Error: {e}")

    if write:
        with (main_directory / "pdbstyle_tree.txt").open("w") as f:
            for line in pdbstyle_tree:
                f.write(f"{line}\n")
    
    return pdbstyle_tree

def download_and_extract_class_sequences(url, min_length, max_length):

    print("Downloading and extracting sequences ...")

    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")

    fasta_content = response.text

    fasta_io = StringIO(fasta_content)

    fasta_io = main_directory / "input/astral-scopedom-seqres-gd-all-2.08-stable.fa"

    classes = {}
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    expected_classes = "abcdeg"    

    for record in SeqIO.parse(fasta_io, "fasta"):

        # Format : d6iyia_ a.1.1.0 (A:) automated matches {Acipenser stellatus [TaxId: 7903]}
        # We expect the class to be the letter after the first space in the description
        class_ = record.description.split()[1][0]
        seq = record.seq.upper()

        if set(seq).issubset(AAs) and min_length <= len(seq) <= max_length:
            if class_ not in classes:
                classes[class_] = []
                print(f"Adding class {class_}")
            classes[class_].append(SeqRecord(Seq(seq), id=record.id, description=record.description))


    return classes

def main():

    create_database_directory()

    """
    pdbstyle_tree = download_pdbstyle(write = True)

    with open("pdbstyle_tree.txt", "r") as file:
        
        pdbstyle_tree = [line.strip() for line in file.readlines()]

    id_to_path = { 
        os.path.basename(path).split('.')[0] : Path(path)
        for path in pdbstyle_tree
    }

    
    """

    fasta_url = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-all-2.08-stable.fa"

    classes = download_and_extract_class_sequences(fasta_url, MIN_LENGTH, MAX_LENGTH)

    result = {}

    result["Small"] = classes.get("g", [])

    structured_solution_keys = ["a", "b", "c", "d", "e"]
    result["Structured_solution"] = sum((classes.get(key, []) for key in structured_solution_keys), [])

    return result

if __name__ == "__main__":


    main()








            

