import numpy as np
from collections import defaultdict
from collections import OrderedDict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import random
import os
from pathlib import PosixPath
import re
import logging
from functools import wraps


ascii_separator = "\n" + ("#" * 40 + "\n") * 2 + "\n"


#   _____ _                _____ _____ ______  _____ 
#  / ____| |        /\    / ____/ ____|  ____|/ ____|
# | |    | |       /  \  | (___| (___ | |__  | (___  
# | |    | |      / /\ \  \___ \\___ \|  __|  \___ \ 
# | |____| |____ / ____ \ ____) |___) | |____ ____) |
#  \_____|______/_/    \_\_____/_____/|______|_____/ 
                                                    
                                                    
class SuperOD(OrderedDict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            
            keys = list(self.keys())[key]
            return SuperOD((k, self[k]) for k in keys)
        else:
            
            return OrderedDict.__getitem__(self, key)
        
class ResidueError(ValueError):
    """Error to handle non canonic residues in files"""
    def __init__(self, res, message="Residue not in the list of canonic residues"):
        self.res = res
        self.message = message
        super().__init__(f"{self.message}: {self.res}")



#  _      ____   _____  _____ _____ _   _  _____ 
# | |    / __ \ / ____|/ ____|_   _| \ | |/ ____|
# | |   | |  | | |  __| |  __  | | |  \| | |  __ 
# | |   | |  | | | |_ | | |_ | | | | . ` | | |_ |
# | |___| |__| | |__| | |__| |_| |_| |\  | |__| |
# |______\____/ \_____|\_____|_____|_| \_|\_____|
                                                
                                                
                             
def setup_logger(name, log_file, level=logging.DEBUG):
    
    """
    Setting up loggers for the different functions
    """
    
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(lineno)d - %(levelname)s \n%(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) 
    logger.addHandler(handler)
    
    return logger

def exception_catcher(logger) -> callable:

    """
    Catch exceptions returned in a function and log them
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
                raise
        return wrapper
    return decorator


#   _____ _    _ ____  ______ _    _ _   _  _____  _____ 
#  / ____| |  | |  _ \|  ____| |  | | \ | |/ ____|/ ____|
# | (___ | |  | | |_) | |__  | |  | |  \| | |    | (___  
#  \___ \| |  | |  _ <|  __| | |  | | . ` | |     \___ \ 
#  ____) | |__| | |_) | |    | |__| | |\  | |____ ____) |
# |_____/ \____/|____/|_|     \____/|_| \_|\_____|_____/ 

def format_pdb_line(atom_serial, atom_name, res_name, res_seq, x, y, z, occupancy=1.00, temp_factor=0.00, element=''):
    return f"ATOM  {atom_serial:5d} {atom_name:<4}{'':1}{res_name:>3} {'':1}{res_seq:4d}{'':1}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}"

def generate_pdb(ordered_dict : dict) -> list[str]:

    """
    Input : dict of 3D coordinates from pdb file 

    Output : List of string, each string is a PDB-compliant line of 
             the file to write
    """
    lines = []

    aa_dict = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }


    for res_seq, atoms in ordered_dict.items():
        for atom_serial, atom_info in atoms.items():
            atom_name = atom_info['atom_name']
            res_name = atom_info['res_name']
            res_code = aa_dict[res_name]  # Convert the three-letter code to one-letter code
            coord = atom_info['coord']
            x, y, z = coord
            element = atom_name[0]  # Assume the element is the first letter of the atom name
            line = format_pdb_line(atom_serial, atom_name, res_code, res_seq, x, y, z, element=element)
            lines.append(line)
            
    return lines

def size_picker_v3(fasta_file, precomputed_file, min_length = 0, max_length = 1000, n_samples = 1) -> list :

    """
    Same thing as v2 but draw n_samples sizes from the distribution
    all at once, avoiding to parse the fasta file n_samples times
    """

    
    if fasta_file:
        sizes = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            length = len(record.seq)
            if min_length <= length <= max_length:

                sizes.append(length)

    if precomputed_file:
        with open(precomputed_file, "r") as f:
            sizes = [int(line.rstrip('\n')) for line in f]
            sizes = [size for size in sizes if min_length <= size <= max_length]

    
    lengths = []

    if len(sizes) < n_samples: 
        remaining = n_samples - len(sizes)
        lengths.extend(sizes)
        zeros = [0] * remaining
        lengths.extend(zeros)
        return lengths

    else:

        return random.sample(sizes, n_samples)

def pdb_struct_to_fasta(logger, writing_path = ".", write = False, streaming = False, pdb_path = None, pdb_struct = None) -> dict:
    
    """
    PDB Record Format:
    
    Line starts with ATOM
    ----
    1-4    : "ATOM"                            (character)
    7-11   : Atom serial number                (right, integer)
    13-16  : Atom name                         (left*, character)
    17     : Alternate location indicator      (character)
    18-20  : Residue name                      (right, character)
    22     : Chain identifier                  (character)
    23-26  : Residue sequence number           (right, integer)
    27     : Code for insertions of residues   (character)
    31-38  : X orthogonal Å coordinate         (right, real (8.3))
    39-46  : Y orthogonal Å coordinate         (right, real (8.3))
    47-54  : Z orthogonal Å coordinate         (right, real (8.3))
    55-60  : Occupancy                         (right, real (6.2))
    61-66  : Temperature factor                (right, real (6.2))
    73-76  : Segment identifier                (left, character)
    77-78  : Element symbol                    (right, character)
    79-80  : Charge                            (character)
    
    Line starts with HETATM
    ------
    1-6    : "HETATM"                          (character)
    7-80   : Same as ATOM records

    Line starts with TER
    ---
    1-3    : "TER"                             (character)
    7-11   : Serial number                     (right, integer)
    18-20  : Residue name                      (right, character)
    22     : Chain identifier                  (character)
    23-26  : Residue sequence number           (right, integer)
    27     : Code for insertions of residues   (character)  
    """
    
    log_messages = []

    if streaming and pdb_path and write:


        aa_dict = {
                        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                    }

        with open(pdb_path, "r") as pdb:

            protein_name = os.path.basename(pdb_path).split(".")[0]

            last_chain_id = None

            records = []
            seq = ""

            while line := pdb.readline():

                if line.startswith("ATOM"):

                    chain_id = line[21]

                    if not bool(chain_id.strip()):

                        chain_id = "A"
                        log_messages.append(f"Chain ID not found in line {line} : setting it to A")

                    if chain_id != last_chain_id and last_chain_id:

                        records.append(SeqRecord(Seq(seq), id=f"{pdb_struct['protein_name']}_{last_chain_id}", description=""))

                        seq = aa_dict[line[17:20].strip()]

                    else:

                        seq += aa_dict[line[17:20].strip()]

                    last_chain_id = chain_id

                # Store the last sequence
                records.append(SeqRecord(Seq(seq), id=f"{pdb_struct['protein_name']}_{last_chain_id}", description=""))

        SeqIO.write(records, f"{writing_path}/{protein_name}_streaming.fasta", "fasta")

        return 0

    elif pdb_struct:

        records = []
        
        for chain_id in pdb_struct["CA"]:
            
            print(f"Writing chain {chain_id}")

            sequence = ""

            for res_number, data in pdb_struct["CA"][chain_id].items():

                sequence += data["res_name"]
                
            records.append(SeqRecord(Seq(sequence), id=f"{pdb_struct['protein_name']}_{chain_id}", description=f""))

        if write == True:

            if writing_path.endswith("/"):
                
                writing_path = writing_path[:-1]

            SeqIO.write(records, f"{writing_path}/{pdb_struct['protein_name']}.fasta", "fasta")

        return records
    
def read_pdb(logger, file_path, secondary_structure_path = None, verbose = False) -> dict:

    """
    Function to process PDB format records.

    PDB Record Format:
    
    Line starts with ATOM
    ----
    1-4    : "ATOM"                            (character)
    7-11   : Atom serial number                (right, integer)
    13-16  : Atom name                         (left*, character)
    17     : Alternate location indicator      (character)
    18-20  : Residue name                      (right, character)
    22     : Chain identifier                  (character)
    23-26  : Residue sequence number           (right, integer)
    27     : Code for insertions of residues   (character)
    31-38  : X orthogonal Å coordinate         (right, real (8.3))
    39-46  : Y orthogonal Å coordinate         (right, real (8.3))
    47-54  : Z orthogonal Å coordinate         (right, real (8.3))
    55-60  : Occupancy                         (right, real (6.2))
    61-66  : Temperature factor                (right, real (6.2))
    73-76  : Segment identifier                (left, character)
    77-78  : Element symbol                    (right, character)
    79-80  : Charge                            (character)
    
    Line starts with HETATM
    ------
    1-6    : "HETATM"                          (character)
    7-80   : Same as ATOM records

    Line starts with TER
    ---
    1-3    : "TER"                             (character)
    7-11   : Serial number                     (right, integer)
    18-20  : Residue name                      (right, character)
    22     : Chain identifier                  (character)
    23-26  : Residue sequence number           (right, integer)
    27     : Code for insertions of residues   (character)  
    """
    
    log_messages = []
    

    aa_dict = {
                    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                }

    pdb_struct = {}

    assert type(file_path) == PosixPath

    pdb_struct["protein_name"] = file_path.stem

    pdb_struct["full"] = defaultdict(SuperOD)
    pdb_struct["CA"] = defaultdict(SuperOD)
    pdb_struct["membrane_coord"] = []
    sequence = ""

    mb_coords = []

    with open(file_path,"r") as f:

        while line := f.readline():

            suspected_membrane_line = False

            if line[:4] == "ATOM":

                # Line format is given in the docstring

                atom_number = int(line[6:11].strip())
                atom_name = line[12:16].strip()

                res_number = int(line[22:26].strip())
                res_name = line[17:20].strip()

                chain_id = line[21]

                x, y, z = map(float, (line[30:38].strip(), line[38:46].strip(), line[46:54].strip()))

                if res_name == "DUM":

                    suspected_membrane_line = True
                    log_messages.append(f"Membrane line found : {line}")

                if not suspected_membrane_line:

                    # Is the chain_id empty ?
                    if not bool(chain_id.strip()):

                        chain_id = "A"

                    if chain_id not in pdb_struct["full"]:

                        pdb_struct["full"][chain_id] = SuperOD()
                        pdb_struct["CA"][chain_id] = SuperOD()

                        if res_number not in pdb_struct["full"][chain_id]:

                            pdb_struct["full"][chain_id][res_number] = SuperOD()

                            pdb_struct["full"][chain_id][res_number][atom_number] = line

                        else:

                            pdb_struct["full"][chain_id][res_number][atom_number] = line

                        log_messages.append(f"Chain {chain_id} found")


                    else:

                        if res_number not in pdb_struct["full"][chain_id]:

                            pdb_struct["full"][chain_id][res_number] = SuperOD()

                            pdb_struct["full"][chain_id][res_number][atom_number] = line

                        else:

                            pdb_struct["full"][chain_id][res_number][atom_number] =line
            
                
                        if atom_name == "CA":

                            # pdb_strcut["CA"][chain_id] has been 
                            # declared as a dict previously
                            pdb_struct["CA"][chain_id][res_number] = {

                                "coord" : [x,y,z],
                                "res_name" : aa_dict[res_name],
                                "res_number" : res_number,
                            }

                            sequence += aa_dict[res_name]

            elif (line[:6] == "HETATM" and "DUM" in line): # or suspected_membrane_line | how many cases fr ? not worth

                x, y, z = map(float, (line[30:38].strip(), line[38:46].strip(), line[46:54].strip()))

                mb_coords.append([x,y,z])

    pdb_struct["sequence"] = sequence
    pdb_struct["membrane_coord"] = np.array(mb_coords)
    pdb_struct["protein_length"] = { }

    # For each chain, compute the length.
    # The pdbs are never evenly formated, thus we can't 
    # infer the length from the residues numbers
    # We just count the number of CA atoms for each chain
    for chain_id in pdb_struct["CA"]:
       
        pdb_struct["protein_length"][chain_id] = len(pdb_struct["CA"][chain_id])

    if secondary_structure_path:

        with open(secondary_structure_path, "r") as f:
            
            ss_dict = defaultdict(dict)

            log_messages.append("Secondary structure found")

            # Line format given by extract_SS.cpp 
            # chain_id << '\t' << res_number << '\t' << secondary_structure << '\n';
            while line := f.readline():

                line = line.split()
                chain_id = line[0]
                res_number = int(line[1])
                secondary_structure = line[2]

                log_messages.append(f"Ss : {secondary_structure}\nChain_id : {chain_id}\nRes number : {res_number}")
                
                ss_dict[chain_id][res_number] = secondary_structure

            
        for chain_id in pdb_struct["CA"]: 
            
            for res_number in pdb_struct["CA"][chain_id]:

                secondary_structure = ss_dict.get(chain_id, {}).get(res_number, "O")
                    
                pdb_struct["CA"][chain_id][res_number].update({"secondary_structure" : secondary_structure})

                if secondary_structure == "H" or secondary_structure == "S":

                    pdb_struct["CA"][chain_id][res_number].update({"folded" : True})
                    log_messages.append(f"Res {res_number} in chain {chain_id} is in a folded region")

                else:

                    pdb_struct["CA"][chain_id][res_number].update({"folded" : False})
    
    log_messages.append(f"{pdb_struct['protein_name']} processed")
    log_messages.append(f"Chains found : {pdb_struct['CA'].keys()}")
    log_messages.append(f"Pdb structure : {pdb_struct}")
    
    if verbose:
        
        logger.info("\n".join(log_messages))

    return pdb_struct


#### TRANS-MEMBRANE PROTEINS ####

def binarize_transmembrane(logger, pdb_struct: dict, margin=10, inner_margin = 0, verbose = False) -> dict:

    """
    This function returns a binary string for each chain in the pdb_struct dictionary.
    The binary string is 1 if the residue is in the membrane +/- a margin for z coordinates
    , 0 otherwise. The margin parameter is in angstroms.
    """
    
    log_messages = []
    
    min_z_membrane = np.min(pdb_struct["membrane_coord"][:, 2]) 
    max_z_membrane = np.max(pdb_struct["membrane_coord"][:, 2]) 

    min_x_membrane = np.min(pdb_struct["membrane_coord"][:, 0])
    max_x_membrane = np.max(pdb_struct["membrane_coord"][:, 0])

    min_y_membrane = np.min(pdb_struct["membrane_coord"][:, 1])
    max_y_membrane = np.max(pdb_struct["membrane_coord"][:, 1])

    in_membrane_binaries = { chain_id: "" for chain_id in pdb_struct["CA"].keys() }
    in_margin_binaries = { chain_id: "" for chain_id in pdb_struct["CA"].keys() }
    
    for chain_id, residues in pdb_struct["CA"].items():

        log_messages.append(f"Binarizing chain {chain_id} ...")

        last_res_number = sorted(residues.keys())[0]
        log_messages.append(f"First residue number : {last_res_number}")

        last_res_number -= 1 # To pass the test for the condition int(res_number) != last_res_number + 1:
        for current_res_number in sorted(residues.keys()): # Some PDBs are not well formated regarding the order of the residues, sort em to be sure
            data = residues[current_res_number]

            x = data["coord"][0]
            y = data["coord"][1]
            z = data["coord"][2]

            # Sometimes, residues are missing from the 3D structure and we don't have their coordinates
            # We can monitor this by checking if the current residue number is not the previous residue number + 1

            if int(current_res_number) != last_res_number + 1:

                # Fill in the gaps with zeros, by default
                in_membrane_binaries[chain_id] += "0" * (int(current_res_number) - last_res_number - 1)
                in_margin_binaries[chain_id] += "0" * (int(current_res_number) - last_res_number - 1)
                log_messages.append(f"Found missing amino acids from {last_res_number} to {current_res_number}")

            # Compute the binary string for the current residue
            is_x = min_x_membrane <= x <= max_x_membrane
            is_y = min_y_membrane <= y <= max_y_membrane
            is_z = min_z_membrane + inner_margin <= z <= max_z_membrane - inner_margin
            is_z_plus_margin = min_z_membrane - margin <= z <= max_z_membrane + margin

            in_membrane = "1" if is_x and is_y and is_z else "0"
            in_margin = "1" if is_x and is_y and is_z_plus_margin else "0"

            # Update the pdb_struct dict on the fly 
            pdb_struct["CA"][chain_id][current_res_number].update({"in_membrane": in_membrane, "in_margin": in_margin})

            in_membrane_binaries[chain_id] += in_membrane
            in_margin_binaries[chain_id] += in_margin

            last_res_number = int(current_res_number)
            
        log_messages.append(f"Binary membrane for chain {chain_id} : {in_membrane_binaries[chain_id]}")
        log_messages.append(f"Binary margin for chain {chain_id} : {in_margin_binaries[chain_id]}")
        
    if verbose:
        
        logger.info("\n".join(log_messages))


    return in_membrane_binaries, in_margin_binaries

def define_tm_segments(logger, binary_dict : dict, pdb_struct : dict, verbose = False):

    """
    Search for transmembrane segments with streches of 1s of at least 15 residues 
    """
    
    log_messages = []

    tm_indices = {chain_id : [] for chain_id in binary_dict.keys() }

    log_messages.append("Searching for transmembrane segments ...")
    log_messages.append(f"Chains in the input dict : {binary_dict.keys()}")

    start_index = None

    for chain_id in binary_dict:

        first_residue = list(pdb_struct["CA"][chain_id].keys())[0]

        for i, bit in enumerate(binary_dict[chain_id]):
            if bit == "1":
                if start_index is None:
                    start_index = i+1
            else:
                if start_index is not None:
                    length = i-start_index+1
                    if length >= 10: # minimum length of a TM segment, although 20 is the length of a typical alpha helical TM segment
                        
                        # python indices are 0-based, so we add 1 to match 
                        # the 1-based residue numbering in the PDB file
                        tm_indices[chain_id].append((start_index, i+1, length))

                    # Wether the segment is long enough or not, we reset the start_index
                    start_index = None

        # If we ended in the middle of a TM segment, we add it to the list
        if start_index is not None:
            length = len(binary_dict[chain_id]) - start_index + 1

            if length >= 15:
                tm_indices[chain_id].append((start_index, len(binary_dict[chain_id]), length))

        # Because the first res is not always 1, we add the first residue number to the segments
        # to retrieve the original coordinates in the pdb_struct dict
        for i, (start, end, length) in enumerate(tm_indices[chain_id]):
            tm_indices[chain_id][i] = (first_residue + start - 1, first_residue + end - 1, length)

        log_messages.append(f"Tm indices for chain {chain_id} : {tm_indices[chain_id]}")
    
    if verbose:
        
        logger.info("\n".join(log_messages))

    return tm_indices

def elongate_tm_segments(logger, tm_indices, pdb_struct, iorf_path, iorf_csv, min_length=20, max_length=70, verbose = False) -> int:

    """
    This function elongates transmembrane segments to a random size drawn from a given size distribution,
    distributing the elongation both upstream and downstream without considering overlapping boundaries between segments.

    Parameters:
    - tm_indices (dict): Dictionary where keys are chain_ids and values are lists of tuples (start, end, length)
      for transmembrane segments.
    - pdb_struct (dict): Dictionary containing protein-related information such as protein lengths.
    - min_length (int): Minimum length of the elongated segment.
    - max_length (int): Maximum length of the elongated segment.

    Returns:
    - int: Function completion status.
    """
    
    log_messages = []
    
    log_messages.append("Elongating transmembrane segments ...")
    log_messages.append(f"Input : {tm_indices.keys()}")

    for chain_id in tm_indices:

        # Draw the random elongation lengts once, as it needs to read a fasta file each time
        desired_lengths = size_picker_v3(fasta_file = iorf_path, precomputed_file = iorf_csv, min_length=min_length, max_length=max_length, n_samples=len(tm_indices[chain_id]))
        chain_length = pdb_struct["protein_length"][chain_id]

        min_chain = min([ aa_num for aa_num in pdb_struct["CA"][chain_id].keys() ])
        max_chain = max([ aa_num for aa_num in pdb_struct["CA"][chain_id].keys() ])

        for i, (current_start, current_end, length_current) in enumerate(tm_indices[chain_id]):

            desired_length = desired_lengths[i]
            
            log_messages.append(f"Chain {chain_id}, segment {i} : {current_start} to {current_end}")

            if desired_length <= length_current:

                tm_indices[chain_id][i] = (current_start, current_end, current_start, current_end)
                log_messages.append(f"Chain {chain_id}, segment {i} : no elongation needed")

            else:
                
                elongation_needed = desired_length - length_current

                # Randomly determine the amount of elongation downstream and upstream ( Nter and Cter )
                downstream = random.randint(0, elongation_needed)
                upstream = elongation_needed - downstream
                # Calculate new coordinates
                # Out of bound new coordinates will be treated as Xs in the future 
                # sequence and will be removed by trimming
                new_end_coordinates = min(max_chain, current_end + downstream)
                new_start_coordinates = max(min_chain, current_start - upstream)
                log_messages.append(
                    
                            f"Old start to old end : {current_start} to {current_end}\n"
                            f"Desired length: {desired_length}\n"
                            f"Current length: {length_current}\n"
                            f"Elongation needed: {elongation_needed}\n"
                            f"Upstream elongation: {upstream}\n"
                            f"Downstream elongation: {downstream}\n" 
                            f"new start variables ( upstream ) : current start {current_start} - upstream {upstream} => {current_start - upstream}\n"
                            f"new end variables  ( downstream ): min(current end {current_end} + downstream {downstream}, chain length {chain_length}) => {min(current_end + downstream, chain_length)}\n"
                            f"new start result : {new_start_coordinates}\n"
                            f"new end result : {new_end_coordinates}\n")
                
                if current_start < new_start_coordinates:

                    logger.info("\n".join(log_messages)) 
                    raise ValueError(f"Current start {current_start} is lesser than new start {new_start_coordinates}")
                    
                
                if current_end > new_end_coordinates:

                    logger.info("\n".join(log_messages)) 
                    raise ValueError(f"Current end {current_end} is greater than new end {new_end_coordinates}")

                # Update the segment information with new start and end positions
                tm_indices[chain_id][i] = (new_start_coordinates, new_end_coordinates, current_start, current_end)

    if verbose:
        
        logger.info("\n".join(log_messages))

    return 0

def extract_elongated_sequences_v2(logger, tm_indices : dict, pdb_struct : dict, gaps : int = 1, verbose = False) -> dict:

    """
    This version of the function does not follow the start -> coordinates and check for buffers.
    Instead, we start from the old start ( membrane segment limitation ) and go backward to the 
    new start ( elongated segment limitation ) once we reach the margin, we stop this step, 
    reverse the accumulated sequence.
    We then normally iterate from old start to old end, and then from old end to new end until 
    we reach the margin again.

    Remainder of the structure of pdb_struct for the full coordinates : 

    pdb_struct["full"][chain_id][res_number][atom_number] = text_line_from_pdb
                    
    """
    
    log_messages = []

    records = []
    records_shorts = []
    structures = { chain_id : {} for chain_id in tm_indices }
    structures_shorts = { chain_id : {} for chain_id in tm_indices }
    protein_name = pdb_struct["protein_name"]

    log_messages.append(f"Extracting elongated sequences ... \nShape of input : {tm_indices.keys()}")
    
    # Iterate per chain
    for chain_id in tm_indices:

        log_messages.append(f"Iterating over chain {chain_id} indices : ")
        log_messages.append(f"Indices : {tm_indices[chain_id]}")

        full = { i : {} for i, _ in enumerate(tm_indices[chain_id]) }
        full_short = { i : {} for i, _ in enumerate(tm_indices[chain_id]) }

        # Iterate per segment for the current chain
        for i, (start, end, old_start, old_end) in enumerate(tm_indices[chain_id]):

            sequence = ""
            sequence_short = ""
            
            # Go backward from the beginning of the segment in the membrane 
            # To the beginning of the elongated segment
            for res_number in range(old_start-1, start, -1):

                if res_number in pdb_struct["CA"][chain_id]:
                    
                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    # Store all info for atoms of this residue 
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    if pdb_struct["CA"][chain_id][res_number]["in_margin"] == "0":

                        break

                    sequence += res
                    full[i].update({ res_number : buffer })

                else:

                    # Elongate until an unknown residue is found
                    break

            # We extracted the sequence backward, we need to reverse it
            sequence = sequence[::-1]

            # Iterate from the beginning of the original segment to the end of the original segment
            for res_number in range(old_start, old_end+1):

                if res_number in pdb_struct["CA"][chain_id]:

                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    sequence += res
                    full[i].update({ res_number : buffer })
                    sequence_short += res
                    full_short[i].update({ res_number : buffer })

                else:

                    sequence += "X"
                    full[i].update({ res_number : "X" })
                    sequence_short += "X"
                    full_short[i].update({ res_number : "X" })

            # Iterate from the end of the original segment to the end of the elongated segment
            for res_number in range(old_end+1, end+1):

                if res_number in pdb_struct["CA"][chain_id]:

                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    if pdb_struct["CA"][chain_id][res_number]["in_margin"] == "0":

                        break

                    sequence += res
                    full[i].update({ res_number : buffer })

                else:

                    break

            sequence = sequence.strip("X")
            sequence_short = sequence_short.strip("X")

            if (not "X" * gaps in sequence) and (sequence != ""):
            
                record = SeqRecord(Seq(sequence), id=f"{protein_name}_{chain_id}_{i+1}_elong", description=f"")
                records.append(record)
                structures[chain_id] = full

            if (not "X" * gaps in sequence_short) and (sequence_short != ""):

                record = SeqRecord(Seq(sequence_short), id=f"{protein_name}_{chain_id}_{i+1}_short", description=f"")
                records_shorts.append(record)
                structures_shorts[chain_id] = full_short


    if verbose:
        
        logger.info("\n".join(log_messages))

    return { 
            "sequences" : records, 
            "sequences_shorts" : records_shorts, 
            "structures" : structures, 
            "structures_shorts" : structures_shorts 
        }
    
def strip_dict(d):

    keys = list(d.keys())
    for key in keys:
        if d[key] == "X":
            del d[key]
        else:
            break
    for key in reversed(keys):
        if key in d and d[key] == "X":
            del d[key]
        else:
            break


def extract_elongated_sequences_v3(logger, tm_indices : dict, pdb_struct : dict, gaps : int = 1, verbose = False) -> dict:

    log_messages = []

    records = []
    records_shorts = []
    structures = { chain_id : {} for chain_id in tm_indices }
    structures_shorts = { chain_id : {} for chain_id in tm_indices }
    protein_name = pdb_struct["protein_name"]

    log_messages.append(f"Extracting elongated sequences ... \nShape of input : {tm_indices.keys()}")
    
    # Iterate per chain
    for chain_id in tm_indices:

        log_messages.append(f"Iterating over chain {chain_id} indices : ")
        log_messages.append(f"Indices : {tm_indices[chain_id]}")

        num_segments = len(tm_indices[chain_id])

        # Iterate per segment for the current chain
        for i, (start, end, old_start, old_end) in enumerate(tm_indices[chain_id]):

            if i == 0:
                position = 'first'
            elif i == num_segments - 1:
                position = 'last'
            else:
                # Not first nor last, we compute the place of the segment in the structure pondered by the number of segments
                position = f"{(i+1)/num_segments}"

            sequence = ""
            sequence_short = ""

            fasta_key = f"{protein_name}_{chain_id}_{i+1}_elong"
            fasta_key_short = f"{protein_name}_{chain_id}_{i+1}_short"
            
            full = {}
            full_short = {}

            # Go backward from the beginning of the segment in the membrane 
            # To the beginning of the elongated segment
            for res_number in range(old_start-1, start, -1):

                if res_number in pdb_struct["CA"][chain_id]:
                    
                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    # Store all info for atoms of this residue 
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    if pdb_struct["CA"][chain_id][res_number]["in_margin"] == "0":
                        break

                    sequence += res
                    full[res_number] = buffer

                else:
                    # Elongate until an unknown residue is found
                    break

            # We extracted the sequence backward, we need to reverse it
            sequence = sequence[::-1]

            # Iterate from the beginning of the original segment to the end of the original segment
            for res_number in range(old_start, old_end+1):

                if res_number in pdb_struct["CA"][chain_id]:

                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    sequence += res
                    full[res_number] = buffer
                    sequence_short += res
                    full_short[res_number] = buffer

                else:
                    sequence += "X"
                    full[res_number] = "X"
                    sequence_short += "X"
                    full_short[res_number] = "X"

            # Iterate from the end of the original segment to the end of the elongated segment
            for res_number in range(old_end+1, end+1):

                if res_number in pdb_struct["CA"][chain_id]:

                    res = pdb_struct["CA"][chain_id][res_number]["res_name"]
                    buffer = { atom_number : data for atom_number, data in pdb_struct["full"][chain_id][res_number].items() }

                    if pdb_struct["CA"][chain_id][res_number]["in_margin"] == "0":
                        break

                    sequence += res
                    full[res_number] = buffer

                else:
                    break

            sequence = sequence.strip("X")
            sequence_short = sequence_short.strip("X")

            strip_dict(full)
            strip_dict(full_short)

            if ("X" * gaps not in sequence) and (sequence != ""):
                record = SeqRecord(Seq(sequence), id=fasta_key, description=f"")
                records.append(record)
                structures[chain_id][fasta_key] = full

            if ("X" * gaps not in sequence_short) and (sequence_short != ""):
                record_short = SeqRecord(Seq(sequence_short), id=fasta_key_short, description=f"")
                records_shorts.append(record_short)
                structures_shorts[chain_id][fasta_key_short] = full_short

    if verbose:
        logger.info("\n".join(log_messages))

    return { 
            "sequences" : records, 
            "sequences_shorts" : records_shorts, 
            "structures" : structures, 
            "structures_shorts" : structures_shorts 
        }


    

                    
#### PERIPHERAL PROTEINS ####

def binarize_peripheral(logger, pdb_struct, close_margin, outer_margin, verbose = False) -> dict:
    
    log_messages = []

    near_membrane_binaries = { chain_id: "" for chain_id in pdb_struct["CA"].keys() }

    log_messages.append(f"Processing {pdb_struct['protein_name']} ...")
    log_messages.append(f"Chains : {pdb_struct['CA'].keys()}")
    
    for chain_id, residues in pdb_struct["CA"].items():

        Z = [ float(pdb_struct["CA"][chain_id][res_number]["coord"][2]) for res_number in pdb_struct["CA"][chain_id] ]

        z_median = np.median(Z)

        z_membrane = pdb_struct["membrane_coord"][:, 2]
        
        min_z_membrane = np.min(z_membrane)
        max_z_membrane = np.max(z_membrane)

        log_messages.append(f"Binarizing chain {chain_id} ...")
        last_res_number = list(residues.keys())[0] -1
        log_messages.append(f"First residue number : {last_res_number}")

        # The proteins are outside the membrane
        # Are they on the cytolic side or the extracellular side ?
        # We check using the mean of the z coordinates of the protein
        # compared to the min and max z coordinates of the membrane
        if min_z_membrane > z_median: 

            for current_res_number, data in residues.items():

                z = data["coord"][2]

                # Sometimes, residues are missing from the 3D structure and we don't have their coordinates
                # We can monitor this by checking if the current residue number is not the previous residue number + 1

                if int(current_res_number) != last_res_number + 1:

                    # Fill in the gaps with zeros, by default
                    near_membrane_binaries[chain_id] += "0" * (int(current_res_number) - last_res_number - 1)

                z = float(z)
                min_z_membrane = float(min_z_membrane)
                close_margin = float(close_margin) 
                outer_margin = float(outer_margin)
                    
                is_z_near_membrane = z > (min_z_membrane - close_margin)
                is_z_in_margin = z > (min_z_membrane - outer_margin)

                near_mebrane = "1" if is_z_near_membrane else "0"
                in_margin = "1" if is_z_in_margin else "0"

                # If residue is in membrane, it should be in the margin
                if near_mebrane == "1" and in_margin != "1":
                    logger.error("in_margin must be True if near_membrane is True")
                    raise ValueError("Check logger")
                
                pdb_struct["CA"][chain_id][current_res_number].update({"near_membrane": near_mebrane, "in_margin": in_margin})

                near_membrane_binaries[chain_id] += near_mebrane

                last_res_number = int(current_res_number)

        if min_z_membrane < z_median:

            for current_res_number, data in residues.items():

                z = np.float32(data["coord"][2])

                # Sometimes, residues are missing from the 3D structure and we don't have their coordinates
                # We can monitor this by checking if the current residue number is not the previous residue number + 1

                if int(current_res_number) != last_res_number + 1:

                    # Fill in the gaps with zeros, by default
                    near_membrane_binaries[chain_id] += "0" * (int(current_res_number) - last_res_number - 1)

                z = float(z)
                max_z_membrane = float(max_z_membrane)
                close_margin = float(close_margin) 
                outer_margin = float(outer_margin)

                is_z_near_membrane = z < (max_z_membrane + close_margin)
                is_z_in_margin = z < (max_z_membrane + outer_margin)

                near_membrane = "1" if is_z_near_membrane else "0"
                in_margin = "1" if is_z_in_margin else "0"

                # If residue is in membrane, it should be in the margin
                if near_membrane == "1" and in_margin != "1":
                    logger.error("in_margin must be True if near_mebrane is True")
                    raise ValueError("Check logger")
                
                pdb_struct["CA"][chain_id][current_res_number].update({"near_membrane": near_membrane, "in_margin": in_margin})

                near_membrane_binaries[chain_id] += near_membrane

                last_res_number = int(current_res_number)
                
        log_messages.append(f"Binary membrane for chain {chain_id} : {near_membrane_binaries[chain_id]}")
        
    if verbose:
        
        logger.info("\n".join(log_messages))
    

    return near_membrane_binaries

def search_peripheral_segments(logger, chain_binaries, min_segment_length) -> dict:
    
    """
    Searches stretches of 1s in the binary strings of the chains.

    Parameters:
        - chain_binaries (dict): Dictionary where keys are chain_ids and values are binary strings
        - min_length (int): Minimum length of the peripheral segment
    
    Returns:
        - dict: Dictionary where keys are chain_ids and values are lists of tuples (start, end) 
                representing the peripheral segments to extract. One tuple per peripheral segment
    """

    segments = {}
    
    pattern = re.compile("1{" + str(min_segment_length) + ",}")

    for chain_id, binary_string in chain_binaries.items():

        matches = [(match.start(), match.end()) for match in re.finditer(pattern, binary_string)]
        segments[chain_id] = matches

    return segments

def elongate_peripheral_segments(logger, segments, iorf_path = None, iorf_csv = None, min_length = 20, max_length = 70, verbose = False):
    
    """
    
    For each peripheral segment of the protein, check draw a length of the distribution given by iorfs csv or fasta
    Elongate the segment on N-ter and C-ter based on the drawn length  
    
    """
    
    log_messages = []

    for chain_id in segments:

        n_segments = len(segments[chain_id])

        if n_segments != 0:

            desired_lengths = size_picker_v3(fasta_file = iorf_path, precomputed_file = iorf_csv, min_length = min_length, max_length = max_length, n_samples = n_segments )
            
            for i, (start, end) in enumerate(segments[chain_id]):

                desired_length = desired_lengths[i]

                if desired_length <= (end - start):

                    segments[chain_id][i] = (start, end, start, end)

                else:

                    elongation_needed = desired_length - (end - start)

                    downstream = random.randint(0, elongation_needed)
                    upstream = elongation_needed - downstream

                    new_end = end + downstream
                    new_start = start - upstream

                    segments[chain_id][i] = (new_start, new_end, start, end)

    if verbose:
        
        logger.info("\n".join(log_messages))

    return 0 


#### Peptides ####
def compute_orientation(P1, P2):

    """
    Given two points P1 and P2 in 3D space, this function computes the angle
    between the vector P1P2 and the z-axis.
    """

    x1, y1, z1 = P1
    x2, y2, z2 = P2

    H = np.array([x2 - x1, y2 - y1, z2 - z1])

    H_magnitude = np.linalg.norm(H)

    H_z = H[2]

    cos_theta = H_z / H_magnitude

    theta = np.arccos(cos_theta)

    theta_degrees = np.degrees(theta)

    # Pointing up is the same as pointing down
    if theta_degrees > 90:
        theta_degrees = 180 - theta_degrees

    return theta_degrees

def extract_horizontal_alpha_helix(logger, pdb_struct, checking_depth = 3, min_length = 10, gaps = 1, verbose = False) -> SeqRecord:

    """
    This function extracts alpha helices from a PDB structure.
    The presupposition is that every pdb_struct only has one short peptide.
    The pdb_struct must have the secondary structure information.

    The secondary structure is given by the dssp v4.0 program.
    Sometimes, it can be a bit "strict", so if a residue is not predicted 
    as an alpha helix component, but is surrounded by residues in +1/-1 
    and or +2/-2 that are, we can consider it as an alpha helix residue.

    Given that some peptides are composed of several alpha helices, but 
    only one is close to the membrane, we will on the fly store the 
    start/end of every alpha helix found and only keep the one that is 
    /!\
    the most horizontal || the closest to the membrane idk yet ?? 
    /!\
    """
    
    log_messages = []

    helices = []
    protein_name = pdb_struct["protein_name"]
    start, end = None, None

    log_messages.append(f"Processing {protein_name}")

    for chain_id in pdb_struct["CA"]:

        sequence = ""

        for res_number in pdb_struct["CA"][chain_id]:

            res = pdb_struct["CA"][chain_id][res_number]["res_name"]

            ss = pdb_struct["CA"][chain_id][res_number]["secondary_structure"]

            if ss == "H":

                
                if "".__eq__(sequence):
                    start = res_number

                sequence += res

            else:

                if "".__eq__(sequence):

                    continue
                
                # Now we enter in the surrounding residues checking
                else:

                    minus_one = pdb_struct["CA"][chain_id].get(res_number-1, {}).get("secondary_structure") == "H"

                    if minus_one:

                        next_aa_are_helix = False

                        for i in range(1, checking_depth + 1):
                            if pdb_struct["CA"][chain_id].get(res_number + i, {}).get("secondary_structure") == "H":
                                next_aa_are_helix =  True
                                break

                        if next_aa_are_helix:

                            sequence += res
                            pdb_struct["CA"][chain_id][res_number].update({"secondary_structure" : "H"})

                        # We reached the end of an alpha helix
                        else:

                            end = res_number
                            if len(sequence) >= min_length and not "X" * gaps in sequence:
                                helices.append((start, end, sequence))
                            sequence = ""

        if len(helices) == 1:

            record = SeqRecord(Seq(helices[0][2]), id=f"{protein_name}", description="")
            log_messages.append(f"Only one sequence extracted : {record}")

        # Search for the most horizontal and closest to the membrane alpha helix
        elif len(helices) > 1:

            log_messages.append(f"Several sequences found : {helices}")

            for i, (start, end, sequence) in enumerate(helices):

                # Compute the angle between the alpha helix and the z-axis
                # The most horizontal alpha helix is the one with the smallest angle
                # The closest to the membrane is the one with the smallest z coordinate
                # of the first residue of the alpha helix

                P1 = pdb_struct["CA"][chain_id][start]["coord"]
                P2 = pdb_struct["CA"][chain_id][end]["coord"]

                theta = compute_orientation(P1, P2)

                helices[i] = (start, end, sequence, theta)

            # Only the most horizontal alpha helix is kept
            helices.sort(key=lambda x: x[3], reverse=True)
            record = SeqRecord(Seq(helices[0][2]), id=f"{protein_name}", description="")

        else:
                
            log_messages.append(f"No alpha helix found in {protein_name}")
            return None
        
    if verbose: 
        
        logger.info("\n".join(log_messages))

    return record


if __name__ == '__main__':

    pass
    



