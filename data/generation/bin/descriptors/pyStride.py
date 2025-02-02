#!/usr/local/bin/python3

import subprocess


def _parse_output(stdout) -> dict:
    
    """
    Parse an ASG line from STRIDE
    
     ASG Detailed secondary structure assignment

    Format:  
    6-8  Residue name
    10-10 Protein chain identifier
    12-15 PDB	residue	number
    17-20 Ordinal residue number
    25-25 One	letter secondary structure code	**)
    27-39 Full secondary structure name
    43-49 Phi	angle
    53-59 Psi	angle
    65-69 Residue solvent accessible area

    Input : 
    
        - stdout of stride
        
    Output : 
    
        - Dict of secondary structure assignment for each AA for each chain.
        
    """
    
    
    
    _ =  [ line for line in stdout.split('\n') if line.startswith('ASG') ]
    
    _dict = {}
    unique_chains = set()
    for line in _:
        
        chain_id = line[9].strip()

        unique_chains.add(chain_id)
        if len(unique_chains) > 1:
            raise ValueError("Multiple chains are not supported")
        res_number = int(line[11:15])
        secondary_structure = line[24]
        
        _dict[res_number] = secondary_structure

    return _dict
        
    
    
def stride(pdbpath, output_type = "percentage") -> dict:
    
    """
    Perform STRIDE analysis

    One-letter secondary structure code is nearly the same as used  in
	DSSP [2] (see Frishman and Argos [1] for details):

	   H	    Alpha helix
	   G	    3-10 helix
	   I	    PI-helix
	   E	    Extended conformation
	   B or	b   Isolated bridge
	   T	    Turn
	   C	    Coil (none of the above)

    Parameters
    ----------
    pdbpath : str
      path to the pdb file to analyse

    Returns
    -------
    output : float
      Percentage of folded residues 
    """

    assert output_type in ["percentage", "full"], "Output type must be either 'percentage' or 'full'"

    folded_letters = "HGIEBb"
    p = subprocess.Popen(['stride', pdbpath],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    try:
        stdout, stderr = p.communicate()
        output = stdout.decode(encoding = "utf-8")
        _dict = _parse_output(output) 

    except Exception as e:
        return None
    
    if len(_dict) == 0:

        if output_type == "percentage":
            return 0.0
        else:
            return None
        
    if output_type == "percentage":
        return len([ _ for _ in _dict.values() if _ in folded_letters]) / len(_dict)  # Return the percentage of folded residues

    else:
            
        return _dict


if __name__ == "__main__": 
    
    import argparse
    import glob
    import pathlib
    from tqdm import tqdm
    import polars as pl 
    
    parser = argparse.ArgumentParser(
                    prog='PyStride'
    )
    
    parser.add_argument("--input", type = str, required = True,
                        help = "PDB directory or file")
    parser.add_argument("--output", type = str, required = True,
                        help = "Output file")
    
    
    args = parser.parse_args()

    FOLDED = "HGIEBb"

    A_HELIX = "HGI"

    B_STRAND = "EBb"














    
    



