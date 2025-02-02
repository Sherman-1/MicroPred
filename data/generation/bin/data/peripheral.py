#!/usr/bin/env python3

import argparse
import logging 
from Bio import SeqIO

from utils import read_pdb, binarize_peripheral, search_peripheral_segments, elongate_peripheral_segments, extract_elongated_sequences_v3
from utils import setup_logger, exception_catcher

peripheral_logger = setup_logger('peripheral_logger', 'transmembrane.log')

@exception_catcher(logging.getLogger('transmembrane_logger'))
def peripheral(pdb_path, close_margin, outer_margin, min_length, max_length, min_segment_length, iorf_csv, iorf_fasta, gaps, verbose = False):
    
    pdb_struct = read_pdb(peripheral_logger,pdb_path, verbose)

    chain_binaries = binarize_peripheral(peripheral_logger, pdb_struct, close_margin, outer_margin, verbose)

    segments = search_peripheral_segments(peripheral_logger,chain_binaries, min_segment_length)

    elongate_peripheral_segments(peripheral_logger, segments, iorf_fasta, iorf_csv, min_length, max_length, verbose)

    res_dict = extract_elongated_sequences_v3(peripheral_logger,segments, pdb_struct, gaps, verbose)

    return res_dict
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file")
    parser.add_argument("--secondary_structures", help="Secondary structure file", default = None)
    parser.add_argument("--min_length", type = int, help = "Mininimum target size for transmembrane segments elongation", default = 20)
    parser.add_argument("--max_length", type = int, help = "Maximum target size for transmembrane segments elongation", default = 70)
    parser.add_argument("--min_segment_length", type = int,  help = "Minimum length for a segment to be considered", default = 15)
    parser.add_argument("--margin", type=  int, help = "Supplementary distance from the membrane to consider amino acids of interest", default = 10)
    parser.add_argument("--close_margin", type = int,
                        help = "Distance to consider an amino acid 'peripheral' to the membrane", 
                        default = 1.5)
    parser.add_argument("--gaps", type = int, help = "Minimum number of missing AA in a segment to discard it", default = 1)

    parser.add_argument("--output", help="Output file", default = None)
    
    i_opts = parser.add_mutually_exclusive_group(required=True)

    i_opts.add_argument('--fasta', help='File with iORFs in FASTA format', default = None)
    i_opts.add_argument('--csv', help='File with iORFs lengths in CSV format', default = None)


    args = parser.parse_args()
    
    if args.margin < args.close_margin:
        raise ValueError("The margin should be greater than the close margin")
    
    res_dict = peripheral(
        
                args.input, 
                args.close_margin, 
                args.margin, 
                args.min_length, 
                args.max_length, 
                args.min_segment_length,
                args.csv, 
                args.fasta, 
                args.gaps
    )
    
    if res_dict != 1:
        print(res_dict["records"])


if __name__ == "__main__":

    main()