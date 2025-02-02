#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program reads a first FASTA and store the length distribution of its sequences.
Then it reads a second FASTA and cut its sequences based on the length distribution of the first FASTA.
You can also provided a maximum length for the cut sequences.
"""



def main(first_fasta, second_fasta, output_fasta, min_length=0, max_length=1000000000):

    """
    Main function that reads the first and second input FASTA files, and writes the cut sequences to the output FASTA file.
    """
    # Get the length distribution from the first FASTA file
    length_distribution = get_length_distribution(first_fasta)
    length_distribution = [l for l in length_distribution if l >= min_length and l <= max_length]

    shortest = min(length_distribution)
    # Cut sequences in the second FASTA file based on the adjusted_distribution and write them to the output FASTA file
    cut_fasta(second_fasta, output_fasta, length_distribution, shortest)


