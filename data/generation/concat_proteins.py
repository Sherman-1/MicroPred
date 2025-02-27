from Bio import SeqIO 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import random 
import re

def concatenate_fasta(file_path, group_size=2, seed=None):
    """
    Reads a FASTA file, randomly shuffles sequences, groups them into batches of
    group_size, and concatenates each group into a new sequence.
    
    Parameters:
      - file_path: Path to the FASTA file.
      - group_size: Number of sequences to concatenate (default is 2).
      - seed: Optional seed for random shuffling.
      
    Returns:
      A list of tuples: (new_header, concatenated_sequence)
    """
    records = list(SeqIO.parse(file_path, "fasta"))
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(records)
    
    concatenated_sequences = []
    total = len(records)
    # Only keep groups that are complete ! 
    num_groups = total // group_size
    
    for i in range(num_groups):

        group = records[i*group_size:(i+1)*group_size]
        new_header = "|".join(record.id for record in group)
        concatenated_seq = "".join([str(record.seq) for record in group])
        concatenated_sequences.append(SeqRecord(Seq(concatenated_seq), id=new_header, description=""))
    
    return concatenated_sequences


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output", type=str, default = None, help="Path to the output FASTA file.")
    parser.add_argument("-n", type=int, default=2, help="Number of sequences to concatenate.")
    parser.add_argument("--seed", type=int, default=66, help="Seed for random shuffling.")

    args = parser.parse_args()

    if args.output is None:
        args.output = re.sub(r"\.fasta$", f"_grouped_{args.group_size}.fasta", args.input)

    concatenated_sequences = concatenate_fasta(args.input, args.n, args.seed)
    SeqIO.write(concatenated_sequences, args.output, "fasta")
    print(f"Concatenated sequences saved to {args.output}.")