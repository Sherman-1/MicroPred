from pathlib import Path    

from Bio import SeqIO 
import random




def shuffle_sequences(input_file, output_file):

    print(f"Shuffling sequences in {input_file} and writing to {output_file}")

    records = list(SeqIO.parse(input_file, "fasta"))

    records = {record.id : ''.join(random.sample(str(record.seq),len(record.seq))) for record in records}

    print(f"Writing {len(records)} shuffled sequences to {output_file}")

    with open(output_file, "w") as output_handle:

        for record_id, sequence in records.items():
            output_handle.write(f">{record_id}\n{sequence}\n")


if __name__ == "__main__":

    files = Path("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/processed_fastas").glob("*.fa*")


    for file in files:
        shuffle_sequences(file, file.with_name(file.stem + "_shuffled" + file.suffix))
