#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger
"""

print("================ Loading Libraries =================")

import time
from tqdm import tqdm 

import torch
from transformers import T5EncoderModel, T5Tokenizer

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import polars as pl 

from pathlib import Path
import glob 

def get_T5_model(device):

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    
    model = model.to(device) 
    model = model.eval()
    
    if device==torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    return model, tokenizer


def read_fasta( fasta_path ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case

    bad_keys = list()

    for key in sequences:

        if not set(sequences[key]).issubset(set(AA)):
            bad_keys.append(key)

    for key in bad_keys:

        del sequences[key]
                
    return sequences


def get_embeddings( model, tokenizer, device, seqs, per_residue, per_protein, 
                   max_residues=2000, max_seq_len=1000, max_batch=100 ):

    """
    # per_residue indicates that embeddings for each residue in a protein should be returned.
    # per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
    # max_residues gives the upper limit of residues within one batch
    # max_seq_len gives the upper sequences length for applying batch-processing
    # max_batch gives the upper number of sequences per batch
    """

    results = {
                "residue_embs" : dict(), 
                "protein_embs" : dict(),
                "sec_structs" : dict() 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()

    print("Parameters for prediction : ")
    print(f"    - per_residue : {per_residue}")
    print(f"    - per_protein : {per_protein}")
    print(f"    - max_residues : {max_residues}")
    print(f"    - max_seq_len : {max_seq_len}")
    print(f"    - max_batch : {max_batch}")


    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1), desc = "Generating embeddings", total=len(seq_dict)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:

            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)


            with torch.no_grad():

                # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                embedding_repr = model(input_ids, attention_mask=attention_mask)

            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]

                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')

    return results


def compute(sequences : dict ,max_residues=2000, max_seq_len=1000, max_batch=100, per_protein=True, per_residue=False):

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print("Using device: {}".format(device))
    print("Loading model ...")

    model, tokenizer = get_T5_model(device)

    embeddings = get_embeddings(model, tokenizer, device, sequences, per_residue, per_protein, max_residues=max_residues, max_seq_len=max_seq_len, max_batch=max_batch)

    return embeddings  


def extract_sequences_from_pdbs(directory):

    sequences_dict = {}
    pdb_parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    pdbs = glob(directory)

    for filename in tqdm(pdbs, desc = "Extracting sequences"):

        try:

            path = Path(filename)

            structure = pdb_parser.get_structure(path.stem, filename)
            
            # We only want one model and one chain 
            for model in structure:
                for chain in model:
                    # Extract polypeptides
                    peptides = ppb.build_peptides(chain)
                    for peptide in peptides:
                        sequence = peptide.get_sequence()
                        sequences_dict[path.stem] = str(sequence)  # Store sequence as string

                    break

                break 

        except Exception as e:

            print(f"Error while processing {filename} : {e}")
    
    return sequences_dict 


def main(fasta_path, batch_size, max_residues, max_seq_len):

    sequences = read_fasta(fasta_path)

    max_seq_len = max([len(seq) for seq in sequences.values()])

    embeddings = compute(sequences, max_residues, max_seq_len, batch_size) 

    data = embeddings["protein_embs"]
    
    rows = []

    for identifier, embedding in data.items():
        row = [identifier] + embedding.tolist()
        rows.append(row)

    column_names = ['id'] + [f'embed_{i}' for i in range(len(next(iter(data.values()))))]
    df = pl.DataFrame(rows, schema=column_names)

    return df

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Extract embeddings from protein sequences')

    parser.add_argument('--fasta', type=str, help='Path to fasta file containing protein sequences')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size for processing sequences')
    parser.add_argument('--max_residues', type=int, default = 4000, help='Maximum number of residues per batch')
    parser.add_argument('--max_seq_len', type=int, default = 1000, help='Maximum sequence length for processing')
    parser.add_argument('--output', type=str, help='Name of the data to use for writting the embeddings')
    parser.add_argument('--per_protein', type=bool, help="Pool per amino acid embeddings into single 1024 tensor per protein", default = True)
    parser.add_argument('--per_amino', type = bool, help="Compute per amino acid embeddings", default = False)

    args = parser.parse_args()

    fasta_path = args.fasta
    batch_size = args.batch_size
    max_residues = args.max_residues
    max_seq_len = args.max_seq_len
    output = args.output
    per_protein = args.per_protein
    per_amino = args.per_amino 

    assert (per_amino and per_protein) == False, "Choose at least one type of embeddings to compute"

    filename = f"{output}.parquet"

    if Path(filename).exists():

        val = input(f"File {filename} already exists. Do you wish to overwrite it ? (y/n)")

        if val.lower() == "n":
                
            print("Exiting ...")
            exit()

        elif val.lower() == "y":

            print("Overwriting file ...")
        
        else:

            print("Invalid input. Exiting ...")
            exit()
            
    if per_protein:
        
        df = main(fasta_path, batch_size, max_residues, max_seq_len)
        df.write_parquet(f"{output}.parquet", compression = "zstd", compression_level=22)
