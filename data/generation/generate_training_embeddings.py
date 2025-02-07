#!/usr/bin/env python3
# -*- coding: utf-8 -*-


print("================ Loading Libraries =================")

import time
from tqdm import tqdm 

import torch
from transformers import T5EncoderModel, T5Tokenizer

import polars as pl 

class ConvNet( torch.nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )

        self.dssp8_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )


    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x         = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat   = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat   = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(  x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat

def load_sec_struct_model(device):
    """
    Downloads and loads the secondary structure prediction model.
    
    wget -nc -P protT5/sec_struct_checkpoint http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt
    """

    checkpoint_dir = "./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"

    try:
        state = torch.load(checkpoint_dir, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_dir}. ")

    model = ConvNet()
    model.load_state_dict(state['state_dict'])

    model.eval() 
    model.to(device)  

    epoch = state.get('epoch', None)
    if epoch is not None:
        print(f"✅ Loaded secondary structure model from epoch: {epoch:.1f}")
    else:
        print("✅ Loaded secondary structure model.")

    return model

def write_prediction_fasta(predictions, out_path):
    class_mapping = {0:"H",1:"E",2:"L"}
    with open(out_path, 'w+') as out_f:
        out_f.write( '\n'.join(
            [ ">{}\n{}".format(
                seq_id, ''.join( [class_mapping[j] for j in yhat] ))
            for seq_id, yhat in predictions.items()
            ]
            ) )
    return None

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

def get_embeddings(device : torch.device, seqs : dict, per_residue : bool, per_protein: bool, sec_struct : bool,
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict()
               }


    
    print("Parameters for inference : ")
    print(f"        Max residues: {max_residues}, Max sequence length: {max_seq_len}, Max batch size: {max_batch}")
    print(f"        Per-residue embeddings: {per_residue}")
    print(f"        Per-protein embeddings: {per_protein}")
    print(f"        Secondary structure prediction: {sec_struct}")

    model, tokenizer = get_T5_model(device)
    
    if sec_struct:
        print("🛠 Loading secondary structure model...")
        sec_struct_model = load_sec_struct_model(device)



    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1), desc = "Generating embeddings", total=len(seq_dict)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        # Is the batch full ? If yes, forward pass
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: 
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
               
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # secondary structure predictions
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1x1024)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

            
    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')

    return results

def main(): 

    train = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train_sequences.csv", separator = ",")
    test = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/test_sequences.csv", separator = ",")

    print("================ Checking CUDA  =================")
    assert torch.cuda.is_available(), "Error : CUDA not detected"
    assert torch.cuda.device_count() > 0, "Error : No CUDA capable device found"
    print("================ CUDA ok ! =================")

    device = torch.device("cuda")
    model, tokenizer = get_T5_model(device)

    per_protein = True
    per_residue = True
    sec_struct = False

    residue_emb = {}
    prot_emb = {}
    for category, data in train.group_by("category"): 

        seqs = [ "".join(s.split()) for s in data["sequence"].to_list() ]
        ids = data["id"].to_list()

        seq_dict = { k : v for k,v in zip(ids, seqs) }

        embeddings = get_embeddings(device, seq_dict, max_batch=1024, max_residues=5000,
                                    per_protein=per_protein, per_residue=per_residue, sec_struct=sec_struct)
        residue_emb.update({k: {"embedding": v, "class_type": category[0]} for k,v in embeddings["residue_embs"].items()})
        prot_emb.update({k: {"embedding": v, "class_type": category[0]} for k,v in embeddings["protein_embs"].items()})

        

    print(f"Number of residue embeddings : {len(residue_emb)}")
    torch.save(obj=residue_emb, f="../training_dataset/trainset_residue_embeddings.pt")
    torch.save(obj=prot_emb, f="../training_dataset/trainset_protein_embeddings.pt")

    residue_emb = {}
    prot_emb = {}
    for category, data in test.group_by("category"): 

        seqs = [ "".join(s.split()) for s in data["sequence"].to_list() ]
        ids = data["id"].to_list()

        seq_dict = { k : v for k,v in zip(ids, seqs) }

        embeddings = get_embeddings(device, seq_dict, max_batch=1024, max_residues=5000,
                                    per_protein=per_protein, per_residue=per_residue, sec_struct=sec_struct)
        residue_emb.update({k: {"embedding": v, "class_type": category[0]} for k,v in embeddings["residue_embs"].items()})
        prot_emb.update({k: {"embedding": v, "class_type": category[0]} for k,v in embeddings["protein_embs"].items()})

    print(f"Number of residue embeddings : {len(residue_emb)}")
    torch.save(obj=residue_emb, f="../training_dataset/testset_residue_embeddings.pt")
    torch.save(obj=prot_emb, f="../training_dataset/testset_protein_embeddings.pt")



if __name__ == "__main__":

    main()