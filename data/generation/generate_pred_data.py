from Bio import SeqIO 
import polars as pl 
from sklearn.model_selection import train_test_split
from bin.descriptors.sequence_descriptors import process_data as get_descriptors, process_data_pool as get_descriptors_pool
from transformers import T5EncoderModel, T5Tokenizer
import torch
import warnings
from tqdm import tqdm 
import time 
import numpy as np

# Better in a yaml file, do it later
fastas = {

    "Swiss_microproteins" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/Uniprot/Swissprot_microproteins.faa",
    "Trembl_microproteins" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/Uniprot/Trembl_microproteins_sample.faa",
    "Scer_iORFs" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/iORFs_ESMFold/Scer_IGORFs.pfasta"

}

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = "cuda"
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    warnings.warn("Warning: Embeddings inference running on CPU!", RuntimeWarning)


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

def embeddings_to_dataframe(results: dict) -> pl.DataFrame:
    """
    Converts dict returned by get_embeddings() to a Polars DF suited for parquet storage.
    Columns: id, protein_emb, residue_emb
    """
    protein_embs = results.get("protein_embs", {})
    residue_embs = results.get("residue_embs", {})

    protein_ids = set(protein_embs.keys())
    residue_ids = set(residue_embs.keys())

    # Check that both dictionaries have exactly the same keys
    assert protein_ids == residue_ids, "IDs don't match between per-residue and per-protein embeddings"

    all_ids = list(protein_ids)  # since they are the same

    ids = []
    prot_list = []
    resid_list = []

    for identifier in tqdm(all_ids, desc="Converting embeddings to dataframe", total=len(all_ids)):
        p_tensor = protein_embs.get(identifier)
        r_tensor = residue_embs.get(identifier)

        ids.append(identifier)
        prot_list.append(p_tensor.tolist() if p_tensor is not None else None)
        resid_list.append(r_tensor.tolist() if r_tensor is not None else None)

    return pl.DataFrame({
        "id": ids,
        "protein_emb": prot_list,
        "residue_emb": resid_list
    })

def return_final_df(seq_descriptors, seq_embeddings, seq_dict):


    print("     Merging dataframes ... ")


    df = (
        seq_descriptors
        .with_columns(
            HCA_score = ( (pl.col("HCA_score") + 10) / 20 ) # x - min / max - min, HCA [-10,10]. Scale to [0,1], every other phychem is already [0,1] 
        )
        .with_columns(

            # Turn seq descriptors into an array like structure for later ( easier dataloading )
            descriptors = pl.concat_list(pl.exclude(["id","category","seq_len"])),
            # Dirty way to keep the names of each descriptors in order ... 
            descriptor_names = seq_descriptors.select(pl.exclude(["id","category","seq_len"])).columns            
        )
        .select(["id","seq_len","descriptors","category","descriptor_names"])
        .join(seq_embeddings, on = "id", how = "inner")
        .join(
            pl.DataFrame({
                "id" : seq_dict.keys(), "sequence" : seq_dict.values()
            }), on = "id", how = "inner"
        )
        .with_columns(

            # Flag unorthodox residues, remove trailing stops and space the characters for the tokenizer
            sequence = (
                pl.col("sequence")
                .str.replace_all(r"[UBOZ\.]", "X")
                .str.replace_all(r"\*$", "")
                .str.split("")   # Convert each sequence to list of characters
                .list.join(" ")  # Join characters of lists with whitespaces for pT5 tokenizer expected format !
            )
        )
    )

    print("     Cleaning data ... ")

    invalids = df.filter(
        (pl.col("descriptors").map_elements(lambda arr: any(x < 0 or x > 1 for x in arr), return_dtype=pl.Boolean))
    )

    assert invalids.height == 0, (
        "Some physico chemical descriptors are not scaled properly. Expected [0,1] range.\n"
        f"Details:\n{invalids['descriptors'].to_list()}"
    )

    final = df.filter(
        ~pl.col("sequence").str.contains_any(["*"])
    )

    if final.height != df.height:

        warnings.warn("Some sequences had stops inframe, they have been removed from the dataset !")

    return final


def get_T5_model(device):

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    if device==torch.device("cpu"):
        print("Model running on CPU. Casting full precision ...")
        model.to(torch.float64)
    
    model = model.to(device) 
    model = model.eval()
    
    return model, tokenizer

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
        print("Loading secondary structure model...")
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

    for category, fasta in fastas.items(): 

        print(f"Processing {category} ... ")

        records = list(SeqIO.parse(fasta, "fasta"))[:10000]

        np.random.shuffle(records)

        records.sort(key = lambda x: len(x.seq), reverse=False)

        seq_dict = { record.id : str(record.seq) for record in records }

        print("     Computing sequence descriptors ... : ")
        seq_descriptors = get_descriptors_pool(records = records, category = category, num_workers=40)

        print("     Computing embeddings ... ")
        seq_embeddings = get_embeddings(device = DEVICE, seqs = seq_dict, per_protein=True, per_residue=True, sec_struct = False, max_batch=300)
        seq_embeddings = embeddings_to_dataframe(seq_embeddings)

        final = return_final_df(seq_descriptors, seq_embeddings, seq_dict)

        print("     Writing to parquet ... ")

        final.write_parquet(f"../predict_data/{category}.parquet", compression="lz4")

        print(f"Written to ../predict_data/{category}.parquet")
        print(f"Done processing {category}, dataset is {final.shape[0]} entries long. ")

if __name__ == "__main__": 

    main()

