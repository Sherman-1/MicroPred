from Bio import SeqIO 
import polars as pl 
from sklearn.model_selection import train_test_split
from bin.descriptors.sequence_descriptors import process_data_pool as get_descriptors
from transformers import T5EncoderModel, T5Tokenizer
import torch
import warnings
from tqdm import tqdm 
import time 
import numpy as np

CLASSES = [
    
    "molten",
    "globular",
    "bitopic",
    "polytopic",
    "disprot"
]

CLASS_TO_INT = dict(zip(CLASSES, range(len(CLASSES)))) 

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
    converts dict returned by get_embeddings() to a Polars DF
    suited for parquet storage. 
    Cols : id, protein_embs, residue_embs
    """

    protein_ids = set(results.get("protein_embs", {}).keys())
    residue_ids = set(results.get("residue_embs", {}).keys())

    # Shouldn't 
    assert len(protein_ids) == len(residue_ids), "IDs don't match between per-residue and per-protein embeddings"

    all_ids = protein_ids.union(residue_ids)

    rows = []
    for identifier in all_ids:

        # { id : { prot_embs : torch.Tensor(1,1024), residue_embs : torch.Tensor(Lx1024) } }
        protein_array = results.get("protein_embs", {}).get(identifier)
        protein_list = protein_array.tolist() if protein_array is not None else None

        # Get the residue embeddings (2D) if they exist, converting to a list of lists
        residue_array = results.get("residue_embs", {}).get(identifier)
        residue_list = residue_array.tolist() if residue_array is not None else None

        rows.append({
            "id": identifier,
            "protein_emb": protein_list,
            "residue_emb": residue_list
        })

    return pl.DataFrame(rows)

def return_final_df(seq_descriptors, seq_embeddings, seq_dict, unique_classes):


    num_classes = len(unique_classes)

    df = (
        seq_descriptors
        .with_columns(
            HCA_score = ( (pl.col("HCA_score") + 10) / 20 ) # x - min / max - min, HCA [-10,10]. Scale to [0,1], every other phychem is already [0,1] 
        )
        .with_columns(

            # Turn seq descriptors into an array like structure for later ( easier dataloading )
            descriptors = pl.concat_list(pl.exclude(["id","category","seq_len"])),
            # Dirty way to keep the names of each descriptors in order ... 
            descriptor_names = seq_descriptors.select(pl.exclude(["id","category","seq_len"])).columns,
            one_hot = pl.col("category").map_elements(lambda x: np.eye(num_classes, dtype=float)[x].tolist(), return_dtype = pl.List(pl.Float32))
            
        )
        .select(["id","seq_len","descriptors","category","one_hot","descriptor_names"])
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

    # Better in a yaml file, do it later
    from pathlib import Path

    fastas = {
        
        "globular" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas/globular_homologs_representatives.fasta",
        "molten" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas/molten_homologs_representatives.fasta",
        "transmembrane" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas/transmembrane_elongated_representatives.fasta",
        "disordered" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas/representative_disordered_sequences.fasta"
    }

    dfs = []

    unique_classes = [ CLASS_TO_INT[CLASS] for CLASS in CLASSES ]

    for category, fasta in fastas.items(): 

        records = list(SeqIO.parse(fasta, "fasta"))

        seq_dict = { record.id : str(record.seq) for record in records }

        print("Computing sequence descriptors ... : ")
        seq_descriptors = get_descriptors(records = records, category = CLASS_TO_INT[category], num_workers = 32)

        print("Computing embeddings ... ")
        seq_embeddings = get_embeddings(device = DEVICE, seqs = seq_dict, per_protein=True, per_residue=True, sec_struct = False, max_batch=300)
        seq_embeddings = embeddings_to_dataframe(seq_embeddings)

        dfs.append(return_final_df(seq_descriptors, seq_embeddings, seq_dict, unique_classes))

    data = pl.concat(dfs)

    train, val = train_test_split(data, test_size = 0.4, stratify=data[["category"]], shuffle=True)

    print(train.select(["category","one_hot"]))

    print(f"Writting {train.height} sequences to train dataset")
    train.write_parquet("../training_dataset/new_train.parquet")

    print(f"Writting {val.height} sequences to validation dataset")
    val.write_parquet("../training_dataset/new_eval.parquet")

if __name__ == "__main__": 

    main()

