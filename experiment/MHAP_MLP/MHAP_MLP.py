from ..src import MHAPooling, MLP 

import torch.nn as nn 


class MHAP_MLP(nn.Module):
    """
    forward batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> mhap_mlp = MHAP_MLP(input_embed_dim = 1024, output_embed_dim = 128, num_classes = 5)
    >>> mlp_output = mhap_mlp.forward(batch, mask = None)
    >>> mlp_output.shape
    torch.Size([64, 5])
    """
    def __init__(self, input_embed_dim: int, num_classes: int, output_embed_dim:int=None):
        super(MHAP_MLP, self).__init__()
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim)
        self.mlp = MLP(input_shape=output_embed_dim, output_shape=num_classes)

    def forward(self, embeddings, mask):
        attn_output, _ = self.mhap(embeddings, mask)  
        mlp_output = self.mlp(attn_output)             
        return mlp_output
    


if __name__ == "__main__":

    import argparse
    import doctest
    import sys

    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)   

                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )

        sys.exit()