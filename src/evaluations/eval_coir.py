import coir
from coir.evaluation import COIR
from sentence_transformers import SentenceTransformer
from beir.retrieval import models
import fire
import numpy as np
import torch
from utils import Retriever

"""
-    # model_name = "nomic-ai/CodeRankEmbed"
-    model_name = "/workspace/Embeddings/contrastors/src/contrastors/ckpts/nomic-embed-text-v1-og-all_w0/step_4000/model_hf"
-    # model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
-    # model_name = "/workspace/Embeddings/contrastors/src/contrastors/ckpts/nomic-embed-text-v1-og-all/final_model_hf"
"""
#ran with 512 seq length for fair comparison against baselines following CoIR paper although higher seq length may yield better pfm
def main(model_name: str, tasks: str = 'all', output_dir: str = 'results', batch_size: int = 256, max_seq_length: int = 512, multiprocess: bool = True):
    print(f"Using model: {model_name}")
    st = SentenceTransformer(model_name, trust_remote_code= True).to(torch.bfloat16)
    st.max_seq_length = max_seq_length
    contrast_encoder = Retriever(st, add_prefix= True, multiprocess=multiprocess)

    if tasks == 'all':
        tasks = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt",
                                        "codefeedback-st","codetrans-contest","synthetic-text2sql",
                                        "cosqa","codesearchnet","codesearchnet-ccr"]
    else:
        tasks = tasks.split()
    
    for task in tasks:
        if task in ['apps', 'cosqa', "synthetic-text2sql"]:
            contrast_encoder.add_prefix = True 
        else:
            contrast_encoder.add_prefix = False 
        
        # Initialize evaluation
        evaluation = COIR(tasks= coir.get_tasks(tasks= [task]),batch_size=batch_size)

        model_folder = "/".join(model_name.split("/")[-2:])
        # Run evaluation
        results = evaluation.run(contrast_encoder, output_folder=f"{output_dir}/coir/{model_folder}")
        print(results)
    
if __name__ == "__main__":
    fire.Fire(main)