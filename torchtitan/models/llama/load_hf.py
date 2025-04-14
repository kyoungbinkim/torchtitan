import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from safetensors import safe_open

import torch

from huggingface_hub import hf_hub_download, HfFileSystem

import re

global titan_state 
titan_state = None

HF_FORMAT = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


'''
Copy from  https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py#L47-L64
'''
def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key

def hf_to_titan(
    state_dict: Dict[str, torch.Tensor],
    n_heads: int = 32,
    n_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of HF-format state dict can be found in the ``meta-llama/Llama-2-7b-hf``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        n_heads (int): Number of heads in the model.
        n_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // n_heads.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // n_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, HF_FORMAT)
            if "q_proj" in key:
                value = _permute(value, n_heads)
            elif "k_proj" in key:
                value = _permute(value, n_kv_heads)

            converted_state_dict[new_key] = value
    return converted_state_dict

def load_pretrained_llama3(
    model,
    model_config,
):
    if model_config.enable_finetune is False:
        return model
    
    import gc
    global titan_state 
    
    
    if titan_state == None:
        from pathlib import Path
        
        state_dict = {}
        _model_dir = Path(model_config.model_path)
        _model_files = model_config.model_files
        
        for model_path in _model_files:
            with safe_open(Path.joinpath(_model_dir ,model_path), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
    
    
        n_heads = model.model_args.n_heads
        n_kv_heads = model.model_args.n_kv_heads
        dim = model.model_args.dim
        
        titan_state = hf_to_titan(
            state_dict,
            n_heads,
            n_kv_heads,
            dim
        )
        
        del state_dict
        gc.collect()
    
    model.load_state_dict(titan_state, strict=False)
    
    return model

@dataclass
class HuggingfaceArgs:
    repo_id:str = 'meta-llama/Llama-3.1-8B',
    local_dir: str = 'assets/',
    tokenizer_path: str = 'original',
    index_path: str = 'model.safetensors.index.json'

def hf_download(
    model, 
    hf_args:HuggingfaceArgs
) -> None:
    repo_id = hf_args.repo_id
    local_dir = hf_args.local_dir
    tokenizer_path = hf_args.tokenizer_path
    index_path = hf_args.index_path
    
    # 토크나이저 다운로드    
    tokenizer_path = (
        f"{tokenizer_path}/tokenizer.model" if tokenizer_path else "tokenizer.model"
    )
    hf_hub_download(
        repo_id=repo_id,
        filename=tokenizer_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    # 인덱스 파일 다운로드
    index_path =  index_path or 'model.safetensors.index.json'
    hf_hub_download(
        repo_id=repo_id,
        filename=index_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    with open(local_dir+'/'+index_path, 'r') as model_index_path:
        model_index = json.load(model_index_path)
    
    # 필요한 레이어 셋 생성
    model_file_path_set = set()
    for layer_name in model.state_dict().keys():
        model_file_path_set.add(
            model_index['weight'][get_mapped_key(layer_name, HF_FORMAT)]
        )

    # 필요한 레이어 다운로드
    for model_file_path in model_file_path_set:
        hf_hub_download(
            repo_id=repo_id,
            filename=model_file_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )