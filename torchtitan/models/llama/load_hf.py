from typing import Any, Dict, Optional

import torch
from safetensors import safe_open

import re


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
    model_config
):
    if model_config.enable_finetune is False:
        return model
    
    import gc
    from pathlib import Path

    _model_dir = Path(model_config.model_path)
    _model_files = model_config.model_files
    state_dict = {}
    
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