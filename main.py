
import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from copy import deepcopy
from tqdm.auto import tqdm, trange
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
# utils
import json
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from copy import deepcopy
from torch.nn import functional as F
from tabulate import tabulate
from tqdm import tqdm, trange
import functools
import math

# this resets up the site so you don't have to restart the runtime to use pysvelte
import site
site.main()
import pysvelte


sns.set_palette('colorblind')
cmap = sns.color_palette('colorblind')

def keep_k(x, k=100, absolute=True, dim=-1):
    shape = x.shape
    x_ = x
    if absolute:
        x_ = abs(x)
    values, indices = torch.topk(x_, k=k, dim=dim)
    res = torch.zeros_like(x)
    res.scatter_(dim, indices, x.gather(dim, indices))
    return res

def get_max_token_length(tokens):
  maxlen = 0
  for t in tokens:
    l = len(t)
    if l > maxlen:
      maxlen = l
  return maxlen

def pad_with_space(t, maxlen):
  spaces_to_add = maxlen - len(t)
  for i in range(spaces_to_add):
    t += " "
  return t

def convert_to_tokens(indices, tokenizer, extended, extra_values_pos, strip=True, pad_to_maxlen=False):
    if extended:
        res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else 
               (f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx-extra_values_pos}]") 
               for idx in indices]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    if pad_to_maxlen:
      maxlen = get_max_token_length(res)
      res = list(map(lambda t: pad_with_space(t, maxlen), res))
    return res


def top_tokens(v_tok, k=100, tokenizer=None, only_english=False, only_ascii=True, with_values=False, 
               exclude_brackets=False, extended=True, extra_values=None, pad_to_maxlen=False):
    if tokenizer is None:
        tokenizer = my_tokenizer
    v_tok = deepcopy(v_tok)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
    if only_english: 
        ignored_indices =[key for val, key in tokenizer.vocab.items() if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    v_tok[ignored_indices] = -np.inf
    extra_values_pos = len(v_tok)
    if extra_values is not None:
        v_tok = torch.cat([v_tok, extra_values])
    values, indices = torch.topk(v_tok, k=k)
    res = convert_to_tokens(indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos,pad_to_maxlen = pad_to_maxlen)
    if with_values:
        res = list(zip(res, values.cpu().numpy()))
    return res


def top_matrix_tokens(mat, k=100, tokenizer=None, rel_thresh=None, thresh=None, 
                      sample_entries=10000, alphabetical=True, only_english=False,
                      exclude_brackets=False, with_values=True, extended=True):
    if tokenizer is None:
        tokenizer = my_tokenizer
    mat = deepcopy(mat)
    ignored_indices = []
    if only_english:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.strip('[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    mat[ignored_indices, :] = -np.inf
    mat[:, ignored_indices] = -np.inf
    cond = torch.ones_like(mat).bool()
    if rel_thresh:
        cond &= (mat > torch.max(mat) * rel_thresh)
    if thresh:
        cond &= (mat > thresh)
    entries = torch.nonzero(cond)
    if sample_entries:
        entries = entries[np.random.randint(len(torch.nonzero(cond)), size=sample_entries)]
    res_indices = sorted(entries, 
                         key=lambda x: x[0] if alphabetical else -mat[x[0], x[1]])
    res = [*map(partial(convert_to_tokens, extended=extended, tokenizer=tokenizer), res_indices)]
            
    if with_values:
        res_ = []
        for (x1, x2), (i1, i2) in zip(res, res_indices):
            res_.append((x1, x2, mat[i1][i2].item()))
        res = res_    
    return res


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.splita('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def get_model_tokenizer_embedding(model_name="gpt2-medium"):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cpu':
    print("WARNING: you should probably restart on a GPU runtime")

  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  emb = model.get_output_embeddings().weight.data.T.detach()
  return model, tokenizer, emb, device


def get_model_info(model):
  num_layers = model.config.n_layer
  num_heads = model.config.n_head
  hidden_dim = model.config.n_embd
  head_size = hidden_dim // num_heads
  return num_layers, num_heads, hidden_dim, head_size

def get_mlp_weights(model,num_layers, hidden_dim):
  Ks = []
  Vs = []
  for j in range(num_layers):
    K = model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T.detach()
    # fuse the layernorm
    ln_2_weight = model.get_parameter(f"transformer.h.{j}.ln_2.weight").detach()
    K = torch.einsum("oi,i -> oi", K, ln_2_weight)
    
    V = model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
    Ks.append(K)
    Vs.append(V)
  
  Ks =  torch.cat(Ks)
  Vs = torch.cat(Vs)
  K_heads = Ks.reshape(num_layers, -1, hidden_dim)
  V_heads = Vs.reshape(num_layers, -1, hidden_dim)
  return K_heads, V_heads

def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
  qkvs = []
  for j in range(num_layers):
    qkv = model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight").detach().T
    ln_weight_1 = model.get_parameter(f"transformer.h.{j}.ln_1.weight").detach()
    
    qkv = qkv - torch.mean(qkv, dim=0) 
    qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
    qkvs.append(qkv.T)

  W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
  W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") for j in range(num_layers)]).detach()
  W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
  W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  return W_Q_heads, W_K_heads, W_V_heads, W_O_heads

def top_singular_vectors(mat, emb, all_tokens, k = 20, N_singular_vectors = 10, with_negative = False,use_visualization=True, filter="topk"):
  U,S,V = torch.linalg.svd(mat)
  Vs = []
  for i in range(N_singular_vectors):
      acts = V[i,:].float() @ emb
      Vs.append(acts)
  if use_visualization:
    Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
    pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
  else:
    Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
    print(tabulate([*zip(*Vs)]))
  if with_negative:
    Vs = []
    for i in range(N_singular_vectors):
      acts = -V[i,:].float() @ emb
      Vs.append(acts)
    if use_visualization:
      Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
      pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
    else:
      Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
      print(tabulate([*zip(*Vs)]))

def plot_MLP_singular_vectors(K,layer_idx, max_rank=None):
  W_matrix = K[layer_idx, :,:]
  U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
  if not max_rank:
    max_rank = len(S)
  if max_rank > len(S):
    max_rank = len(S) -1
  plt.plot(S[0:max_rank].detach().cpu().numpy())
  plt.yscale('log')
  plt.ylabel("Singular value")
  plt.xlabel("Rank")
  plt.title("Distribution of the singular vectors")
  plt.show()

def cosine_sim(x,y):
    return torch.dot(x,y) / (torch.norm(x) * torch.norm(y))


def normalize_and_entropy(V, eps=1e-6):
    absV = torch.abs(V)
    normV = absV / torch.sum(absV)
    entropy = torch.sum(normV * torch.log(normV + eps)).item()
    return -entropy

def OV_top_singular_vectors(W_V_heads, W_O_heads, emb, layer_idx, head_idx, all_tokens, k=20, N_singular_vectors=10, use_visualization=True, with_negative=False, filter="topk", return_OV=False):
  W_V_tmp, W_O_tmp = W_V_heads[layer_idx, head_idx, :], W_O_heads[layer_idx, head_idx]
  OV = W_V_tmp @ W_O_tmp
  U,S,V = torch.linalg.svd(OV)
  Vs = []
  for i in range(N_singular_vectors):
      acts = V[i,:].float() @ emb
      Vs.append(acts)
  if use_visualization:
    Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
    pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
  else:
    Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
    print(tabulate([*zip(*Vs)]))
  if with_negative:
    Vs = []
    for i in range(N_singular_vectors):
      acts = -V[i,:].float() @ emb
      Vs.append(acts)
    if use_visualization:
      Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
      pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
    else:
      Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
      print(tabulate([*zip(*Vs)]))
  if return_OV:
    return OV

if __name__ == "__main__":
  # Load up the model and get all the key weight matrices.
  model, tokenizer, emb, device = get_model_tokenizer_embedding()
  my_tokenizer = tokenizer
  num_layers, num_heads, hidden_dim, head_size = get_model_info(model)
  all_tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]

  K,V = get_mlp_weights(model, num_layers = num_layers, hidden_dim = hidden_dim)
  W_Q_heads, W_K_heads, W_V_heads, W_O_heads = get_attention_heads(model, num_layers=num_layers, hidden_dim=hidden_dim, num_heads=num_heads, head_size = head_size)


  OV_top_singular_vectors(W_V_heads, W_O_heads, emb, layer_idx=22, head_idx=10,k=20, N_singular_vectors=15, all_tokens = all_tokens)
