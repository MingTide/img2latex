import torch
import torch.nn.functional as F

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config, n_tok, id_end):
        super(Decoder, self).__init__()
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == "greedy" else config.beam_size

    def forward(self, img, formula):
        batch_size = img.shape[0]
        dim_embeddings = self._config.attn_cell_config.get("dim_embeddings")
        E = embedding_initializer((self._n_tok,dim_embeddings) )
        start_token_init = embedding_initializer((dim_embeddings,))
        start_token = nn.Parameter(start_token_init)
        embeddings = get_embeddings(formula, E, dim_embeddings,start_token,batch_size)



def embedding_initializer(shape):
    E = torch.empty( shape).uniform_(-1.0, 1.0)
    E = F.normalize(E, p=2, dim=-1)
    return E
def get_embeddings(formula, E, dim, start_token, batch_size):
    """
        PyTorch version of embedding lookup + start_token concat.
        Args:
            formula: LongTensor of shape [batch_size, seq_len], token ids
            E: Tensor of shape [vocab_size, dim], embedding matrix
            dim: int, embedding dimension
            start_token: Tensor of shape [dim], a learnable vector
            batch_size: int

        Returns:
            embeddings: Tensor of shape [batch_size, seq_len, dim]
    """
    # 1. 查嵌入：等价于 tf.nn.embedding_lookup
    formula_ = F.embedding(formula, E)  # shape: [batch_size, seq_len, dim]
    # 2. 构造 [batch_size, 1, dim] 的 start_token 向量
    start_token_ = start_token.view(1, 1, dim)  # shape: [1, 1, dim]
    start_tokens = start_token_.expand(batch_size, 1, dim)  # shape: [B, 1, dim]
    # 3. 拼接 start_token 和 formula 的前 n-1 个 token 的嵌入
    embeddings = torch.cat([start_tokens, formula_[:, :-1, :]], dim=1)  # [B, seq_len, dim]
    return embeddings


