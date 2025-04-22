import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections


class cross_attention(nn.Sequential):
    def __init__(self, hidden_dim):
        super(cross_attention, self).__init__()
        transformer_emb_size_drug = hidden_dim
        transformer_n_layer_drug = 4
        transformer_intermediate_size_drug = hidden_dim
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1
        
        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    
    def forward(self, emb, ex_e_mask, device1):
        global device
        device = device1
        encoded_layers, attention_scores = self.encoder(emb, ex_e_mask)
        return encoded_layers, attention_scores


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings."""
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossFusion(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossFusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections for query, key, and value for both RNA and molecule
        self.query_rna = nn.Linear(hidden_size, self.all_head_size)
        self.key_mole = nn.Linear(hidden_size, self.all_head_size)
        self.value_rna = nn.Linear(hidden_size, self.all_head_size)
        self.value_mole = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # RNA and molecule hidden states
        rna_hidden = hidden_states[0]  # Shape: (batch_size, seq_len_rna, hidden_size)
        mole_hidden = hidden_states[1]  # Shape: (batch_size, seq_len_mole, hidden_size)
        
        # Attention masks
        rna_mask = attention_mask[0]  # Shape: (batch_size, seq_len_rna)
        mole_mask = attention_mask[1]  # Shape: (batch_size, seq_len_mole)
        
        # Prepare masks for attention
        rna_mask = rna_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len_rna)
        rna_mask = ((1.0 - rna_mask) * -10000.0).to(device)
        
        mole_mask = mole_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len_mole)
        mole_mask = ((1.0 - mole_mask) * -10000.0).to(device)

        # Compute query, key, and value projections
        query_rna = self.query_rna(rna_hidden)  # Shape: (batch_size, seq_len_rna, all_head_size)
        key_mole = self.key_mole(mole_hidden)  # Shape: (batch_size, seq_len_mole, all_head_size)
        value_rna = self.value_rna(rna_hidden)  # Shape: (batch_size, seq_len_rna, all_head_size)
        value_mole = self.value_mole(mole_hidden)  # Shape: (batch_size, seq_len_mole, all_head_size)

        # Transpose for multi-head attention
        query_rna = self.transpose_for_scores(query_rna)  # Shape: (batch_size, num_heads, seq_len_rna, head_size)
        key_mole = self.transpose_for_scores(key_mole)  # Shape: (batch_size, num_heads, seq_len_mole, head_size)
        value_rna = self.transpose_for_scores(value_rna)  # Shape: (batch_size, num_heads, seq_len_rna, head_size)
        value_mole = self.transpose_for_scores(value_mole)  # Shape: (batch_size, num_heads, seq_len_mole, head_size)

        # Compute co-attention scores (RNA queries attend to molecule keys)
        attention_scores = torch.matmul(query_rna, key_mole.transpose(-1, -2))  # Shape: (batch_size, num_heads, seq_len_rna, seq_len_mole)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply molecule mask to prevent attending to padded tokens
        attention_scores = attention_scores + mole_mask  # Broadcasting mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute attended embeddings
        # RNA attended by molecule
        context_layer_rna = torch.matmul(attention_probs, value_mole)  # Shape: (batch_size, num_heads, seq_len_rna, head_size)
        context_layer_rna = context_layer_rna.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_len_rna, num_heads, head_size)
        new_context_layer_shape_rna = context_layer_rna.size()[:-2] + (self.all_head_size,)
        context_layer_rna = context_layer_rna.view(*new_context_layer_shape_rna)  # Shape: (batch_size, seq_len_rna, all_head_size)

        # Molecule attended by RNA (using transposed attention probabilities)
        context_layer_mole = torch.matmul(attention_probs.transpose(-1, -2), value_rna)  # Shape: (batch_size, num_heads, seq_len_mole, head_size)
        context_layer_mole = context_layer_mole.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_len_mole, num_heads, head_size)
        new_context_layer_shape_mole = context_layer_mole.size()[:-2] + (self.all_head_size,)
        context_layer_mole = context_layer_mole.view(*new_context_layer_shape_mole)  # Shape: (batch_size, seq_len_mole, all_head_size)

        # Output of co-attention
        context_layer = [context_layer_rna, context_layer_mole]
        # Attention probabilities (single matrix for co-attention)
        attention_probs = [attention_probs]

        return context_layer, attention_probs


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = self.dropout(hidden_states_rna)
        hidden_states_rna = self.LayerNorm(hidden_states_rna + input_tensor[0])
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = self.dropout(hidden_states_mole)
        hidden_states_mole = self.LayerNorm(hidden_states_mole + input_tensor[1])
        return [hidden_states_rna, hidden_states_mole]    


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = CrossFusion(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores    


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = F.gelu(hidden_states_rna)
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = F.gelu(hidden_states_mole)
        
        return [hidden_states_rna, hidden_states_mole]    


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = self.dropout(hidden_states_rna)
        hidden_states_rna = self.LayerNorm(hidden_states_rna + input_tensor[0])
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = self.dropout(hidden_states_mole)
        hidden_states_mole = self.LayerNorm(hidden_states_mole + input_tensor[1])
        return [hidden_states_rna, hidden_states_mole]    


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores    


class Encoder_1d(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_1d, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])
        self.mod = nn.Embedding(2, hidden_size)
    
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        seq_rna_emb1 = torch.tensor([0]).expand(hidden_states[0].size()[0], hidden_states[0].size()[1]).to(device)
        seq_rna_emb1 = self.mod(seq_rna_emb1)
        hidden_states[0] = hidden_states[0] + seq_rna_emb1
        
        seq_mole_emb1 = torch.tensor([0]).expand(hidden_states[2].size()[0], hidden_states[2].size()[1]).to(device)
        seq_mole_emb1 = self.mod(seq_mole_emb1)
        hidden_states[2] = hidden_states[2] + seq_mole_emb1
        
        stru_rna_emb1 = torch.tensor([1]).expand(hidden_states[1].size()[0], hidden_states[1].size()[1]).to(device)
        stru_rna_emb1 = self.mod(stru_rna_emb1)
        hidden_states[1] = hidden_states[1] + stru_rna_emb1
        
        stru_mole_emb1 = torch.tensor([1]).expand(hidden_states[3].size()[0], hidden_states[3].size()[1]).to(device)
        stru_mole_emb1 = self.mod(stru_mole_emb1)
        hidden_states[3] = hidden_states[3] + stru_mole_emb1
        
        rna_hidden = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
        mole_hidden = torch.cat((hidden_states[2], hidden_states[3]), dim=1)
        
        rna_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)
        mole_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)

        hidden_states = [rna_hidden, mole_hidden]        
        attention_mask = [rna_mask, mole_mask]
        
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_scores