import torch
from torch import nn 
import torch.nn.functional as F
import math
from typing import Callable
import numpy as np

class LayerNormalization(nn.Module):
    def __init__(self, 
                eps: float = 1e-6
                ) -> None:
        """
        Initialize the LayerNormalization module.

        Args:
            eps (float): A small value to prevent division by zero for numerical stability. Default is 1e-6.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.eps = eps  # Save the epsilon value for numerical stability
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scale parameter, initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable shift parameter, initialized to 0

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the LayerNormalization module.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)  # Compute the mean of the input tensor along the last dimension
        std = x.std(-1, keepdim=True)  # Compute the standard deviation of the input tensor along the last dimension
        # Normalize the input tensor, scale by alpha, and shift by bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 d_model: int,       # Dimension of the embedding vector
                 d_ff: int,          # Dimension of the feed-forward layer
                 dropout_proba: float # Dropout probability
                 ) -> None:
        
        super().__init__()         # Initialize the parent class (nn.Module)
        self.linear_1 = nn.Linear(d_model, d_ff) # First linear transformation (batch_size, seq_len,d_model) -> (batch_size, seq_len, d_ff)
        self.dropout = nn.Dropout(dropout_proba) # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model) # Second linear transformation (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

    def forward(self, x):
        x = self.linear_1(x)       # Apply the first linear transformation
        x = F.relu(x)              # Apply ReLU activation function
        x = self.dropout(x)        # Apply dropout for regularization
        x = self.linear_2(x)       # Apply the second linear transformation
        return x   

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, 
                d_model: int, 
                h: int, 
                dropout_proba: float
                ) -> None:
        """
        Initialize the MultiHeadAttentionBlock module.

        Args:
            d_model (int): Dimension of the model.
            h (int): Number of attention heads.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Ensure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of each attention head

        # Linear layers to project inputs to query, key, and value
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Linear layer to project concatenated outputs back to d_model
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_proba)

    @staticmethod
    def self_attention(q: torch.Tensor, 
                    k: torch.Tensor, 
                    v: torch.Tensor, 
                    dropout: nn.Dropout, 
                    attn_mask: torch.Tensor = None
                    ) -> torch.Tensor:
        """
        Compute self-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_k).
            dropout (nn.Dropout): Dropout layer for regularization.
            attn_mask (torch.Tensor, optional): Attention mask tensor. Should be broadcastable to the shape of attention_scores.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and attention weights.
        """
        d_k = q.shape[-1]
        # Compute attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # Ensure that the mask is broadcastable to the shape of attention_scores
            if attn_mask.dim() == 2:
                # attn_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len) for broadcasting
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 3:
                # attn_mask: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len) for broadcasting
                attn_mask = attn_mask.unsqueeze(1)
            
            elif attn_mask.dim() == 4:
            # Ensure mask is broadcastable to attention_scores
                assert attn_mask.shape[1] == 1, "attn_mask must have shape (batch_size, 1, seq_len_decoder, seq_len_encoder)"
            
            if attn_mask.shape[-2] != attention_scores.shape[-2]:
                # You may need to slice or modify the mask to fit the dimensions
                # For example, if the `attn_mask` should apply to the first `seq_len_q` tokens
                attn_mask = attn_mask[:, :, :attention_scores.shape[-2], :]

            if attn_mask.shape != attention_scores.shape:
                # Ensure the mask can be broadcast to the shape of attention_scores
                attn_mask = attn_mask.expand_as(attention_scores)
                print(f"attn_mask shape after adjustment: {attn_mask.shape}")

            
            print(f"attention_scores dim: {attention_scores.shape}")
            print(f"attention mask dim: {attn_mask.shape}")
            # Broadcast mask to match the shape of attention_scores
            attention_scores = attention_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax to obtain attention weights
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute weighted sum of values
        return (attention_scores @ v), attention_scores

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                attn_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Forward pass through the MultiHeadAttentionBlock module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head self-attention and final linear transformation.
        """
        # Apply linear transformations to input Q, K, V
        query = self.W_Q(Q)
        key = self.W_K(K)
        value = self.W_V(V)

        # Split Q, K, V into multiple heads and rearrange dimensions
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Perform self-attention
        x, self.attention_scores = MultiHeadAttentionBlock.self_attention(query, key, value, self.dropout, attn_mask)

        # Concatenate heads and reshape back to original dimensions
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.W_O(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
                decoder_blocks: nn.ModuleList
                ) -> None:
        """
        Initialize the Decoder module.

        Args:
            decoder_blocks (nn.ModuleList): List of decoder blocks to be applied sequentially.
        """
        super().__init__()
        self.decoder_blocks = decoder_blocks  # Store the list of decoder blocks
        self.norm = LayerNormalization()  # Initialize layer normalization

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_output_mask: torch.Tensor,
                decoder_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_length, d_model).
            encoder_output_mask (torch.Tensor): Mask tensor applied to the encoder output, shape (batch_size, 1, seq_length, seq_length).
            decoder_mask (torch.Tensor): Mask tensor applied to the decoder input, shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor after applying all decoder blocks and final normalization.
        """
        # Pass the input tensor through each decoder block sequentially
        for block in self.decoder_blocks:
            x = block(x, encoder_output, encoder_output_mask, decoder_mask)
        
        # Apply layer normalization to the final output
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout_proba: float
                 ) -> None:
        """
        Initialize the DecoderBlock module.

        Args:
            self_attention_block (MultiHeadAttentionBlock): Instance of the multi-head self-attention block.
            cross_attention_block (MultiHeadAttentionBlock): Instance of the multi-head cross-attention block.
            feed_forward_block (FeedForwardBlock): Instance of the feed-forward block.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        # Initialize residual connections for self-attention, cross-attention, and feed-forward blocks
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout_proba) for _ in range(3)
        ])

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_output_mask: torch.Tensor,
                decoder_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the DecoderBlock module.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer (or input embedding), shape (batch_size, seq_length, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_length, d_model).
            encoder_output_mask (torch.Tensor): Mask tensor applied to the encoder output, shape (batch_size, 1, seq_length, seq_length).
            decoder_mask (torch.Tensor): Mask tensor applied to the decoder input, shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, cross-attention, and feed-forward blocks with residual connections, shape (batch_size, seq_length, d_model).
        """
        # Apply self-attention block with residual connection
        # Lambda function is used to correctly pass the multiple arguments (query, key, value, decoder_mask) to the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        
        # Apply cross-attention block with residual connection
        # Lambda function is used to correctly pass the multiple arguments (query, key, value, encoder_output_mask) to the cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, encoder_output_mask))
        
        # Apply feed-forward block with residual connection
        # Direct function call since feed-forward block only needs the input tensor
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x

class Encoder(nn.Module):
    def __init__(self, 
                encoder_blocks: nn.ModuleList
                ) -> None:
        """
        Initialize the Encoder module.

        Args:
            encoder_blocks (nn.ModuleList): List of encoder blocks to be applied sequentially.
        """
        super().__init__()
        self.encoder_blocks = encoder_blocks  # Store the list of encoder blocks
        self.norm = LayerNormalization()  # Initialize layer normalization

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            mask (torch.Tensor): Mask tensor for attention mechanism, used to mask certain positions.

        Returns:
            torch.Tensor: Output tensor after applying all encoder blocks and final normalization.
        """
        # Pass the input tensor through each encoder block sequentially
        for block in self.encoder_blocks:
            x = block(x, mask)
        
        # Apply layer normalization to the final output
        return self.norm(x)

class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout_proba: float
                 ) -> None:
        """
        Initialize the EncoderBlock module.

        Args:
            self_attention_block (MultiHeadAttentionBlock): Instance of the multi-head self-attention block.
            feed_forward_block (FeedForwardBlock): Instance of the feed-forward block.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Initialize residual connections for both the self-attention and feed-forward blocks
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout_proba) for _ in range(2)
        ])

    def forward(self, 
                x: torch.Tensor, 
                src_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the EncoderBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor): Source mask tensor for attention mechanism, used to mask certain positions.

        Returns:
            torch.Tensor: Output tensor after applying the self-attention and feed-forward blocks with residual connections.
        """
        # Apply self-attention block with residual connection
        
        # The lambda function is used for the self-attention block to handle the multiple arguments required 
        # (query, key, value, and source mask). The feed-forward block, needing only the input tensor, can be passed 
        # directly without a lambda function. 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Apply feed-forward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x

                # Return the output

class InputEmbeddings(nn.Module):
    """
    The InputEmbeddings class converts token indices into dense embedding vectors.
    """
    def __init__(self, 
                d_model: int, 
                vocab_size: int
                ) -> None:
        """
        Initialize the InputEmbeddings module.

        Args:
            d_model (int): Dimension of the embedding vector.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.d_model = d_model  # Save the dimension of the embedding
        self.vocab_size = vocab_size  # Save the vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Create an embedding layer with vocab_size entries, each of dimension d_model

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the InputEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor of dense embedding vectors with shape (batch_size, seq_length, d_model).
        """
        # Forward pass through the embedding layer
        embeddings = self.embedding(x)
        
        # Multiply the output of the embedding layer by the square root of the embedding dimension (d_model)
        # This scaling is often used in transformer models to stabilize training
        scaled_embeddings = embeddings * math.sqrt(self.d_model)
        
        return scaled_embeddings


"""
Positional encodings are vectors added to the input embeddings that provide information about the position of each 
token in the sequence. These vectors have the same dimension as the embeddings, allowing them to be summed directly.

The PositionalEncoding class is designed to generate and apply positional encodings to the input embeddings, allowing 
the model to capture the order of the tokens in a sequence.
"""

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,       # Dimension of the embedding vectors
                 seq_len: int,       # Maximum sequence length
                 dropout_proba: float # Dropout probability
                 ) -> None:
        super().__init__()         # Initialize the parent class (nn.Module)
        self.d_model = d_model     # Save the dimension of the embedding
        self.seq_len = seq_len     # Save the sequence length
        self.dropout = nn.Dropout(dropout_proba) # Initialize the dropout layer

        # Create a matrix of shape (seq_len, d_model) to hold positional encodings
        pos_encoding = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1) containing positions 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the div_term which is used to scale the positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sine function to the even positions (0, 2, 4, ...)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine function to the odd positions (1, 3, 5, ...)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to match the input shape (1, seq_len, d_model)
        pos_encoding = pos_encoding.unsqueeze(0)

        # Register the positional encoding matrix as a buffer, so it's not considered a parameter - not updated during backprop
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        x = x + self.pos_encoding[:, :x.shape[1], :].detach()

        # Apply dropout to the result
        return self.dropout(x)

class ProjectionLayer(nn.Module):
    def __init__(self, 
                d_model: int, 
                vocab_size: int
                ) -> None:
        """
        Initialize the ProjectionLayer module.

        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)  # Linear layer to project from d_model to vocab_size

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the ProjectionLayer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, vocab_size) with log probabilities.
        """
        # Apply the linear projection
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, vocab_size)
        projected = self.projection(x)
        
        # Apply log softmax to obtain log probabilities
        # Log softmax is applied along the last dimension (vocab_size)
        log_probs = F.log_softmax(projected, dim=-1)
        
        return log_probs

class ResidualConnection(nn.Module):
    def __init__(self, 
                dropout_proba: float
                ) -> None:
        """
        Initialize the ResidualConnection module.

        Args:
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_proba)  # Dropout layer for regularization
        self.norm = LayerNormalization()  # Layer normalization to stabilize training

    def forward(self, 
                x: torch.Tensor, 
                sublayer: Callable[[torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
        """
        Forward pass through the ResidualConnection module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            sublayer (Callable[[torch.Tensor], torch.Tensor]): A sublayer function or module
                (e.g., self-attention or feed-forward network) that will be applied to the input
                after normalization.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model) after applying
                the sublayer with residual connection, layer normalization, and dropout.
        """
        normalized_x = self.norm(x)  # Apply layer normalization to the input
        sublayer_output = sublayer(normalized_x)  # Apply the sublayer (e.g., self-attention or feed-forward)
        dropped_out_output = self.dropout(sublayer_output)  # Apply dropout to the sublayer output
        return x + dropped_out_output  # Add the original input (residual connection) to the transformed input

class Transformer(nn.Module):

    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        """
        Initialize the Transformer module.

        Args:
            encoder (Encoder): Encoder module of the transformer.
            decoder (Decoder): Decoder module of the transformer.
            src_embed (InputEmbeddings): Source input embeddings.
            tgt_embed (InputEmbeddings): Target input embeddings.
            src_pos (PositionalEncoding): Positional encoding for source inputs.
            tgt_pos (PositionalEncoding): Positional encoding for target inputs.
            projection_layer (ProjectionLayer): Linear projection layer to generate final output probabilities.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, 
               src: torch.Tensor, 
               src_mask: torch.Tensor
               ) -> torch.Tensor:
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, seq_length).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, seq_length, d_model).
        """
        # Apply source embeddings and positional encoding
        x = self.src_embed(src)
        x = self.src_pos(x)
        
        # Pass through the encoder
        x = self.encoder(x, src_mask)
        return x

    def decode(self, 
               encoder_output: torch.Tensor, 
               src_mask: torch.Tensor, 
               tgt: torch.Tensor, 
               tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode the target sequence using the encoder output.

        Args:
            encoder_output (torch.Tensor): Encoded source representation of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, seq_length, seq_length).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, seq_length).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, seq_length, d_model).
        """
        # Apply target embeddings and positional encoding
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        
        # Pass through the decoder
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

    def project(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Project the decoder output to the vocabulary size.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, seq_length, vocab_size).
        """
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int,
                        tgt_vocab_size: int,
                        src_seq_len: int,
                        tgt_seq_len: int,
                        d_model: int = 512,
                        N: int = 6,  # Number of encoder and decoder blocks
                        h: int = 8,  # Number of heads in MHA block
                        dropout_proba: float = 0.1,
                        d_ff: int = 2048  # Number of parameters in the Feed Forward Layer
                        ) -> Transformer:
    """
    Build a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequence.
        tgt_seq_len (int): Length of the target sequence.
        d_model (int): Dimension of the model. Default is 512.
        N (int): Number of encoder and decoder blocks. Default is 6.
        h (int): Number of heads in the multi-head attention block. Default is 8.
        dropout_proba (float): Dropout probability for regularization. Default is 0.1.
        d_ff (int): Number of parameters in the feed-forward layer. Default is 2048.

    Returns:
        Transformer: The constructed Transformer model.
    """
    # Create the embedding layers
    src_embed =  InputEmbeddings(d_model, src_vocab_size)
    tgt_embed =  InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos =  PositionalEncoding(d_model, src_seq_len, dropout_proba)
    tgt_pos =  PositionalEncoding(d_model, tgt_seq_len, dropout_proba)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_proba)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout_proba)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_proba)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout_proba)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder =  Encoder(nn.ModuleList(encoder_blocks))
    decoder =  Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer =  ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer =  Transformer(encoder=encoder,
                                decoder=decoder,
                                src_embed=src_embed,
                                tgt_embed=tgt_embed,
                                src_pos=src_pos,
                                tgt_pos=tgt_pos,
                                projection_layer=projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def inference(model: Transformer,
              sentence_in_src_lang: str,
              src_tokenizer: Callable[[str], torch.Tensor],
              tgt_tokenizer: Callable[[torch.Tensor], str],
              max_tgt_seq_len: int = 50,
              start_token_id: int = 1,
              end_token_id: int = 2
              ) -> str:
    """
    Perform inference with the Transformer model.

    Args:
        model (Transformer): The trained Transformer model.
        sentence_in_src_lang (str): The input sentence in the source language.
        src_tokenizer (Callable[[str], torch.Tensor]): Function to tokenize the source sentence into a tensor of token indices.
        tgt_tokenizer (Callable[[torch.Tensor], str]): Function to decode the output tensor of token indices into a sentence in the target language.
        max_tgt_seq_len (int): Maximum length of the target sequence to generate. Default is 50.
        start_token_id (int): Token ID that represents the start of the sequence. Default is 1.
        end_token_id (int): Token ID that represents the end of the sequence. Default is 2.

    Returns:
        str: The translated sentence in the target language.
    """
    model.eval()  # Set the model to evaluation mode

    # Tokenize the source sentence
    src_tensor = src_tokenizer(sentence_in_src_lang).unsqueeze(0)  # Shape: (1, src_seq_len)

    # Create the source mask (no masking required here, but could be modified if needed)
    src_mask = torch.ones((1, 1, src_tensor.size(1), src_tensor.size(1)))

    # Encode the source sentence
    encoder_output = model.encode(src_tensor, src_mask)

    # Initialize the target sequence with the start token
    tgt_tensor = torch.tensor([[start_token_id]], dtype=torch.long)

    for _ in range(max_tgt_seq_len):
        # Create the target mask
        tgt_mask = torch.ones((1, 1, tgt_tensor.size(1), tgt_tensor.size(1)))

        # Decode the current target sequence
        decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, tgt_mask)

        # Get the next token's probabilities from the projection layer
        next_token_log_probs = model.project(decoder_output[:, -1, :])  # Shape: (1, vocab_size)

        # Select the token with the highest probability
        next_token_id = next_token_log_probs.argmax(dim=-1).item()

        # Append the predicted token to the target sequence
        tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token_id]], dtype=torch.long)], dim=1)

        # Stop if the end token is predicted
        if next_token_id == end_token_id:
            break

    # Decode the full target sequence to a sentence
    translated_sentence = tgt_tokenizer(tgt_tensor.squeeze(0))

    return translated_sentence
