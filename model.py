import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding to inject sequence position information.
    
    Since the Transformer has no recurrence or convolution, we need to add
    positional information to the embeddings. We use sine and cosine functions
    of different frequencies.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        pos = position in the sequence
        i = dimension index
        d_model = embedding dimension
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, we linearly project
    the queries, keys, and values h times with different learned projections.
    On each of these projected versions, we perform the attention function
    in parallel, yielding h output values which are concatenated and once
    again projected.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Keys tensor of shape (batch_size, num_heads, seq_len, d_k)
            V: Values tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor
            
        Returns:
            Attention output and attention weights
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for padding and future positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This consists of two linear transformations with a ReLU activation in between.
    It is applied to each position separately and identically.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    A single encoder layer consisting of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and normalization
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    A single decoder layer consisting of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (attending to encoder output)
    4. Add & Norm
    5. Position-wise feed-forward network
    6. Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
            
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    The Encoder stack consisting of N encoder layers.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of token indices, shape (batch_size, seq_len)
            mask: Optional mask tensor
            
        Returns:
            Encoder output tensor of shape (batch_size, seq_len, d_model)
        """
        # Embedding and scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Decoder(nn.Module):
    """
    The Decoder stack consisting of N decoder layers.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor of token indices, shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
            
        Returns:
            Decoder output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embedding and scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for machine translation.
    
    This model combines an encoder and decoder with a final linear projection
    to output vocabulary probabilities.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_encoder_layers,
                              num_heads, d_ff, max_len, dropout)
        
        self.decoder = Decoder(tgt_vocab_size, d_model, num_decoder_layers,
                              num_heads, d_ff, max_len, dropout)
        
        # Final linear layer to project to vocabulary size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters with Xavier uniform
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass of the Transformer.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
            
        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
        
    def encode(self, src, src_mask=None):
        """Encode source sequence only.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            src_mask: Optional source mask tensor
            
        Returns:
            Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence given encoder output.
        
        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output tensor
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
            
        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)


def create_padding_mask(seq, pad_idx):
    """Create mask for padding tokens.
    
    Args:
        seq: Sequence tensor of shape (batch_size, seq_len)
        pad_idx: Index of padding token
        
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """Create mask to prevent attending to future positions.
    
    Args:
        size: Size of the sequence
        
    Returns:
        Mask tensor of shape (1, size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask


def create_target_mask(tgt, pad_idx):
    """Create combined mask for target sequence (padding + look-ahead).
    
    Args:
        tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
        pad_idx: Index of padding token
        
    Returns:
        Combined mask tensor
    """
    tgt_len = tgt.size(1)
    
    # Create padding mask
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    
    # Create look-ahead mask
    look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)
    
    # Combine masks
    tgt_mask = tgt_padding_mask & look_ahead_mask.unsqueeze(0)
    
    return tgt_mask

if __name__ == "__main__":
    # Example usage and model testing
    print("Testing Transformer Model...")
    
    # Model hyperparameters
    src_vocab_size = 32000
    tgt_vocab_size = 32000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Test forward pass
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create masks
    pad_idx = 0
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_target_mask(tgt, pad_idx)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\nForward pass test:")
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")
    
    print("\nModel architecture test passed! ✓")