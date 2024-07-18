import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
  
  
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #pe la viet tat cua positional embedding
        pe = torch.zeros(seq_len, d_model)
        #position de danh dau vi tri tu 0 - seq_len - 1, unsqueeze giup bien doi thanh ma tran (seq, 1), boi vi ban dau khoi tao arange no chi dơn gian la vector. unsqueeze giup chug ta them mot cot doc vao
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #div term la mau so cua cthuc tinh position, o day dung log j j do cho don gian 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #1 chieu cho batch, o day chung ta co 1 batch sentence
        pe = pe.unsqueeze(0) #(1, seq, d_model)
        self.register_buffer("pe", pe) #giup luu lai vao buffer, vi positional chi can tinh 1 lan
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
    
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        #parameter giup khai bao day la tham so trong mo hinh, co the duoc cap nhat qua qua trinh train
        #cac tham so trong nay chu yeu lien quan den cthuc layer norm nen chiu
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) #tinh toan gia tri trung binh theo mot chieu khong gian ma ban chon, neu k chi dinh cu the no tinh tat ca, o day la chieu cuoi cung
        std = x.std(dim = -1, keepdim=True) #tinh do lech chuan
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #bias mac dinh la true nen pytorch da dinh nghia sang bias cho ta r
        #o day linear la phep bien doi tuyen tinh linear(chieu vao, chieu ra)
        #co the thay cong thuc o ben duoi dau tien no dung linear 1 bien doi 512 sang 2048 chieu sau do tinh relu va dropout roi dung linear 2 bien doi ve nhu cu
    
    def forward(self, x):
        #(batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
    
class MultiHeadsAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "Error, so chieu cua vector embedding phai chia het duoc cho so luong heads"
        self.d_k = d_model / h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0, -1e9)
        attention_score = attention_score.softmax(dim = -1) #batch, h, seq_len, seq_len
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score
        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) #dua head ve vi tri thu 2
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_score = MultiHeadsAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #(batch, h, seq, d_k) -> (batch, seq, h, d_k) -> (batch, seq, d_model), -1 trong view de pytorch tu dong tinh toan chieu do, contiguos de luu lien tuc thay doi cua x dam bao cho ve sau thay doi dung nhu du kien
        x = x.transpose(1, 2).contiguos().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
    


#x la dau vao dau tien, sublayer la function tinh toan x sau khi xu li, o day hoi khac mot chut la norm output truoc roi moi cong voi ket qua cu
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
     
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadsAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for i in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) #layer la tung doi tuong EncoderBlock trong ModuleList
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadsAttentionBlock, cross_attention_block: MultiHeadsAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for i in range(3)])
        
    #src mask la mask cho encoder, tgt cho decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
#map tu vao trong vocabulary
class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        #(batch, seq, d_model) -> (batch, seq, vocab)
        return torch.log_softmax(self.linear(x), dim = -1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, linear_layer: LinearLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def linearMethod(self, x):
        return self.linear_layer(x)
    

def build(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, Nx: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    #tao embedding layer
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    #tao positional
    #o doan nay 1 cai positional la duoc roi boi vi chung hoat dong nhu nhau nma ghi ra cho ro rang
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    #co N block encoder va decoder nen cho nay de tao ra cno
    #encoder block
    encoder_bucket = []
    for i in range(Nx):
        encoder_self_attention_block = MultiHeadsAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_bucket.append(encoder_block)

    #decoder block
    decoder_bucket = []
    for i in range(Nx):
        decoder_self_attention_block = MultiHeadsAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadsAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_bucket.append(decoder_block)
    
    #gop lai encoder va decoder chung
    encoder = Encoder(nn.ModuleList(encoder_bucket))
    decoder = Decoder(nn.ModuleList(decoder_bucket))

    #tao lop linear 
    #cho nay de la vocab size cua tgt vi lop linear nay hoat dong o dau ra cua decoder
    linear_layer = LinearLayer(d_model, tgt_vocab_size)

    #tao transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, linear_layer)

    #khoi tao tham so dau tien, dung phuong phap Xavier
    for p in transformer.parameters(): #phuong thuc parameters() duoc ke thua tu nn.Module
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) #phuong thuc nay se tu dong cap nhat tham so cho tung phan tu trong transformer, khong can phai cap nhat tay 
        
    return transformer



