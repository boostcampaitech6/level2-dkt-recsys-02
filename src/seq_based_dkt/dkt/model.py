import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import numpy as np
import math
import torch.nn.functional as F

class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concatentaed Embedding Projection
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
    def forward(self, test, question, tag, correct, mask, interaction):
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


#################### Transformer_based ####################
# 한 sequence 내에서 계산
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        """
        Q, K, V : (batch_size, num_heads, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        """
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units) # (batch_size, num_heads, max_len, max_len)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # (batch_size, num_heads, max_len, hidden_units) / # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_O = nn.Linear(hidden_units * num_heads, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        """
        enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        
        """
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # (batch_size, num_heads, max_len, hidden_units)
        output, attn_dist = self.attention(Q, K, V, mask) # output : (batch_size, num_heads, max_len, hidden_units) / attn_dist : (batch_size, num_heads, max_len, max_len)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() # (batch_size, max_len, num_heads, hidden_units) / contiguous() : 가변적 메모리 할당
        output = output.view(batch_size, seqlen, -1) # (batch_size, max_len, hidden_units * num_heads)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual) # (batch_size, max_len, hidden_units)
        return output, attn_dist

# FFN, IS에서도 똑같이
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()

        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        # activation relu -> gelu ?
        output = self.W_2(F.relu(self.dropout(self.W_1(x))))
        output = self.layerNorm(self.dropout(output) + residual)
        return output

### IntersampleAttention ###
class IntersampleAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        """
        scale : math.sqrt(self.hidden_units)
        *** hidden_units는 앞서 Q, K, V를 생성할 때 self.hidden_units // self.num_heads 이렇게 정의, 변수 이름만 같음
        Q : (batch_size, num_heads, seq_len, hidden_units)
        K : (num_heads, batch_size, seq_len, hidden_units)
        Q_expanded : (batch_size, num_heads, 1, seq_len, hidden_units)
        K_expanded : (1, num_heads, batch_size, seq_len, hidden_units)
        """
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.scale = math.sqrt(self.hidden_units)
        self.dropout=nn.Dropout(self.dropout_rate)

    def forward(self, Q, K, V, mask):
        """
        Q, K, V : (batch_size, num_heads, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)   attention 계산 시 사용되는 마스크
        """
        Q_expanded = Q.unsqueeze(2)
        K_expanded = K.unsqueeze(0)
        # 확장된 Q와 K 벡터 간의 내적을 계산하여 어텐션 스코어를 구함
        # 결과 shape: (batch_size, num_heads, batch_size, seq_len, seq_len)
        attn_score = torch.matmul(Q_expanded, K_expanded.transpose(-2, -1)) / self.scale
        attn_score = attn_score.squeeze(0)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)

        # softmax를 통해 어텐션 가중치를 정규화
        attn_weights = self.dropout(F.softmax(attn_score, dim=-1))
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

class IS_MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.intersample_attention = IntersampleAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        """
        enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        
        """
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # (batch_size, max_len, hidden_units) --> (batch_size, seqlen, num_heads, new_hidden_units)
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # (batch_size, num_heads, max_len, hidden_units)
        output, attn_dist = self.intersample_attention(Q, K, V, mask) # output : (batch_size, num_heads, max_len, hidden_units) / attn_dist : (batch_size, num_heads, max_len, max_len)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() # (batch_size, max_len, num_heads, hidden_units) / contiguous() : 가변적 메모리 할당
        output = output.view(batch_size, seqlen, -1) # (batch_size, max_len, hidden_units * num_heads)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual) # (batch_size, max_len, hidden_units)
        # 
        return output, attn_dist

# Decoder
class DE_MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, Q_input ,KV_input, mask):
        """
        enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        
        """
        Q_residual = Q_input # residual connection을 위해 residual 부분을 저장
        KV_residual = KV_input
        batch_size, seqlen = Q_input.size(0), Q_input.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # (batch_size, max_len, hidden_units) --> (batch_size, seqlen, num_heads, new_hidden_units)
        Q = self.W_Q(Q_input).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)
        K = self.W_K(KV_input).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)
        V = self.W_V(KV_input).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads) # (batch_size, max_len, num_heads, hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # (batch_size, num_heads, max_len, hidden_units)
        output, attn_dist = self.attention(Q, K, V, mask) # output : (batch_size, num_heads, max_len, hidden_units) / attn_dist : (batch_size, num_heads, max_len, max_len)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() # (batch_size, max_len, num_heads, hidden_units) / contiguous() : 가변적 메모리 할당
        output = output.view(batch_size, seqlen, -1) # (batch_size, max_len, hidden_units * num_heads)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + Q_residual) # (batch_size, max_len, hidden_units)
        # 
        return output, attn_dist


class SAINTBlock_Encoder(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)
        self.intersample_attention= IS_MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.intersample_FFN = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        """
        input_enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        """
        # sequence내에서
        output_enc_1, attn_dist = self.attention(input_enc, mask)
        output_enc_1 = self.pointwise_feedforward(output_enc_1)
        # 모든 sample에 대해(batch)
        output_enc_2, attn_dist = self.intersample_attention(output_enc_1, mask)
        output_enc_2 = self.intersample_FFN(output_enc_2)

        return output_enc_2, attn_dist

class SAINTBlock_Decoder(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)
        self.decoder_attention= DE_MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        #self.intersample_attention= IS_MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        #self.intersample_FFN = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, Q_input_enc, KV_input, mask):
        """
        input_enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        """
        Q_output, attn_dist = self.attention(Q_input_enc, mask)
        output, attn_dist = self.DE_MultiHeadAttention(Q_output, KV_input, mask)
        output = self.pointwise_feedforward(output)

        return output, attn_dist


class SASRecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        """
        input_enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        """
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist

########## EmBedding ##########
# PastEncoder
class PastEncoderEmbedding(nn.Module):
    def __init__(self,
                past_cat_en_list,
                past_num_en_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 10,
                ):
        super().__init__()
        """
        past_cat_en_list = {
            "encoder에 들어갈 Feature": past data,
            ...
        },
        past_num_en_list = {
            "encoder에 들어갈 Feature": past data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim//3
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_testTag = n_testTag

        self.past_test_embed = nn.Embedding(n_tests+1 , self.intd)
        self.past_question_embed = nn.Embedding(n_questions +1 , self.intd)
        self.past_tag_embed = nn.Embedding(n_tags +1 , self.intd)
        self.past_testTag_embed = nn.Embedding(10 , self.intd)

        self.past_cat_emb=nn.Sequential(nn.Linear(self.intd*4, self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(past_num_en_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, past_cat_en_list, past_num_en_list):
        past_test_embed = self.past_test_embed(past_cat_en_list["test"].int())
        past_question_embed = self.past_question_embed(past_cat_en_list["question"].int())
        past_tag_embed = self.past_tag_embed(past_cat_en_list["tag"].int())
        past_testTag_embed = self.past_testTag_embed(past_cat_en_list["testTag"].int())
        
        past_cat_emb_list=[past_test_embed, past_question_embed, past_tag_embed, past_testTag_embed]
        past_cat_emb = torch.concat(past_cat_emb_list, dim=-1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)

        past_num_keys = [key for key in past_num_en_list.keys()]
        past_num_Linear_list = []
        for key in past_num_keys:
            past_num_Linear_list.append(past_num_en_list[key].unsqueeze(dim=2))
        
        past_num_concat = torch.concat(past_num_Linear_list).float()
        num_linear = self.num_linear(past_num_concat)

        past_emb = torch.concat([past_cat_emb, num_linear], dim=-1)
        past_emb = self.emb_layernorm(past_emb)

        return past_emb

# PastDecoder
class PastDecoderEmbedding(nn.Module):
    def __init__(self,
                past_cat_de_list,
                past_num_de_list,
                hidden_dim: int = 64,
                n_interaction: int = 10
                ):
        super().__init__()
        """
        past_cat_de_list = {
            "decoder에 들어갈 Feature": past data,
            ...
        },
        past_num_de_list = {
            "decoder에 들어갈 Feature": past data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim//3
        self.past_testTag_embed = nn.Embedding(10 , self.intd)
        self.past_interaction_embed = nn.Embedding(10 , self.intd)

        self.past_cat_emb=nn.Sequential(nn.Linear(self.intd*2, self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(past_num_de_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, past_cat_de_list, past_num_en_list):
        past_testTag_embed = self.past_testTag_embed(past_cat_de_list["testTag"].int())
        past_interaction_embed = self.past_interaction_embed(past_cat_de_list["interaction"].int())
        
        past_cat_emb_list=[past_testTag_embed, past_interaction_embed]
        past_cat_emb = torch.concat(past_cat_emb_list, dim=-1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)

        past_num_keys = [key for key in past_num_en_list.keys()]
        past_num_Linear_list = []
        for key in past_num_keys:
            past_num_Linear_list.append(past_num_en_list[key].unsqueeze(dim=2))
        
        past_num_concat = torch.concat(past_num_Linear_list).float()
        num_linear = self.num_linear(past_num_concat)

        past_emb = torch.concat([past_cat_emb, num_linear], dim=-1)
        past_emb = self.emb_layernorm(past_emb)

        return past_emb

# CurrentEncoder
class CurrentEncoderEmbedding(nn.Module):
    def __init__(self,
                current_cat_en_list,
                current_num_en_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 10,
                ):
        super().__init__()
        """
        current_cat_en_list = {
            "encoder에 들어갈 Feature": current data,
            ...
        },
        current_num_en_list = {
            "encoder에 들어갈 Feature": current data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim//3
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_testTag = n_testTag

        self.current_test_embed = nn.Embedding(n_tests+1 , self.intd)
        self.current_question_embed = nn.Embedding(n_questions +1 , self.intd)
        self.current_tag_embed = nn.Embedding(n_tags +1 , self.intd)
        self.current_testTag_embed = nn.Embedding(10 , self.intd)

        self.current_cat_emb=nn.Sequential(nn.Linear(self.intd*4, self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(current_num_en_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, current_cat_en_list, current_num_en_list):
        current_test_embed = self.current_test_embed(current_cat_en_list["test"].int())
        current_question_embed = self.current_question_embed(current_cat_en_list["question"].int())
        current_tag_embed = self.current_tag_embed(current_cat_en_list["tag"].int())
        current_testTag_embed = self.current_testTag_embed(current_cat_en_list["testTag"].int())
        
        current_cat_emb_list=[current_test_embed, current_question_embed, current_tag_embed, current_testTag_embed]
        current_cat_emb = torch.concat(current_cat_emb_list, dim=-1)
        current_cat_emb = self.current_cat_emb(current_cat_emb)

        current_num_keys = [key for key in current_num_en_list.keys()]
        current_num_Linear_list = []
        for key in current_num_keys:
            current_num_Linear_list.append(current_num_en_list[key].unsqueeze(dim=2))
        
        current_num_concat = torch.concat(current_num_Linear_list).float()
        num_linear = self.num_linear(current_num_concat)

        current_emb = torch.concat([current_cat_emb, num_linear], dim=-1)
        current_emb = self.emb_layernorm(current_emb)

        return current_emb

# CurrentDecoder
class CurrentDecoderEmbedding(nn.Module):
    def __init__(self,
                current_cat_de_list,
                current_num_de_list,
                hidden_dim: int = 64,
                n_interaction: int = 10
                ):
        super().__init__()
        """
        current_cat_de_list = {
            "decoder에 들어갈 Feature": current data,
            ...
        },
        current_num_de_list = {
            "decoder에 들어갈 Feature": current data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim//3
        self.current_testTag_embed = nn.Embedding(10 , self.intd)
        self.current_interaction_embed = nn.Embedding(10 , self.intd)

        self.current_cat_emb=nn.Sequential(nn.Linear(self.intd*2, self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(current_num_de_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, current_cat_de_list, current_num_de_list):
        current_testTag_embed = self.current_testTag_embed(current_cat_de_list["testTag"].int())
        current_interaction_embed = self.current_interaction_embed(current_cat_de_list["interaction"].int())
        
        current_cat_emb_list=[current_testTag_embed, current_interaction_embed]
        current_cat_emb = torch.concat(current_cat_emb_list, dim=-1)
        current_cat_emb = self.current_cat_emb(current_cat_emb)

        current_num_keys = [key for key in current_num_de_list.keys()]
        current_num_Linear_list = []
        for key in current_num_keys:
            current_num_Linear_list.append(current_num_de_list[key].unsqueeze(dim=2))
        
        current_num_concat = torch.concat(current_num_Linear_list).float()
        num_linear = self.num_linear(current_num_concat)

        current_emb = torch.concat([current_cat_emb, num_linear], dim=-1)
        current_emb = self.emb_layernorm(current_emb)

        return current_emb


########## SAINTRec ##########
class SAINTRec(nn.Module):
    def __init__(self,
                hidden_dim: int = 64, 
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_heads = 2,   
                drop_out: float = 0.5,   # 원래는 0.1
                max_seq_len: float = 20,
                **kwargs  
                ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers   # attention block 수
        # self.n_tests = n_tests
        # self.n_questions = n_questions
        # self.n_tags = n_tags
        # self.n_heads = n_heads   # self-attention
        self.drop_out=drop_out
        self.max_len = max_seq_len
        #self.n_conti_features = n_conti_features
        self.embed_size=64
        
        hd, intd = hidden_dim, hidden_dim // 3   # hyper-parameter

        # Encoder, Decoder blocks
        self.past_encoder_blocks = nn.ModuleList([SAINTBlock_Encoder(n_heads, hd, drop_out) for _ in range(n_layers)])
        self.past_decoder_blocks = nn.ModuleList([SAINTBlock_Decoder(n_heads, hd, drop_out) for _ in range(n_layers)])

        # LSTM layer
        self.past_lstm = nn.LSTM(
            input_size = hd,
            hidden_size = hd,
            num_layers = 1,
            batch_first = True,
            bidirectional = False,
            dropout = drop_out,
            )

        # Encoder, Decoder blocks
        self.current_encoder_blocks = nn.ModuleList([SAINTBlock_Encoder(n_heads, hd, drop_out) for _ in range(n_layers)])
        self.current_decoder_blocks = nn.ModuleList([SAINTBlock_Decoder(n_heads, hd, drop_out) for _ in range(n_layers)])

        # LSTM layer
        self.current_lstm = nn.LSTM(
            input_size = hd,
            hidden_size = hd,
            num_layers = 1,
            batch_first = True,
            bidirectional = False,
            dropout = drop_out,
            )

        # predict layer
        self.dropout = nn.Dropout(drop_out)
        self.predict_layer = nn.Sequential(
            nn.Linear(hd, 1),
            nn.Sigmoid()
        )

    def forward(self, past_encoder_feat, past_decoder_feat, current_encoder_feat, current_decoder_feat):
        #batch_size = interaction.size(0) 뒤에 쓰이는지 확인

        # past_embedding
        past_emb = PastEncoderEmbedding(past_encoder_feat["past_cat_en_list"],past_encoder_feat["past_num_en_list"])
        past_emb = PastDecoderEmbedding(past_decoder_feat["past_cat_de_list"],past_decoder_feat["past_num_de_list"])

        # masking 
        # mask_pad = torch.BoolTensor((correct > 0).to('cpu')).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, self.max_len, self.max_len)), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        # mask = (mask_pad & mask_time).to('cuda') # (batch_size, 1, max_len, max_len)
        mask = mask_time.to('cuda')
        
        # Encoder
        for block in self.past_encoder_blocks:
            past_emb, attn_dist = block(past_emb, mask)

        # lstm
        past_emb, _ = self.past_lstm(past_emb)

        # Decoder
        for block in self.past_decoder_blocks:
            past_emb, attn_dist = block(past_emb, mask)


        # current_embedding
        current_emb = currentEncoderEmbedding(current_encoder_feat["current_cat_en_list"],current_encoder_feat["current_num_en_list"])
        current_emb = currentDecoderEmbedding(current_decoder_feat["current_cat_de_list"],current_decoder_feat["current_num_de_list"])

        # # masking 
        # # mask_pad = torch.BoolTensor((correct > 0).to('cpu')).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        # mask_time = (1 - torch.triu(torch.ones((1, 1, self.max_len, self.max_len)), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        # # mask = (mask_pad & mask_time).to('cuda') # (batch_size, 1, max_len, max_len)
        # mask = mask_time.to('cuda')
        
        # Encoder
        for block in self.current_encoder_blocks:
            current_emb, attn_dist = block(current_emb, mask)

        # lstm
        current_emb, _ = self.current_lstm(current_emb)

        # Decoder
        for block in self.current_decoder_blocks:
            current_emb, attn_dist = block(current_emb, mask)

        final_emb = torch.concat([past_emb, current_emb], dim = -1)

        output = self.predict_layer(self.dropout(final_emb))

        return output.squeeze(dim=-1)





##########  SASRec_pos ##########
# 성능 안 좋음 폐기! 
class SASRec_pos(nn.Module):
    def __init__(self,
                hidden_dim: int = 64, 
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_heads = 2,   
                drop_out: float = 0.5,   # 원래는 0.1
                max_seq_len: float = 20,
                n_conti_features: int = 3,
                **kwargs  
                ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_heads = n_heads   # self-attention
        self.drop_out=drop_out
        self.max_len = max_seq_len
        self.n_conti_features = n_conti_features
        self.embed_size=64
        
        hd, intd = hidden_dim, hidden_dim // 3   # hyper-parameter
        ######################### past #########################
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.past_embedding_interaction = nn.Embedding(3, intd) 
        self.past_embedding_test = nn.Embedding(n_tests + 1, intd)
        self.past_embedding_question = nn.Embedding(n_questions + 1, intd)
        self.past_embedding_tag = nn.Embedding(n_tags + 1, intd)
        #self.embedding_testTag=nn.Embedding(10, intd)   # 학년으로 추정
        self.past_correct_emb = nn.Embedding(3, hd, padding_idx = 0)
        # position embedding
        self.pos_emb = nn.Embedding(self.max_len, hd)

        # 수치형 변환 + LayerNorm
        self.past_conti_emb=nn.Sequential(nn.Linear(3,hd//2),nn.LayerNorm(hd // 2, eps=1e-6))

        # 범주형 (concat emb) linear + LayerNorm
        self.past_cat_emb=nn.Sequential(nn.Linear(intd*4, hd//2),
                                    nn.LayerNorm(hd//2, eps=1e-6))
        # embedding LayerNorm
        self.emb_layernorm = nn.LayerNorm(hd, eps=1e-6)

        # LSTM layer
        self.past_lstm = nn.LSTM(
            input_size = hd,
            hidden_size = hd,
            num_layers = 1,
            batch_first = True,
            bidirectional = False,
            dropout = drop_out,
            )

        self.past_blocks = nn.ModuleList([SASRecBlock(n_heads, hd, drop_out) for _ in range(n_layers)])

        # predict layer
        self.dropout = nn.Dropout(drop_out)
        self.predict_layer = nn.Sequential(
            nn.Linear(hd, 1),
            nn.Sigmoid()
        )

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng,):
        batch_size = interaction.size(0)
        ######################### past #########################
        # past cat embedding
        past_embed_interaction = self.past_embedding_interaction(interaction.int())
        past_embed_test = self.past_embedding_test(test.int())
        past_embed_question = self.past_embedding_question(question.int())
        past_embed_tag = self.past_embedding_tag(tag.int())

        # concat cat_embs
        past_cat_emb = torch.concat([past_embed_interaction, past_embed_test, 
                                    past_embed_question, past_embed_tag], dim=-1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)


        ### 3pl dim ↑
        past_3pl_concat=torch.concat([dffclt.unsqueeze(dim=2), 
                                    dscrmn.unsqueeze(dim=2), 
                                    gussng.unsqueeze(dim=2)],dim=-1)
        past_3pl_concat= past_3pl_concat.float()
        past_conti_emb=self.past_conti_emb(past_3pl_concat)

        # concat past_cat + past_conti
        past_emb = torch.concat([past_cat_emb, past_conti_emb], dim = -1)
        # past_emb += self.past_correct_emb(correct.int())

        # position embedding
        positions = np.tile(np.array(range(self.max_len)), [batch_size, 1])
        past_emb += self.pos_emb(torch.LongTensor(positions).to("cuda")) # (batch_size, max_len, hidden_units)

        past_emb = self.emb_layernorm(past_emb)

        # masking 
        mask_pad = torch.BoolTensor((correct > 0).to('cpu')).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, correct.size(1), correct.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to('cuda') # (batch_size, 1, max_len, max_len)
        
        # self-attention
        for block in self.past_blocks:
            past_emb, attn_dist = block(past_emb, mask)

        # lstm
        past_emb, _ = self.past_lstm(past_emb)

        output =self.predict_layer(self.dropout(past_emb))
        
        return output.squeeze(dim=-1)






