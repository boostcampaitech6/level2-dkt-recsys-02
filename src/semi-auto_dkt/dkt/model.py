import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

import numpy as np
import math
import torch.nn.functional as F
import re

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




############################## SAINT_tuning ##############################
###### Past Embedding Layer ######
class PastEncoderEmbedding(nn.Module):
    def __init__(self,
                past_en_cat_list,
                past_en_num_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 10,
                ):
        super().__init__()
        """
        past_en_cat_list = {
            "encoder에 들어갈 Feature": past data,
            ...
        },
        past_en_num_list = {
            "encoder에 들어갈 Feature": past data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim // 3
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_testTag = n_testTag

        self.past_test_embed = nn.Embedding(n_tests+1, self.intd)
        self.past_question_embed = nn.Embedding(n_questions+1, self.intd)
        self.past_tag_embed = nn.Embedding(n_tags+1, self.intd)
        self.past_testTag_embed = nn.Embedding(11, self.intd)

        self.past_cat_emb = nn.Sequential(nn.Linear(self.intd * 4, self.hidden_dim // 2),
                                          nn.LayerNorm(self.hidden_dim // 2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(past_en_num_list), self.hidden_dim // 2),
                                        nn.LayerNorm(self.hidden_dim // 2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

    def forward(self, past_en_cat_list, past_en_num_list):
        past_test_embed = self.past_test_embed(past_en_cat_list["past_test"].int())
        past_question_embed = self.past_question_embed(past_en_cat_list["past_question"].int())
        past_tag_embed = self.past_tag_embed(past_en_cat_list["past_tag"].int())
        past_testTag_embed = self.past_testTag_embed(past_en_cat_list["past_testTag"].int())
        
        past_cat_emb_list = [past_test_embed, past_question_embed, past_tag_embed, past_testTag_embed]
        past_cat_emb = torch.concat(past_cat_emb_list, dim=-1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)

        batch_size, seq_len = past_cat_emb.size(0), past_cat_emb.size(1)

        past_num_keys = [key for key in past_en_num_list.keys()]
        past_num_Linear_list = []
        for key in past_num_keys:
            past_num_Linear_list.append(past_en_num_list[key])

        past_num_concat = torch.concat(past_num_Linear_list).float().reshape(batch_size, seq_len, -1)
        num_linear = self.num_linear(past_num_concat)

        past_emb = torch.concat([past_cat_emb, num_linear], dim=-1)
        past_emb = self.emb_layernorm(past_emb)

        return past_emb


class PastDecoderEmbedding(nn.Module):
    def __init__(self,
                past_de_cat_list,
                past_de_num_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 11,
                ):
        super().__init__()
        """
        past_de_cat_list = {
            "decoder에 들어갈 Feature": past data,
            ...
        },
        past_de_num_list = {
            "decoder에 들어갈 Feature": past data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim // 3

        self.past_testTag_embed = nn.Embedding(11, self.intd)
        self.past_interaction_embed = nn.Embedding(3, self.intd)

        self.past_cat_emb = nn.Sequential(nn.Linear(self.intd * 2, self.hidden_dim // 2),
                                    nn.LayerNorm(self.hidden_dim // 2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(past_de_num_list), self.hidden_dim // 2),
                                    nn.LayerNorm(self.hidden_dim // 2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

    def forward(self, past_de_cat_list, past_de_num_list):
        past_testTag_embed = self.past_testTag_embed(past_de_cat_list["past_testTag"].int())
        past_interaction_embed = self.past_interaction_embed(past_de_cat_list["past_interaction"].int())

        past_cat_emb_list = [past_testTag_embed, past_interaction_embed]
        past_cat_emb = torch.concat(past_cat_emb_list, dim=-1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)

        batch_size, seq_len = past_cat_emb.size(0), past_cat_emb.size(1)

        past_num_keys = [key for key in past_de_num_list.keys()]
        past_num_Linear_list = []
        for key in past_num_keys:
            past_num_Linear_list.append(past_de_num_list[key])

        past_num_concat = torch.concat(past_num_Linear_list).float().reshape(batch_size, seq_len, -1)
        num_linear = self.num_linear(past_num_concat)

        past_emb = torch.concat([past_cat_emb, num_linear], dim=-1)
        past_emb = self.emb_layernorm(past_emb)

        return past_emb

###### Currnet Embedding Layer ######
class CurrentEncoderEmbedding(nn.Module):
    def __init__(self,
                current_en_cat_list,
                current_en_num_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 10,
                ):
        super().__init__()
        """
        current_en_cat_list = {
            "encoder에 들어갈 Feature": current data,
            ...
        },
        current_en_num_list = {
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
        self.current_testTag_embed = nn.Embedding(11 , self.intd)

        self.current_cat_emb=nn.Sequential(nn.Linear(self.intd*4, self.hidden_dim//2)
                                          ,nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(current_en_num_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, current_en_cat_list, current_en_num_list):
        # print(current_en_cat_list["current_test"])
        # print(torch.unique(current_en_cat_list["current_test"]))
        current_test_embed = self.current_test_embed(current_en_cat_list["current_test"].int())
        current_question_embed = self.current_question_embed(current_en_cat_list["current_question"].int())
        current_tag_embed = self.current_tag_embed(current_en_cat_list["current_tag"].int())
        # print(torch.unique(current_en_cat_list["current_testTag"]))
        current_testTag_embed = self.current_testTag_embed(current_en_cat_list["current_testTag"].int())
        
        current_cat_emb_list=[current_test_embed, current_question_embed, current_tag_embed, current_testTag_embed]
        current_cat_emb = torch.concat(current_cat_emb_list, dim=-1)
        current_cat_emb = self.current_cat_emb(current_cat_emb)   # (batch_size, seq_len, self.hidden_dim//2)

        batch_size, seq_len = current_cat_emb.size(0), current_cat_emb.size(1)

        current_num_keys = [key for key in current_en_num_list.keys()]
        current_num_Linear_list = []
        for key in current_num_keys:
            current_num_Linear_list.append(current_en_num_list[key])
        
        current_num_concat = torch.concat(current_num_Linear_list).float().reshape(batch_size, seq_len, -1)  
        # (batch_size, seq_len, len(current_num_Linear_list))
        num_linear = self.num_linear(current_num_concat)

        current_emb = torch.concat([current_cat_emb, num_linear], dim=-1)   # (batch_size, seq_len, self.hidden_dim)
        current_emb = self.emb_layernorm(current_emb) 

        return current_emb


class CurrentDecoderEmbedding(nn.Module):
    def __init__(self,
                current_de_cat_list,
                current_de_num_list,
                hidden_dim: int = 64,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_testTag: int = 11,
                ):
        super().__init__()
        """
        current_de_cat_list = {
            "decoder에 들어갈 Feature": current data,
            ...
        },
        current_de_num_list = {
            "decoder에 들어갈 Feature": current data,
            ...
        }
        """
        self.hidden_dim = hidden_dim
        self.intd = hidden_dim//3

        self.current_testTag_embed = nn.Embedding(11 , self.intd)
        self.current_interaction_embed = nn.Embedding(3 , self.intd)

        self.current_cat_emb=nn.Sequential(nn.Linear(self.intd*2, self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.num_linear = nn.Sequential(nn.Linear(len(current_de_num_list), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2, eps=1e-6))

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)


    def forward(self, current_de_cat_list, current_de_num_list):
        current_testTag_embed = self.current_testTag_embed(current_de_cat_list["current_testTag"].int())
        current_interaction_embed = self.current_interaction_embed(current_de_cat_list["current_interaction"].int())
        
        current_cat_emb_list=[current_testTag_embed, current_interaction_embed]
        current_cat_emb = torch.concat(current_cat_emb_list, dim=-1)
        # print("print(current_cat_emb.size()): ",current_cat_emb.size())
        current_cat_emb = self.current_cat_emb(current_cat_emb)   # (batch_size, seq_len, self.hidden_dim//2)
        # print("print(current_cat_emb.size()) 2:",current_cat_emb.size())
        batch_size, seq_len = current_cat_emb.size(0), current_cat_emb.size(1)

        current_num_keys = [key for key in current_de_num_list.keys()]
        current_num_Linear_list = []
        for key in current_num_keys:
            current_num_Linear_list.append(current_de_num_list[key])
        
        current_num_concat = torch.concat(current_num_Linear_list).float().reshape(batch_size, seq_len, -1)  
        # (batch_size, seq_len, len(current_num_Linear_list))
        num_linear = self.num_linear(current_num_concat)   # (batch_size, seq_len, self.hidden_dim//2)

        current_emb = torch.concat([current_cat_emb, num_linear], dim=-1)   # (batch_size, seq_len, self.hidden_dim)
        current_emb = self.emb_layernorm(current_emb) 

        return current_emb


###### SAINT ######

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(ModelBase):
    def __init__(
            self,
            hidden_dim: int = 64,
            n_layers: int = 1,
            n_tests: int = 1538,
            n_questions: int = 9455,
            n_tags: int = 913,
            n_heads: int = 2,
            drop_out: float = 0.1,
            max_seq_len: float = 20,
            # n_conti_features: int = 13, ### 추가한 feature 개수
            Tfixup: bool = True,
            **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.hidden_dim = hidden_dim
        self.dropout = drop_out
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        # self.n_conti_features = n_conti_features
        self.Tfixup = Tfixup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Embedding
        # self.current_en_emb_layer = CurrentEncoderEmbedding(current_encoder_feat["current_en_cat_list"], current_encoder_feat["current_en_num_list"])
        # self.current_de_emb_layer = CurrentDecoderEmbedding(current_decoder_feat["current_de_cat_list"], current_encoder_feat["current_de_num_list"])
        self.past_correct_emb = nn.Embedding(2, self.hidden_dim)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)

        # self.lstm = nn.LSTM(
        #     input_size = self.hidden_dim,
        #     hidden_size = self.hidden_dim,
        #     num_layers = n_layers,
        #     batch_first = True,
        #     bidirectional = False,
        #     dropout = self.dropout,
        #     )

        self.past_transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='gelu')

        self.current_transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='gelu')

        self.fc = nn.Linear(self.hidden_dim*2, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None


        # T-Fixup
        if self.Tfixup:
            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")


    def tfixup_initialization(self):
        # padding idx의 경우 모두 0으로 통일
        padding_idx = 0

        for name, param in self.named_parameters():
            if len(param.shape) == 1:
                continue
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링
        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.n_layers) ** (-1 / 4) * param
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)


    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, past_encoder_feat, past_decoder_feat,
                        current_encoder_feat, current_decoder_feat):
        
        # batch_size = interaction.size(0)
        # seq_len = interaction.size(1)

        # encoder/decoder Embedding

        #print("current_encoder_feat.keys(): ", current_encoder_feat.keys())
        #print("current_decoder_feat.keys(): ", current_decoder_feat.keys() )

        # print(current_encoder_feat["current_en_cat_list"]["current_testTag"])
        ## past
        past_en_emb_layer = PastEncoderEmbedding(past_encoder_feat["past_en_cat_list"], past_encoder_feat["past_en_num_list"])
        past_de_emb_layer = PastDecoderEmbedding(past_decoder_feat["past_de_cat_list"], past_decoder_feat["past_de_num_list"])
        # print("past_encoder_feat.keys(): ", past_encoder_feat.keys())
        past_en_emb = past_en_emb_layer(past_encoder_feat["past_en_cat_list"], past_encoder_feat["past_en_num_list"])
        past_de_emb = past_de_emb_layer(past_decoder_feat["past_de_cat_list"], past_decoder_feat["past_de_num_list"])
        # print("past_encoder_feat[\"past_en_cat_list\"].keys(): ", past_encoder_feat["past_en_cat_list"].keys())

        past_correct_emb = self.past_correct_emb(past_encoder_feat["past_en_cat_list"]["past_correct"].int())

        past_en_emb += past_correct_emb

    
        ## current 
        current_en_emb_layer = CurrentEncoderEmbedding(current_encoder_feat["current_en_cat_list"], current_encoder_feat["current_en_num_list"])
        current_de_emb_layer = CurrentDecoderEmbedding(current_decoder_feat["current_de_cat_list"], current_decoder_feat["current_de_num_list"])

        current_en_emb = current_en_emb_layer(current_encoder_feat["current_en_cat_list"], current_encoder_feat["current_en_num_list"])
        current_de_emb = current_de_emb_layer(current_decoder_feat["current_de_cat_list"], current_decoder_feat["current_de_num_list"])

        batch_size, seq_len = current_en_emb.size(0), current_en_emb.size(1)
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

        # past
        past_en_emb = past_en_emb.permute(1, 0, 2)
        past_de_emb = past_de_emb.permute(1, 0, 2)

        # current
        current_en_emb = current_en_emb.permute(1, 0, 2)
        current_de_emb = current_de_emb.permute(1, 0, 2)

        # current_en_emb, _ = self.lstm(current_en_emb)
        # current_de_emb, _ = self.lstm(current_de_emb)

        # past Positional encoding
        past_en_emb = self.pos_encoder(past_en_emb)
        past_de_emb = self.pos_decoder(past_de_emb)

        # current Positional encoding
        current_en_emb = self.pos_encoder(current_en_emb)
        current_de_emb = self.pos_decoder(current_de_emb)

        past_out = self.past_transformer(past_en_emb, past_de_emb.to(torch.float32),
                        src_mask=self.enc_mask.to(torch.float32),
                        tgt_mask=self.dec_mask.to(torch.float32),
                        memory_mask=self.enc_dec_mask.to(torch.float32))

        current_out = self.current_transformer(current_en_emb, current_de_emb.to(torch.float32),
                               src_mask=self.enc_mask.to(torch.float32),
                               tgt_mask=self.dec_mask.to(torch.float32),
                               memory_mask=self.enc_dec_mask.to(torch.float32))

        past_out = past_out.permute(1, 0, 2)
        current_out = current_out.permute(1, 0, 2)

        past_out = past_out.contiguous().view(batch_size, -1, self.hidden_dim)
        current_out = current_out.contiguous().view(batch_size, -1, self.hidden_dim)

        out= torch.concat([past_out, current_out], dim=-1)

        out = self.fc(out).view(batch_size, -1)

        return out    


# class Saint(ModelBase):
#     def __init__(
#             self,
#             hidden_dim: int = 64,
#             n_layers: int = 2,
#             n_tests: int = 1538,
#             n_questions: int = 9455,
#             n_tags: int = 913,
#             n_heads: int = 2,
#             drop_out: float = 0.1,
#             max_seq_len: float = 20,
#             n_conti_features: int = 13, ### 추가한 feature 개수
#             Tfixup: bool = True,
#             **kwargs
#     ):
#         super().__init__(
#             hidden_dim,
#             n_layers,
#             n_tests,
#             n_questions,
#             n_tags,
#         )

#         self.dropout = drop_out
#         self.n_heads = n_heads
#         self.max_seq_len = max_seq_len
#         self.n_conti_features = n_conti_features
#         self.Tfixup = Tfixup
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # Embedding
#         self.current_en_emb_layer = CurrentEncoderEmbedding(current_encoder_feat["current_en_cat_list"], current_encoder_feat["current_en_num_list"])
#         self.current_de_emb_layer = CurrentDecoderEmbedding(current_decoder_feat["current_de_cat_list"], current_encoder_feat["current_de_num_list"])

#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)
#         self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)

#         self.transformer = nn.Transformer(
#             d_model=self.hidden_dim,
#             nhead=self.n_heads,
#             num_encoder_layers=self.n_layers,
#             num_decoder_layers=self.n_layers,
#             dim_feedforward=self.hidden_dim,
#             dropout=self.dropout,
#             activation='gelu')

#         self.fc = nn.Linear(self.hidden_dim, 1)
#         self.activation = nn.Sigmoid()

#         self.enc_mask = None
#         self.dec_mask = None
#         self.enc_dec_mask = None


#         # T-Fixup
#         if self.Tfixup:
#             # 초기화 (Initialization)
#             self.tfixup_initialization()
#             print("T-Fixup Initialization Done")

#             # 스케일링 (Scaling)
#             self.tfixup_scaling()
#             print(f"T-Fixup Scaling Done")


#     def tfixup_initialization(self):
#         # padding idx의 경우 모두 0으로 통일
#         padding_idx = 0

#         for name, param in self.named_parameters():
#             if len(param.shape) == 1:
#                 continue
#             if re.match(r'^embedding*', name):
#                 nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
#                 nn.init.constant_(param[padding_idx], 0)
#             elif re.match(r'.*ln.*|.*bn.*', name):
#                 continue
#             elif re.match(r'.*weight*', name):
#                 # nn.init.xavier_uniform_(param)
#                 nn.init.xavier_normal_(param)


#     def tfixup_scaling(self):
#         temp_state_dict = {}

#         # 특정 layer들의 값을 스케일링
#         for name, param in self.named_parameters():
#             if re.match(r'^embedding*', name):
#                 temp_state_dict[name] = (9 * self.n_layers) ** (-1 / 4) * param
#             elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
#                 temp_state_dict[name] = (0.67 * (self.n_layers) ** (-1 / 4)) * param
#             elif re.match(r"encoder.*value.weight$", name):
#                 temp_state_dict[name] = (0.67 * (self.n_layers) ** (-1 / 4)) * (param * (2**0.5))

#         for name in self.state_dict():
#             if name not in temp_state_dict:
#                 temp_state_dict[name] = self.state_dict()[name]

#         self.load_state_dict(temp_state_dict)


#     def get_mask(self, seq_len):
#         mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

#         return mask.masked_fill(mask==1, float('-inf'))

#     def forward(self, past_encoder_feat, past_decoder_feat,
#                         current_encoder_feat, current_decoder_feat):
        
#         # batch_size = interaction.size(0)
#         # seq_len = interaction.size(1)

#         # encoder/decoder Embedding
#         current_en_emb = self.current_en_emb_layer(current_encoder_feat["current_cat_en_list"], current_encoder_feat["current_num_en_list"])
#         current_de_emb = self.current_de_emb_layer(current_decoder_feat["current_cat_de_list"], current_encoder_feat["current_num_de_list"])
#         batch_size, seq_len = current_en_emb.size(0), current_en_emb.size(1)

#         # ATTENTION MASK 생성
#         # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
#         # 사실 이렇게 3개로 나눌 필요가 없다
#         if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
#             self.enc_mask = self.get_mask(seq_len).to(self.device)

#         if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
#             self.dec_mask = self.get_mask(seq_len).to(self.device)

#         if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
#             self.enc_dec_mask = self.get_mask(seq_len).to(self.device)


#         current_en_emb = current_en_emb.permute(1, 0, 2)
#         current_de_emb = current_de_emb.permute(1, 0, 2)

#         # Positional encoding
#         current_en_emb = self.pos_encoder(current_en_emb)
#         current_de_emb = self.pos_decoder(current_de_emb)

#         out = self.transformer(current_en_emb, current_de_emb.to(torch.float32),
#                                src_mask=self.enc_mask.to(torch.float32),
#                                tgt_mask=self.dec_mask.to(torch.float32),
#                                memory_mask=self.enc_dec_mask.to(torch.float32))

#         out = out.permute(1, 0, 2)
#         out = out.contiguous().view(batch_size, -1, self.hidden_dim)
#         out = self.fc(out).view(batch_size, -1)

#         return out