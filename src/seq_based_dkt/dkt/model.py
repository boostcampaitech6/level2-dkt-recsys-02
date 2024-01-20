import torch
import torch.nn as nn
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import torch.nn.functional as F


class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_conti_features: int = 13, ### 추가한 feature 개수
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_conti_features = n_conti_features   ### 연속형 feature 개수 정의
        
        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        # padding을 위한 추가적인 범주 위해 +1
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)
        self.embedding_testTag=nn.Embedding(10, 5)

        ### 1->intd 선형 변환 + activation 추가
        self.lin_activation=nn.Sequential(nn.Linear(1,5),nn.Sigmoid()) 

        # Concatentaed Embedding Projection
        # 추가된 연속형 feature 차원 * 개수 (n_conti_features*intd) 추가
        self.comb_proj = nn.Linear(intd * 4 + n_conti_features * 5, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed_testTag = self.embedding_testTag(testTag.int())
        
        ### 3pl dim ↑ & 비선형성 추가
        dffclt_linear=self.lin_activation(dffclt.unsqueeze(dim=2))
        dscrmn_linear=self.lin_activation(dscrmn.unsqueeze(dim=2))
        gussng_linear=self.lin_activation(gussng.unsqueeze(dim=2))
        ### FE ###
        user_correct_answer_linear = self.lin_activation(user_correct_answer.unsqueeze(dim=2))
        user_total_answer_linear = self.lin_activation(user_total_answer.unsqueeze(dim=2))
        user_acc_linear = self.lin_activation(user_acc.unsqueeze(dim=2))
        user_mean_linear = self.lin_activation(user_mean.unsqueeze(dim=2))
        assessment_mean_linear = self.lin_activation(assessment_mean.unsqueeze(dim=2))
        test_mean_linear = self.lin_activation(test_mean.unsqueeze(dim=2))
        knowledgeTag_mean_linear = self.lin_activation(knowledgeTag_mean.unsqueeze(dim=2))
        time_to_solve_linear = self.lin_activation(time_to_solve.unsqueeze(dim=2))
        prior_testTag_frequency_linear = self.lin_activation(prior_testTag_frequency.unsqueeze(dim=2))


        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                dffclt_linear,
                dscrmn_linear,
                gussng_linear,
                embed_testTag,
                user_correct_answer_linear,
                user_total_answer_linear,
                user_acc_linear,
                user_mean_linear,
                assessment_mean_linear,
                test_mean_linear,
                knowledgeTag_mean_linear,
                time_to_solve_linear,
                prior_testTag_frequency_linear,
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
        # 상속받은 부모 클래스의 생성자를 호출하여 초기화 수행 (부모 클래스 init method 사용)
        # 전달한 인자는 LSTM에서 받은 인자로 초기화, 나머지는 부모 클래스에서 정의된 기본값으로 초기화
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

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        dffclt = dffclt,
                                        dscrmn = dscrmn,
                                        gussng = gussng,
                                        testTag = testTag, 
                                        user_correct_answer = user_correct_answer, 
                                        user_total_answer = user_total_answer,
                                        user_acc = user_acc, 
                                        user_mean = user_mean, 
                                        assessment_mean = assessment_mean, 
                                        test_mean = test_mean,
                                        knowledgeTag_mean = knowledgeTag_mean, 
                                        time_to_solve = time_to_solve, 
                                        prior_testTag_frequency= prior_testTag_frequency,
                                        )
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

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        dffclt = dffclt,
                                        dscrmn = dscrmn,
                                        gussng = gussng,
                                        testTag = testTag, 
                                        user_correct_answer = user_correct_answer, 
                                        user_total_answer = user_total_answer,
                                        user_acc = user_acc, 
                                        user_mean = user_mean, 
                                        assessment_mean = assessment_mean, 
                                        test_mean = test_mean,
                                        knowledgeTag_mean = knowledgeTag_mean, 
                                        time_to_solve = time_to_solve, 
                                        prior_testTag_frequency= prior_testTag_frequency,
                                        )

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   # 패널티 부과(학습이 안되도록)
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)   # mask 넣어줌..(에러 안 나게)
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

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        dffclt = dffclt,
                                        dscrmn = dscrmn,
                                        gussng = gussng,
                                        testTag = testTag, 
                                        user_correct_answer = user_correct_answer, 
                                        user_total_answer = user_total_answer,
                                        user_acc = user_acc, 
                                        user_mean = user_mean, 
                                        assessment_mean = assessment_mean, 
                                        test_mean = test_mean,
                                        knowledgeTag_mean = knowledgeTag_mean, 
                                        time_to_solve = time_to_solve, 
                                        prior_testTag_frequency= prior_testTag_frequency,
                                        )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]   # encoder layer에서 최종 결과값
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out



### 추가한 모델 ###
class BERT_LSTM(ModelBase):
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

        ### LSTM 추가 ###
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        dffclt = dffclt,
                                        dscrmn = dscrmn,
                                        gussng = gussng,
                                        testTag = testTag, 
                                        user_correct_answer = user_correct_answer, 
                                        user_total_answer = user_total_answer,
                                        user_acc = user_acc, 
                                        user_mean = user_mean, 
                                        assessment_mean = assessment_mean, 
                                        test_mean = test_mean,
                                        knowledgeTag_mean = knowledgeTag_mean, 
                                        time_to_solve = time_to_solve, 
                                        prior_testTag_frequency= prior_testTag_frequency,
                                        )
        
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]   # encoder layer에서 최종 결과값
        out,_ = self.lstm(out)

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

########## self-attention ##########
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


### SASRec ###
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

class SASRec(nn.Module):   # 원래 SASRec에서 hidden_units(일단 지움)이랑 hidden_dim이랑 같은지 확인
    def __init__(self,
                hidden_dim: int = 64,   # Linear층에서 들어갈 차원 수
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                n_conti_features: int = 11,
                # self-attention에 필요한 인자들
                n_heads = 2,   # 있어야 함 
                drop_out=0.2,   # 있어야 함
                max_seq_len = 5 # 어쩔수없이 받는것 - 사용하지않는인자
                ):
        super(SASRec, self).__init__()
        num_heads = n_heads
        num_layers = n_layers
        self.embedding_size = hidden_dim
        dropout_rate = drop_out
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.n_conti_features = n_conti_features
        
        hd, intd = hidden_dim, hidden_dim // 3   # hyper-parameter
        
        
        #past
        
        ##categorial 변환 layer
        self.past_embedding_interaction = nn.Embedding(3, intd, padding_idx = 0) 
        self.past_embedding_test = nn.Embedding(n_tests + 1, intd, padding_idx = 0)
        self.past_embedding_question = nn.Embedding(n_questions + 1, intd, padding_idx = 0)
        self.past_embedding_tag = nn.Embedding(n_tags + 1, intd, padding_idx = 0)
        self.past_embedding_testTag=nn.Embedding(10, intd, padding_idx = 0)   # 학년으로 추정
        self.past_correct_emb = nn.Embedding(3, hd, padding_idx = 0)

        ## 수치형 변환 layer
        self.past_lin_activation=nn.Sequential(nn.Linear(1,(intd*5)//n_conti_features),nn.Sigmoid())

        ## 범주형 (concat emb) linear + LayerNorm
        self.past_cat_emb=nn.Sequential(nn.Linear(intd*5, hd//2),
                                    nn.LayerNorm(hd//2, eps=1e-6))
        ## 수치형 (concat emb) linear + LayerNorm
        self.past_conti_emb=nn.Sequential(nn.Linear(((intd*5)//n_conti_features)*n_conti_features, hidden_dim//2),
                                    nn.LayerNorm(hd//2, eps=1e-6))
        
        #current
        ##categorial 변환 layer
        self.current_embedding_interaction = nn.Embedding(3, intd, padding_idx = 0) 
        self.current_embedding_test = nn.Embedding(n_tests + 1, intd, padding_idx = 0)
        self.current_embedding_question = nn.Embedding(n_questions + 1, intd, padding_idx = 0)
        self.current_embedding_tag = nn.Embedding(n_tags + 1, intd, padding_idx = 0)
        self.current_embedding_testTag=nn.Embedding(10, intd, padding_idx = 0)   # 학년으로 추정
        self.current_correct_emb = nn.Embedding(3, hd, padding_idx = 0)

        ## 수치형 변환 layer
        self.current_lin_activation=nn.Sequential(nn.Linear(1,(intd*5)//n_conti_features),nn.Sigmoid())

        ## 범주형 (concat emb) linear + LayerNorm
        self.current_cat_emb=nn.Sequential(nn.Linear(intd*5, hd//2),
                                    nn.LayerNorm(hd//2, eps=1e-6))
        ## 수치형 (concat emb) linear + LayerNorm
        self.current_conti_emb=nn.Sequential(nn.Linear(((intd*5)//n_conti_features)*n_conti_features, hidden_dim//2),
                                    nn.LayerNorm(hd//2, eps=1e-6))
        
        
        
        # embedding LayerNorm
        self.emb_layernorm = nn.LayerNorm(hd, eps=1e-6)

        # LSTM layer past
        self.past_lstm = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.embedding_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = dropout_rate,
            )

        # LSTM crruent layer
        self.current_lstm = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.embedding_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = dropout_rate,
            )
        
        self.blocks = nn.ModuleList([SASRecBlock(num_heads, self.embedding_size, dropout_rate) for _ in range(num_layers)])

        # predict layer
        self.dropout = nn.Dropout(dropout_rate)
        self.predict_layer = nn.Sequential(
            nn.Linear(hd * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, past_test, past_question, past_tag, past_correct,
                past_dffclt, past_dscrmn, past_gussng,
                past_testTag, past_user_correct_answer, past_user_total_answer, 
                past_user_acc, past_user_mean, past_relative_answer_mean,
                past_time_to_solve, past_time_to_solve_mean, past_prior_testTag_frequency,
                past_mask, past_interaction,
                current_test, current_question, current_tag, current_correct,
                current_dffclt, current_dscrmn, current_gussng,
                current_testTag, current_user_correct_answer, current_user_total_answer,
                current_user_acc, current_user_mean, current_relative_answer_mean,
                current_time_to_solve, current_time_to_solve_mean, current_prior_testTag_frequency,
                current_mask, current_interaction):

        past_cat_list=['past_test', 'past_question', 'past_tag', 'past_correct', 'past_testTag']
        past_conti_list=['past_dffclt', 'past_dscrmn', 'past_gussng',
                        'past_user_correct_answer', 'past_user_total_answer', 
                        'past_user_acc', 'past_user_mean', 'past_relative_answer_mean',
                        'past_time_to_solve', 'past_time_to_solve_mean', 'past_prior_testTag_frequency']

        current_cat_list=['current_test', 'current_question', 'current_tag', 'current_correct', 'current_testTag']
        current_conti_list=['current_dffclt', 'current_dscrmn', 'current_gussng',
                            'current_user_correct_answer', 'current_user_total_answer',
                            'current_user_acc', 'current_user_mean', 'current_relative_answer_mean',
                            'current_time_to_solve', 'current_time_to_solve_mean', 'current_prior_testTag_frequency']

        # past cat embedding
        past_embed_interaction = self.past_embedding_interaction(past_interaction.int())
        past_embed_test = self.past_embedding_test(past_test.int())
        past_embed_question = self.past_embedding_question(past_question.int())
        past_embed_tag = self.past_embedding_tag(past_tag.int())
        past_embed_testTag = self.past_embedding_testTag(past_testTag.int())

        ### 3pl dim ↑ & 비선형성 추가
        past_dffclt_linear=self.past_lin_activation(past_dffclt.unsqueeze(dim=2))
        past_dscrmn_linear=self.past_lin_activation(past_dscrmn.unsqueeze(dim=2))
        past_gussng_linear=self.past_lin_activation(past_gussng.unsqueeze(dim=2))
        ### FE ###
        past_user_correct_answer_eb = self.past_lin_activation(past_user_correct_answer.unsqueeze(dim=2))
        past_user_total_answer_eb = self.past_lin_activation(past_user_total_answer.unsqueeze(dim=2))
        past_user_acc_eb = self.past_lin_activation(past_user_acc.unsqueeze(dim=2))
        past_user_mean_eb = self.past_lin_activation(past_user_mean.unsqueeze(dim=2))
        past_relative_answer_mean_eb = self.past_lin_activation(past_relative_answer_mean.unsqueeze(dim=2))
        past_time_to_solve_eb = self.past_lin_activation(past_time_to_solve.unsqueeze(dim=2))
        past_time_to_solve_mean_eb = self.past_lin_activation(past_time_to_solve_mean.unsqueeze(dim=2))
        past_prior_testTag_frequency_eb = self.past_lin_activation(past_prior_testTag_frequency.unsqueeze(dim=2))
        
        # concat cat_embs
        past_cat_emb = torch.concat([past_embed_interaction, past_embed_test, 
                                    past_embed_question, past_embed_tag, 
                                    past_embed_testTag], dim=-1)

        # cat_emb -> linear + LN
        past_cat_emb = self.past_cat_emb(past_cat_emb)

        # concat conti_embs
        past_conti_emb = torch.concat([
                                past_dffclt_linear,
                                past_dscrmn_linear,
                                past_gussng_linear,
                                past_user_correct_answer_eb,
                                past_user_total_answer_eb,
                                past_user_acc_eb,
                                past_user_mean_eb,
                                past_relative_answer_mean_eb,
                                past_time_to_solve_eb,
                                past_time_to_solve_mean_eb,
                                past_prior_testTag_frequency_eb
                            ], dim=2)

        # conti_emb -> Linear +LN
        past_conti_emb = self.past_conti_emb(past_conti_emb)

        # concat cat + conti -> final past_emb
        past_emb = torch.concat([past_cat_emb, past_conti_emb], dim = -1)
        # final past_emb + past_correct_emb --> final final past_emb
        past_emb += self.past_correct_emb(past_correct.int())
        # final emb -> LN
        past_emb = self.emb_layernorm(past_emb) # LayerNorm

        # masking 
        mask_pad = torch.BoolTensor(past_correct.to('cpu') > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, past_correct.size(1), past_correct.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to('cuda') # (batch_size, 1, max_len, max_len)
        
        # self-attention 
        for block in self.blocks:
            past_emb, attn_dist = block(past_emb, mask)

        # LSTM layer
        past_emb, _ = self.past_lstm(past_emb)

        ######### current ######### 
        # current cat embedding
        current_embed_interaction = self.current_embedding_interaction(current_interaction.int())
        current_embed_test = self.current_embedding_test(current_test.int())
        current_embed_question = self.current_embedding_question(current_question.int())
        current_embed_tag = self.current_embedding_tag(current_tag.int())
        current_embed_testTag = self.current_embedding_testTag(current_testTag.int())

        ### 3pl dim ↑ & 비선형성 추가
        current_dffclt_linear=self.current_lin_activation(current_dffclt.unsqueeze(dim=2))
        current_dscrmn_linear=self.current_lin_activation(current_dscrmn.unsqueeze(dim=2))
        current_gussng_linear=self.current_lin_activation(current_gussng.unsqueeze(dim=2))
        ### FE ###
        current_user_correct_answer_eb = self.current_lin_activation(current_user_correct_answer.unsqueeze(dim=2))
        current_user_total_answer_eb = self.current_lin_activation(current_user_total_answer.unsqueeze(dim=2))
        current_user_acc_eb = self.current_lin_activation(current_user_acc.unsqueeze(dim=2))
        current_user_mean_eb = self.current_lin_activation(current_user_mean.unsqueeze(dim=2))
        current_relative_answer_mean_eb = self.current_lin_activation(current_relative_answer_mean.unsqueeze(dim=2))
        current_time_to_solve_eb = self.current_lin_activation(current_time_to_solve.unsqueeze(dim=2))
        current_time_to_solve_mean_eb = self.current_lin_activation(current_time_to_solve_mean.unsqueeze(dim=2))
        current_prior_testTag_frequency_eb = self.current_lin_activation(current_prior_testTag_frequency.unsqueeze(dim=2))

        # concat cat_embs
        current_cat_emb = torch.concat([current_embed_interaction, current_embed_test, 
                                        current_embed_question, current_embed_tag, 
                                        current_embed_testTag], dim=-1)

        # cat_emb -> linear + LN
        current_cat_emb = self.current_cat_emb(current_cat_emb)

        # concat conti_embs
        current_conti_emb = torch.concat([
                                    current_dffclt_linear,
                                    current_dscrmn_linear,
                                    current_gussng_linear,
                                    current_user_correct_answer_eb,
                                    current_user_total_answer_eb,
                                    current_user_acc_eb,
                                    current_user_mean_eb,
                                    current_relative_answer_mean_eb,
                                    current_time_to_solve_eb,
                                    current_time_to_solve_mean_eb,
                                    current_prior_testTag_frequency_eb
                                ], dim=2)

        # conti emb -> linear + LN
        current_conti_emb = self.current_conti_emb(current_conti_emb)

        # concat cat + conti -> final current_emb
        current_emb = torch.concat([current_cat_emb, current_conti_emb], dim = -1)
        # final emb -> LN
        current_emb = self.emb_layernorm(current_emb) # LayerNorm

        # masking 
        mask_pad = torch.BoolTensor(current_correct.to('cpu') > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, current_correct.size(1), current_correct.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to('cuda') # (batch_size, 1, max_len, max_len)
        
        # self-attention 
        for block in self.blocks:
            current_emb, attn_dist = block(current_emb, mask)

        # LSTM layer
        current_emb, _ = self.current_lstm(current_emb)



        emb = torch.concat([past_emb, current_emb], dim = -1)
        output = self.predict_layer(self.dropout(emb))
        return output
"""
Each past/current 
1. cat embedding
2. cat embedding -> linear & LayerNorm

3. continuous linear & activation
4. continuous -> linear & activation


5. concat(cat embedding + continuous)
6. concat_out + answerCode
7. LayerNorm

8. mask 생성
    ### mask ###
    - padding mask는 data preprocessing 과정에서 만듦, 차원만 (batch_size, 1, 1, max_len)로 맞춰주기
    - mask_time은 똑같이(시간별)
    mask_time = (1 - torch.triu(torch.ones((1, 1, past_answerCode.size(1), past_answerCode.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
    mask = (mask_pad & mask_time).to(self.device)
9. self-attention
    for block in self.past_blocks:
        past_emb, attn_dist = block(past_emb, mask)

10. LSTM layer
===
11. concat(past_out+current_out)
12. predict layer
13. return final output
"""