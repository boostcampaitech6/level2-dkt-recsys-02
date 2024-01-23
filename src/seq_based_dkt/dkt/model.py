import numpy as np
import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
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
        seq_len = interaction.size(1)

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
    


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(ModelBase):
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



        # # embedding combination projection
        #self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        # # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # # self.embedding_position = nn.Embedding(max_seq_len, self.hidden_dim)

        # # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)

        return (h, c)


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
        

        ####################### ENCODER #####################

        q = self.query(X).permute(1, 0, 2)
        q = self.query(X)[:, -1:, :].permute(1, 0, 2)

        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = X + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = X + out
        out = self.ln2(out)

        # encoded_layers = self.encoder(inputs_embeds=X)
        # out = encoded_layers[0]   # encoder layer에서 최종 결과값

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)

        return out
    


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
            n_layers: int = 2,
            n_tests: int = 1538,
            n_questions: int = 9455,
            n_tags: int = 913,
            n_heads: int = 2,
            drop_out: float = 0.1,
            max_seq_len: float = 20,
            n_conti_features: int = 13, ### 추가한 feature 개수
            **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        # self.dropout = self.args.dropout
        self.dropout = drop_out
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.n_conti_features = n_conti_features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # encoder combination projection
        self.enc_comb_proj = nn.Linear(self.hidden_dim//3 * 3 + (self.n_conti_features-1) * 5, self.hidden_dim)

        # decoder combination projection
        self.dec_comb_proj = nn.Linear(self.hidden_dim//3 * 4 + self.n_conti_features * 5, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)


        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, test, question, tag, correct, mask, interaction, 
                dffclt, dscrmn, gussng, 
                testTag, user_correct_answer, user_total_answer,
                user_acc, user_mean, assessment_mean, test_mean,
                knowledgeTag_mean, time_to_solve, prior_testTag_frequency):
        
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # Embedding
        # embed_interaction = self.embedding_interaction(interaction.int())
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


        embed_enc = torch.cat(
            [   embed_test,
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
                prior_testTag_frequency_linear,
            ],
            dim=2,
        )

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
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


        embed_dec = torch.cat(
            [   embed_test,
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
                prior_testTag_frequency_linear,
                time_to_solve_linear,
                embed_interaction,
            ],
            dim=2,
        )

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)


        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(embed_enc, embed_dec.to(torch.float32),
                               src_mask=self.enc_mask.to(torch.float32),
                               tgt_mask=self.dec_mask.to(torch.float32),
                               memory_mask=self.enc_dec_mask.to(torch.float32))

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)

        return out    