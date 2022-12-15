import torch
from torch import nn
from nets.graph_layers import  MultiHeadAttentionLayerforCritic, ValueDecoder, Encoder_LSTM, Encoder_GRU,Decoder_GRU,Decoder_LSTM


class Critic(nn.Module):

    def __init__(self,
             exp_encoder,
             embedding_dim,
             hidden_dim,
             n_heads,
             n_layers,
             normalization,
             ):
        
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.exp_encoder = exp_encoder
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.drop_out = 0.05
        
        self.encoder_base = nn.Sequential(*(
                MultiHeadAttentionLayerforCritic(self.n_heads, 
                                    self.embedding_dim * 2, 
                                    self.hidden_dim * 2, 
                                    self.normalization)
                        for _ in range(1)))

        self.encoder_lstm = Encoder_LSTM(self.embedding_dim*2, self.embedding_dim*2, self.hidden_dim*2, self.n_layers, self.drop_out)
        #self.encoder_gru = Encoder_GRU(self.embedding_dim*2, self.embedding_dim*2, self.hidden_dim*2, self.n_layers, self.drop_out)
            
        self.value_head = ValueDecoder(input_dim = self.embedding_dim * 2,
                                       embed_dim = self.embedding_dim * 2)

    def forward(self, input):
        # get concatenated input
        h_features = torch.cat(input, -1).detach()
        #print(h_features.shape)
        # if self.exp_name == 0:
        # # pass through encoder
        if self.exp_encoder == 'attention':
            h_em = self.encoder_base(h_features)
        else:
            h_em,_,_ = self.encoder_lstm(h_features)
        #     h_em = self.encoder_base(h_features)
        # if (self.exp_name == 1) or (self.exp_name == 2):
        #     h_em,_,_ = self.encoder_lstm(h_features)
        # if (self.exp_name == 3) or (self.exp_name == 4):
        #     h_em,_,_ = self.encoder_gru(h_features)
        # pass through value_head, get estimated value
        baseline_value = self.value_head(h_em)
        return baseline_value.detach().squeeze(), baseline_value.squeeze()
        
