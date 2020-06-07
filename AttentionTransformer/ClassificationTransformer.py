import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import Encoder
from .Decoder import Decoder

class ClassificationTransformer(nn.Module):

    def __init__(self, vocab_size, num_classes, seq_len, pad_id, emb_dim = 512, dim_model = 512, dim_inner = 2048, layers = 6, heads = 8, dim_key = 64, dim_value = 64, dropout = 0.1, num_pos = 200):

        super(ClassificationTransformer, self).__init__()

        self.pad_id = pad_id

        self.encoder = Encoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder = Decoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.target_word_projection = nn.Linear(dim_model, vocab_size, bias=False)

        self.num_classes = num_classes

        self.classification_layer = nn.Linear(vocab_size * seq_len, num_classes, bias=False)

        self.x_logit_scale = 1


    def get_pad_mask(self, sequence, pad_id):

        return (sequence != pad_id).unsqueeze(-2)

    def get_subsequent_mask(self, sequence):

        print(f'Trg Sub Seq: {sequence.size()}')

        batch_size, seq_length = sequence.size()

        subsequent_mask = (
            1 - torch.triu(
                torch.ones((1, seq_length, seq_length), device=sequence.device), diagonal = 1
            )
        ).bool()

        return subsequent_mask

    def forward(self, source_seq, target_seq):

        source_seq = self.get_pad_mask(source_seq, self.pad_id)
        target_seq = self.get_pad_mask(target_seq, self.pad_id) & self.get_subsequent_mask(target_seq)

        encoder_output = self.encoder(source_seq, source_mask)
        decoder_output = self.decoder(target_seq, target_mask, encoder_output, source_mask)

        seq_logit = self.target_word_projection(decoder_output) * self.x_logit_scale

        seq_logit = seq_logit.view(seq_logit.size(0), -1) # [batch_size, seq_len * vocab_size]

        classes = self.classification_layer(seq_logit)

        return classes