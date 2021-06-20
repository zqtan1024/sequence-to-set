import pdb
import torch
from torch import nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
import copy


class SSNTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, pos, src, mask):
        # self attention
        q = k = self.with_pos_embed(tgt, pos)
        v = tgt
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, pos)
        k = v = src
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class SSNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, embed, src, mask):
        pos, tgt = torch.split(embed, src.size(2), dim=-1)
        tgt = tgt.unsqueeze(0).expand(src.size(0), -1, -1)
        pos = pos.unsqueeze(0).expand(src.size(0), -1, -1)
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, pos, src, mask)

        return output


class SSNFuse(nn.Module):
    """ Fuse bert embeddings with query embeddings and predict boundaries """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, bert, query, mask):
        bert_embed = bert.unsqueeze(1).expand(-1, query.size(1), -1, -1)
        query_embed = query.unsqueeze(2).expand(-1, -1, bert.size(-2), -1)
        fuse = torch.cat([bert_embed, query_embed], dim=-1)
        x = self.W(fuse)

        x = self.v(torch.tanh(x)).squeeze(-1)
        mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x[~mask]=-1e25
        x = x.softmax(dim=-1)

        return x


class SSN(BertPreTrainedModel):
    """ A Sequence-to-Set Network for Named Entity Recognition """

    VERSION = '1.0'

    def __init__(self, config: BertConfig, embed: torch.tensor, cls_token: int, entity_types: int,
                 prop_drop: float, freeze_transformer: bool, num_decoder_layers:int = 3, lstm_layers:int = 3, lstm_drop: float = 0.4, pos_size: int = 25, 
                 char_lstm_layers:int = 1, char_lstm_drop:int = 0.2, char_size:int = 25, use_glove: bool = True, use_pos:bool = True, use_char_lstm:bool = True, 
                 pool_type:str = "max", reduce_dim = False, bert_before_lstm = False, num_query: int=60):
        super(SSN, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_glove = use_glove
        self.use_pos = use_pos
        self.char_lstm_layers=char_lstm_layers
        self.char_lstm_drop=char_lstm_drop
        self.char_size=char_size
        self.use_char_lstm=use_char_lstm
        self.pool_type = pool_type
        self.reduce_dim = reduce_dim
        self.bert_before_lstm = bert_before_lstm
        
        lstm_input_size = 0
        if self.bert_before_lstm:
            lstm_input_size = config.hidden_size
        # use_glove or use_pos or use_char_lstm
        if use_glove:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size = char_size, hidden_size = char_size, num_layers = char_lstm_layers,  bidirectional = True, dropout = char_lstm_drop, batch_first = True)
            self.char_embedding = nn.Embedding(103, char_size)


        if self.use_glove or self.use_pos or self.use_char_lstm or self.bert_before_lstm:
            lstm_hidden_size = lstm_input_size
            if self.bert_before_lstm:
                lstm_hidden_size = config.hidden_size//2
            self.lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = lstm_hidden_size, num_layers = lstm_layers,  bidirectional = True, dropout = lstm_drop, batch_first = True)
            if self.reduce_dim or self.bert_before_lstm:
                if not self.bert_before_lstm:
                    self.reduce_dimension = nn.Linear(2 * lstm_input_size + config.hidden_size, config.hidden_size)
        
        # Decode
        self.query_embed = nn.Embedding(num_query, config.hidden_size * 2)
        
        decoder_layer = SSNTransformerDecoderLayer(d_model=config.hidden_size, d_ffn=1024, dropout=0.1)
        self.decoder = SSNTransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)

        self.entity_classifier = nn.Linear(config.hidden_size, entity_types)
        self.entity_left = SSNFuse(config.hidden_size)
        self.entity_right = SSNFuse(config.hidden_size)

        self.dropout = nn.Dropout(prop_drop)
        self._cls_token = cls_token
        self._entity_types = entity_types

        # weight initialization
        self.init_weights()
        if use_glove:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def combine(self, sub, sup_mask, pool_type="max"):
        """ Combine different level representations """

        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):

        # encode
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]

        batch_size = encodings.shape[0]
        token_count = token_masks_bool.long().sum(-1,keepdim=True)
        h_token = self.combine(h, token_masks, self.pool_type)

        embeds = []
        if self.bert_before_lstm:
            embeds = [h_token]
        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_glove:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count*batch_size, max_char_count)
            char_encoding[char_count==0][:, 0] = 101
            char_count[char_count==0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input = char_embed, lengths = char_count.tolist(), enforce_sorted = False, batch_first = True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = self.combine(char_embed, token_masks_char, self.pool_type)
            embeds.append(h_token_char)

        if len(embeds)>0:
            h_token_pos_wordvec_char = torch.cat(embeds, dim = -1)
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(input = h_token_pos_wordvec_char, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token_pos_wordvec_char_packed_o, (_, _) = self.lstm(h_token_pos_wordvec_char_packed)
            h_token_pos_wordvec_char, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o, batch_first=True)

            rep = [h_token_pos_wordvec_char]
            if not self.bert_before_lstm:
                rep.append(h_token)
            h_token = torch.cat(rep, dim=-1)
            if self.reduce_dim and not self.bert_before_lstm:
                h_token = self.reduce_dimension(h_token)
        
        # decode
        query_embed = self.query_embed.weight
        tgt = self.decoder(query_embed, h_token, token_masks_bool)

        entity_clf = self.entity_classifier(tgt)
        entity_left = self.entity_left(h_token, tgt, token_masks_bool)
        entity_right = self.entity_right(h_token, tgt, token_masks_bool)

        return entity_clf, (entity_left, entity_right)


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        """
        Do something different in training.

        """
        
        return self._common_forward(encodings, context_masks, token_masks, token_masks_bool, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count)


    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        """
        Do something different in evaluation.

        """
        
        return self._common_forward(encodings, context_masks, token_masks, token_masks_bool, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count)


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access
_MODELS = {
    'ssn': SSN,
}


def get_model(name):
    return _MODELS[name]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
