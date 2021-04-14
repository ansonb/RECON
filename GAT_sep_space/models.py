import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        # print('== entity_embeddings shape')
        # print(entity_embeddings.shape)
        # import pdb
        # pdb.set_trace()

        if edge_type_nhop.shape[0]==0:
          edge_embed_nhop = torch.tensor([])
        else:
          # if torch.isnan(relation_embed).any():
            # import pdb; pdb.set_trace()
          edge_embed_nhop = relation_embed[
              edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        # print('== edge_embed_nhop shape')
        # print(edge_embed_nhop.shape)

        # print('== len of edge_list, edge_list_nhop, edge_embed_nhop')
        # print(len(edge_list),len(edge_list_nhop),len(edge_embed_nhop))

        # x = torch.cat([att(x, edge_list_nhop, edge_embed, edge_list_nhop, edge_embed_nhop)
        #                for att in self.attentions], dim=1)
        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        assert not torch.isnan(self.W).any()
        assert not torch.isnan(relation_embed).any()
        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        if edge_type_nhop.shape[0]==0:
          edge_embed_nhop = torch.tensor([])
        else:
          edge_embed_nhop = out_relation_1[
              edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, initial_entity_emb_params):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)
        

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        self.W_ent2rel = nn.Parameter(torch.zeros(
            size=(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_ent2rel.data, gain=1.414)
        self.nonlinearity_ent2rel = torch.tanh

    def forward(self, Corpus_, batch_entities, adj, train_indices_nhop):

        edge_list = adj[0]
        edge_type = adj[1]

        if train_indices_nhop.shape[0]==0:
          edge_list_nhop = torch.tensor([])
          edge_type_nhop = torch.tensor([])
        else:
          edge_list_nhop = torch.cat(
              (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
          edge_type_nhop = torch.cat(
              [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        if CUDA:
          mask_indices = torch.unique(batch_entities).cuda()
          mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        else:
          mask_indices = torch.unique(batch_entities)
          mask = torch.zeros(self.entity_embeddings.shape[0])
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data

        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1, mask


    def batch_test(self, Corpus_, batch_entities, adj, train_indices_nhop, entity_embeddings):

        # entity_embeddings = torch.identity(self.entity_embeddings)
        relation_embeddings = self.relation_embeddings.detach()

        edge_list = adj[0]
        edge_type = adj[1]

        if train_indices_nhop.shape[0]==0:
          edge_list_nhop = torch.tensor([])
          edge_type_nhop = torch.tensor([])
        else:
          edge_list_nhop = torch.cat(
              (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
          edge_type_nhop = torch.cat(
              [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)


        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = relation_embeddings[edge_type]

        start = time.time()

        entity_embeddings = F.normalize(
            entity_embeddings.data, p=2, dim=1).detach()

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, entity_embeddings, relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        # import pdb; pdb.set_trace()
        if CUDA:
          mask_indices = torch.unique(batch_entities).cuda()
          mask = torch.zeros(entity_embeddings.shape[0]).cuda()
        else:
          mask_indices = torch.unique(batch_entities)
          mask = torch.zeros(entity_embeddings.shape[0])
        mask[mask_indices] = 1.0

        entities_upgraded = entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)


        return out_entity_1, out_relation_1, mask


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    # # def forward(self, Corpus_, adj, batch_inputs):
    # #     conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
    # #         batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
    # #     out_conv = self.convKB(conv_input)
    # #     return out_conv

    # # def batch_test(self, batch_inputs):
    # #     conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
    # #         batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
    # #     out_conv = self.convKB(conv_input)
    # #     return out_conv

    # def forward(self, Corpus_, adj, batch_inputs):
    #     conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :], self.final_relation_embeddings[
    #         batch_inputs[:, 1]], self.final_entity_embeddings[batch_inputs[:, 2], :]), dim=1)
    #     out_conv = self.convKB(conv_input)
    #     return out_conv

    # def batch_test(self, batch_inputs):
    #     conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :], self.final_relation_embeddings[
    #         batch_inputs[:, 1]], self.final_entity_embeddings[batch_inputs[:, 2], :]), dim=1)
    #     out_conv = self.convKB(conv_input)
    #     return out_conv

    def forward(self, Corpus_, adj, batch_inputs, model_gat):
        source_embeds = self.final_entity_embeddings[batch_inputs[:, 0], :]
        relation_embeds = self.final_relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = self.final_entity_embeddings[batch_inputs[:, 2], :]

        W_ent2rel = model_gat.W_ent2rel[batch_inputs[:, 1]]
        source_embeds = source_embeds.unsqueeze(1)
        tail_embeds = tail_embeds.unsqueeze(1)
        source_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(source_embeds,W_ent2rel))
        tail_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(tail_embeds,W_ent2rel))

        conv_input = torch.cat((source_embeds.squeeze(), relation_embeds, tail_embeds.squeeze()), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs, model_gat):
        source_embeds = self.final_entity_embeddings[batch_inputs[:, 0], :]
        relation_embeds = self.final_relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = self.final_entity_embeddings[batch_inputs[:, 2], :]

        W_ent2rel = model_gat.W_ent2rel[batch_inputs[:, 1]]
        source_embeds = source_embeds.unsqueeze(1)
        tail_embeds = tail_embeds.unsqueeze(1)
        source_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(source_embeds,W_ent2rel))
        tail_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(tail_embeds,W_ent2rel))

        conv_input = torch.cat((source_embeds.squeeze(), relation_embeds, tail_embeds.squeeze()), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv    

class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds

class EntityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate, hidden_dim_entity, entity_embed_dim, entity_conv_filter_size, word_vocab, word_embed_dim, char_embed_dim, word_embed_matrix, char_feature_size, conv_filter_size, max_word_len_entity, char_vocab):
        super(EntityEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate

        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, self.drop_rate)
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, self.drop_rate)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
          bidirectional=self.is_bidirectional)

        # self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size,padding=0)
        self.max_pool = nn.MaxPool1d(max_word_len_entity + conv_filter_size - 1, max_word_len_entity + conv_filter_size - 1)

        self.conv1d_entity = nn.Conv1d(2*hidden_dim_entity, entity_embed_dim, entity_conv_filter_size)
        self.max_pool_entity = nn.MaxPool1d(128, 128) # max batch len for context is 128


    def forward(self, words, chars, conv_mask):
        batch_size = words.shape[0]
        max_batch_len = words.shape[1]
        # import pdb; pdb.set_trace()

        words = words.view(words.shape[0]*words.shape[1],words.shape[2])
        chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        outputs, hc = self.lstm(words_input)

        # h_drop = self.dropout(hc[0])
        h_n = hc[0].view(self.layers, 2, words.shape[0], self.hidden_dim)
        h_n = h_n[-1,:,:,:].squeeze() # (num_dir,batch,hidden)
        h_n = h_n.permute((1,0,2)) # (batch,num_dir,hidden)
        h_n = h_n.contiguous().view(h_n.shape[0],h_n.shape[1]*h_n.shape[2]) # (batch,num_dir*hidden)
        h_n_batch = h_n.view(batch_size,max_batch_len,h_n.shape[1])

        h_n_batch = h_n_batch.permute(0, 2, 1)
        conv_entity = self.conv1d_entity(h_n_batch)
        conv_mask = conv_mask.unsqueeze(dim=1)
        conv_mask = conv_mask.repeat(1,conv_entity.shape[1],1)
        conv_entity.data.masked_fill_(conv_mask.data, -float('inf'))

        max_pool_entity = nn.MaxPool1d(conv_entity.shape[-1], conv_entity.shape[-1])
        entity_embed = max_pool_entity(conv_entity)
        entity_embed = entity_embed.permute(0, 2, 1).squeeze(dim=1)
        
        return entity_embed
