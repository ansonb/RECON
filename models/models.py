import torch
from torch import nn
from torch.autograd import Variable
import sys
sys.path.insert(0, '..')
from RECON.parsing.legacy_sp_models import MAX_EDGES_PER_GRAPH
from utils.build_adjecent_matrix import adjecent_matrix
from models.layers import GraphConvolution
from utils.embedding_utils import make_start_embedding, get_head_indices, get_tail_indices
from utils import context_utils
import torch.nn.functional as F

CUDA = torch.cuda.is_available()

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
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate, entity_embed_dim, conv_filter_size, entity_conv_filter_size, word_embeddings, char_embed_dim, max_word_len_entity, char_vocab, char_feature_size):
        super(EntityEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate

        # self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, self.drop_rate)
        self.word_embeddings = word_embeddings
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, self.drop_rate)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
          bidirectional=bool(self.is_bidirectional))

        # self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len_entity + conv_filter_size - 1, max_word_len_entity + conv_filter_size - 1)

        self.conv1d_entity = nn.Conv1d(2*hidden_dim, entity_embed_dim, entity_conv_filter_size)
        self.max_pool_entity = nn.MaxPool1d(32, 32) # max batch len for context is 128


    def forward(self, words, chars, conv_mask):
        batch_size = words.shape[0]
        max_batch_len = words.shape[1]

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

class GPGNN(nn.Module):

    def __init__(self, p, embeddings, max_sent_len, n_out, MAX_EDGES_PER_GRAPH=MAX_EDGES_PER_GRAPH):
        super(GPGNN, self).__init__()

        print("Parameters:", p)
        self.p = p
        if self.p.get('max_num_nodes'):
          self.MAX_EDGES_PER_GRAPH = self.p['max_num_nodes']*(self.p['max_num_nodes']-1)
        else:
          self.MAX_EDGES_PER_GRAPH = MAX_EDGES_PER_GRAPH

        # Input shape: (max_sent_len,)
        # Input type: int
        self.max_sent_len = max_sent_len

        self.word_embedding = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(p=p['dropout1'])

        self.pos_embedding = nn.Embedding(4, p['position_emb'], padding_idx=0)
        nn.init.orthogonal_(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(batch_first=True, input_size=embeddings.shape[1] + int(p['position_emb']),
                                           hidden_size=int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), bidirectional=bool(p['bidirectional']))

        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal_(parameter)

        self.dropout2 = nn.Dropout(p=p['dropout1'])

        if(p['layer_number'] == 1 or p['projection_style'] == 'tie'):
            self.representation_to_adj = nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2)
            nn.init.xavier_uniform_(self.representation_to_adj.weight)
        else:
            self.representation_to_adj = nn.ModuleList([nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2) for i in range(p['layer_number'])])
            for i in self.representation_to_adj:
                nn.init.xavier_uniform_(i.weight)
        
        self.identity_transformation = nn.Parameter(
            torch.eye(p['embedding_dim'] * 2), requires_grad=True)
        
        self.start_embedding = nn.Parameter(torch.from_numpy(
            make_start_embedding(p['max_num_nodes'], p['embedding_dim'])).float(), requires_grad=False)
        
        self.head_indices = nn.Parameter(torch.LongTensor(
            get_head_indices(p['max_num_nodes'], p['embedding_dim'])), requires_grad=False)
        
        self.tail_indices = nn.Parameter(torch.LongTensor(
            get_tail_indices(p['max_num_nodes'], p['embedding_dim'])), requires_grad=False)
        
        self.linear3 = nn.Linear(
            in_features=p['embedding_dim'] * 2 * p['layer_number'], out_features=n_out)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, sentence_input, entity_markers, num_entities):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len) ? edge markers?
        num_entities: (batch_size,) a list of number of entities of each instance in the batch
        
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the sentences for MAX_EDGES_PER_GRAPH times. As we will need it to be encoded differently
        # with difference target entity pairs.
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, max_sent_len) => (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        expanded_sentence_input = torch.transpose(
            sentence_input.expand(self.MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        
        # Word and position embeddings lookup.
        # shape: batch, MAX_EDGES_PER_GRAPH, max_sent_len, wordemb_dim
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        # Merge them together!
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)
        merged_embeddings = merged_embeddings.view(-1, self.max_sent_len, merged_embeddings.size()[-1])

        # Encode the setntences with LSTM. 
        # NOTE that the api of LSTM, GRU and RNN are different, the following only works for LSTM
        rnn_output, _ = self.rnn1(merged_embeddings)
        # rnn_output shape: batch * self.MAX_EDGES_PER_GRAPH, max_sent_len, hidden
        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1).view(sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, -1)
        rnn_result = self.dropout2(rnn_result)


        if(self.p['layer_number'] == 1 or self.p['projection_style'] == 'tie'):
            # 1 layer case or tied-matrices cases
            rnn_result = self.representation_to_adj(rnn_result).view(sentence_input.size()[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # magic number here
            if(self.p['non-linear'] != "linear"):
                try:
                    rnn_result = getattr(F, self.p['non-linear'])(rnn_result)
                except:
                    raise NotImplementedError
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * (self.p['max_num_nodes']-1), 1).view(sentence_input.size()[0], self.p['max_num_nodes']-1, 1, (self.p['embedding_dim'] * 2) ** 2)
            rnn_result = torch.cat([identity_stuffing, rnn_result], dim=2).view(sentence_input.size()[0], self.p['max_num_nodes']**2-1, (self.p['embedding_dim'] * 2) ** 2)
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
            adjecent_matrix = torch.cat([rnn_result, identity_stuffing], dim=1).view(sentence_input.size()[0], self.p['max_num_nodes'], self.p['max_num_nodes'], self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                     * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])

            # adjecent_matrix = torch.matmul(adjecent_matrix, block_matrix).view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            adjecent_matrix = adjecent_matrix.view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])
            if(self.p['layer_number'] == 1):
                layer_1 = torch.matmul(adjecent_matrix, self.start_embedding).view(
                    sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_1 = getattr(F, self.p['non-linear1'])(layer_1)
                    except:
                        raise NotImplementedError

                heads = torch.gather(layer_1, 2, self.head_indices)
                tails = torch.gather(layer_1, 2, self.tail_indices)
                relation = heads * tails
                main_output = self.linear3(relation).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
            else:
                layer_tmp = self.start_embedding
                relation_list = []
                for i in range(self.p['layer_number']):
                    layer_tmp = torch.matmul(adjecent_matrix, layer_tmp)
                    if(self.p['non-linear1'] != 'linear'):
                        try:
                            layer_tmp = getattr(
                                F, self.p['non-linear1'])(layer_tmp)
                        except:
                            raise NotImplementedError
                    layer_result = layer_tmp.view(
                        sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * 18)
                    
                    heads = torch.gather(layer_result, 2, self.head_indices)
                    tails = torch.gather(layer_result, 2, self.tail_indices)
                    relation = heads * tails
                    relation_list.append(relation)
                main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
        else:
            adjecent_matrix = []
            relation_list = []
            for i in range(self.p['layer_number']):
                rnn_result_tmp = self.representation_to_adj[i](rnn_result).view(sentence_input.size(
                )[0], self.p['max_num_nodes']-1, self.p['max_num_nodes'], (self.p['embedding_dim'] * 2) ** 2)  # 9 is the num of node, 8 * 9 is the edge_num
                
                if(self.p['non-linear1'] != "linear"):
                    try:
                        rnn_result_tmp = getattr(
                            F, self.p['non-linear1'])(rnn_result_tmp)
                    except:
                        raise NotImplementedError
                        
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * (self.p['max_num_nodes']-1), 1).view(sentence_input.size()[0], (self.p['max_num_nodes']-1), 1, (self.p['embedding_dim'] * 2) ** 2)
                
                rnn_result_tmp = torch.cat([identity_stuffing, rnn_result_tmp], dim=2).view(sentence_input.size()[0], self.p['max_num_nodes']**2-1, (self.p['embedding_dim'] * 2) ** 2)
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
                adjecent_matrix.append(None)
                adjecent_matrix[i] = torch.cat([rnn_result_tmp, identity_stuffing], dim=1).view(sentence_input.size()[0], self.p['max_num_nodes'], self.p['max_num_nodes'], self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                                * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])

                adjecent_matrix[i] = adjecent_matrix[i].view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])
            layer_tmp = self.start_embedding
            for i in range(self.p['layer_number']):
                layer_tmp = torch.matmul(adjecent_matrix[i], layer_tmp)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_tmp = getattr(
                            F, self.p['non-linear1'])(layer_tmp)
                    except:
                        raise NotImplementedError
                layer_result = layer_tmp.view(
                    sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])

                heads = torch.gather(layer_result, 2, self.head_indices)
                tails = torch.gather(layer_result, 2, self.tail_indices)
                relation_list.append(heads * tails)
            main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
            return main_output

class RECON_EAC(nn.Module):
    """
    Our model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out, char_vocab, MAX_EDGES_PER_GRAPH=MAX_EDGES_PER_GRAPH):
        super(RECON_EAC, self).__init__()

        print("Parameters:", p)
        self.p = p
        if self.p.get('max_num_nodes'):
          self.MAX_EDGES_PER_GRAPH = self.p['max_num_nodes']*(self.p['max_num_nodes']-1)
        else:
          self.MAX_EDGES_PER_GRAPH = MAX_EDGES_PER_GRAPH

        # Input shape: (max_sent_len,)
        # Input type: int
        self.max_sent_len = max_sent_len

        self.word_embedding = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(p=p['dropout1'])

        self.pos_embedding = nn.Embedding(4, p['position_emb'], padding_idx=0)
        nn.init.orthogonal_(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(batch_first=True, input_size=embeddings.shape[1] + int(p['position_emb']),
                                           hidden_size=int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), bidirectional=bool(p['bidirectional']))

        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal_(parameter)

        self.dropout2 = nn.Dropout(p=p['dropout1'])

        if(p['layer_number'] == 1 or p['projection_style'] == 'tie'):
            self.representation_to_adj = nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2)
            nn.init.xavier_uniform_(self.representation_to_adj.weight)
        else:
            self.representation_to_adj = nn.ModuleList([nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2) for i in range(p['layer_number'])])
            for i in self.representation_to_adj:
                nn.init.xavier_uniform_(i.weight)
        
        self.identity_transformation = nn.Parameter(
            torch.eye(p['embedding_dim'] * 2), requires_grad=True)
        
        self.start_embedding = nn.Parameter(torch.from_numpy(
            make_start_embedding(p['max_num_nodes'], p['embedding_dim'])).float(), requires_grad=False)
        
        self.head_indices = nn.Parameter(torch.LongTensor(
            get_head_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])), requires_grad=False)
        
        self.tail_indices = nn.Parameter(torch.LongTensor(
            get_tail_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])), requires_grad=False)
        
        self.linear3 = nn.Linear(
            in_features=p['embedding_dim'] * 2 * p['layer_number'], out_features=n_out)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.entity_embedding_module = EntityEmbedding(p['char_embed_dim']+embeddings.shape[1], p['hidden_dim_ent'], p['num_entEmb_layers'], p['is_bidirectional_ent'], p['drop_out_rate_ent'], 
          p['entity_embed_dim'], p['conv_filter_size'], p['entity_conv_filter_size'], self.word_embedding, p['char_embed_dim'], p['max_char_len'], char_vocab, p['char_feature_size'])

    def forward(self, sentence_input, entity_markers, num_entities, unique_entites, entity_indices, context_words, context_chars, context_mask, entities_position, max_occurred_entity_in_batch_pos):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len) ? edge markers?
        num_entities: (batch_size,) a list of number of entities of each instance in the batch
        
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the sentences for MAX_EDGES_PER_GRAPH times. As we will need it to be encoded differently
        # with difference target entity pairs.
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, max_sent_len) => (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        
        # import time
        # s1=time.time()
        entity_embeddings = self.entity_embedding_module(context_words, context_chars, context_mask)
        # s2=time.time()
        # print('t1',s2-s1)
        start_embedding = context_utils.make_start_entity_embeddings(entity_embeddings, entities_position, unique_entites, self.p['embedding_dim'], max_occurred_entity_in_batch_pos, self.start_embedding, max_num_nodes=self.p['max_num_nodes'])
        # s3=time.time()
        # print('t2',s3-s2)

        expanded_sentence_input = torch.transpose(
            sentence_input.expand(self.MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        
        # Word and position embeddings lookup.
        # shape: batch, self.MAX_EDGES_PER_GRAPH, max_sent_len, wordemb_dim
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        # Merge them together!
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)
        merged_embeddings = merged_embeddings.view(-1, self.max_sent_len, merged_embeddings.size()[-1])

        # Encode the setntences with LSTM. 
        # NOTE that the api of LSTM, GRU and RNN are different, the following only works for LSTM
        rnn_output, _ = self.rnn1(merged_embeddings)
        # rnn_output shape: batch * self.MAX_EDGES_PER_GRAPH, max_sent_len, hidden
        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1).view(sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, -1)
        rnn_result = self.dropout2(rnn_result)


        if(self.p['layer_number'] == 1 or self.p['projection_style'] == 'tie'):
            # 1 layer case or tied-matrices cases
            rnn_result = self.representation_to_adj(rnn_result).view(sentence_input.size()[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # magic number here
            if(self.p['non-linear'] != "linear"):
                try:
                    rnn_result = getattr(F, self.p['non-linear'])(rnn_result)
                except:
                    raise NotImplementedError
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * 8, 1).view(sentence_input.size()[0], 8, 1, (self.p['embedding_dim'] * 2) ** 2)
            rnn_result = torch.cat([identity_stuffing, rnn_result], dim=2).view(sentence_input.size()[0], 80, (self.p['embedding_dim'] * 2) ** 2)
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
            adjecent_matrix = torch.cat([rnn_result, identity_stuffing], dim=1).view(sentence_input.size()[0], 9, 9, self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                     * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)

            # adjecent_matrix = torch.matmul(adjecent_matrix, block_matrix).view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            adjecent_matrix = adjecent_matrix.view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            if(self.p['layer_number'] == 1):
                layer_1 = torch.matmul(adjecent_matrix, self.start_embedding).view(
                    sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_1 = getattr(F, self.p['non-linear1'])(layer_1)
                    except:
                        raise NotImplementedError

                heads = torch.gather(layer_1, 2, self.head_indices)
                tails = torch.gather(layer_1, 2, self.tail_indices)
                relation = heads * tails
                main_output = self.linear3(relation).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
            else:
                layer_tmp = self.start_embedding
                relation_list = []
                for i in range(self.p['layer_number']):
                    layer_tmp = torch.matmul(adjecent_matrix, layer_tmp)
                    if(self.p['non-linear1'] != 'linear'):
                        try:
                            layer_tmp = getattr(
                                F, self.p['non-linear1'])(layer_tmp)
                        except:
                            raise NotImplementedError
                    layer_result = layer_tmp.view(
                        sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                    
                    heads = torch.gather(layer_result, 2, self.head_indices)
                    tails = torch.gather(layer_result, 2, self.tail_indices)
                    relation = heads * tails
                    relation_list.append(relation)
                main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
        else:
            adjecent_matrix = []
            relation_list = []
            for i in range(self.p['layer_number']):
                rnn_result_tmp = self.representation_to_adj[i](rnn_result).view(sentence_input.size(
                )[0], self.p['max_num_nodes']-1, self.p['max_num_nodes'], (self.p['embedding_dim'] * 2) ** 2)  # 9 is the num of node, 8 * 9 is the edge_num
                
                if(self.p['non-linear1'] != "linear"):
                    try:
                        rnn_result_tmp = getattr(
                            F, self.p['non-linear1'])(rnn_result_tmp)
                    except:
                        raise NotImplementedError
                        
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * (self.p['max_num_nodes']-1), 1).view(sentence_input.size()[0], self.p['max_num_nodes']-1, 1, (self.p['embedding_dim'] * 2) ** 2)
                
                rnn_result_tmp = torch.cat([identity_stuffing, rnn_result_tmp], dim=2).view(sentence_input.size()[0], self.p['max_num_nodes']**2-1, (self.p['embedding_dim'] * 2) ** 2)
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
                adjecent_matrix.append(None)
                adjecent_matrix[i] = torch.cat([rnn_result_tmp, identity_stuffing], dim=1).view(sentence_input.size()[0], self.p['max_num_nodes'], self.p['max_num_nodes'], self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                                * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])

                adjecent_matrix[i] = adjecent_matrix[i].view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 2 * self.p['max_num_nodes'], self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])
            layer_tmp = start_embedding
            for i in range(self.p['layer_number']):
                layer_tmp = torch.matmul(adjecent_matrix[i], layer_tmp)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_tmp = getattr(
                            F, self.p['non-linear1'])(layer_tmp)
                    except:
                        raise NotImplementedError
                layer_result = layer_tmp.view(
                    sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * 2 * self.p['max_num_nodes'])

                heads = torch.gather(layer_result, 2, self.head_indices)
                tails = torch.gather(layer_result, 2, self.tail_indices)
                relation_list.append(heads * tails)
            main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
            return main_output

class RECON_EAC_KGGAT(nn.Module):
    """
    Our model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out, char_vocab, MAX_EDGES_PER_GRAPH=MAX_EDGES_PER_GRAPH):
        super(RECON_EAC_KGGAT, self).__init__()

        print("Parameters:", p)
        self.p = p
        if self.p.get('max_num_nodes'):
          self.MAX_EDGES_PER_GRAPH = self.p['max_num_nodes']*(self.p['max_num_nodes']-1)
        else:
          self.MAX_EDGES_PER_GRAPH = MAX_EDGES_PER_GRAPH

        # Input shape: (max_sent_len,)
        # Input type: int
        self.max_sent_len = max_sent_len

        self.word_embedding = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(p=p['dropout1'])

        self.pos_embedding = nn.Embedding(4, p['position_emb'], padding_idx=0)
        nn.init.orthogonal_(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(batch_first=True, input_size=embeddings.shape[1] + int(p['position_emb']),
                                           hidden_size=int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), bidirectional=bool(p['bidirectional']))

        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal_(parameter)

        self.dropout2 = nn.Dropout(p=p['dropout1'])

        if(p['layer_number'] == 1 or p['projection_style'] == 'tie'):
            self.representation_to_adj = nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2)
            nn.init.xavier_uniform_(self.representation_to_adj.weight)
        else:
            self.representation_to_adj = nn.ModuleList([nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2) for i in range(p['layer_number'])])
            for i in self.representation_to_adj:
                nn.init.xavier_uniform_(i.weight)
        
        self.identity_transformation = nn.Parameter(
            torch.eye(p['embedding_dim'] * 2), requires_grad=True)
        
        self.start_embedding = nn.Parameter(torch.from_numpy(
            make_start_embedding(p['max_num_nodes'], p['embedding_dim'])).float(), requires_grad=False)
        
        self.head_indices = nn.Parameter(torch.LongTensor(
            get_head_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])), requires_grad=False)
        
        self.tail_indices = nn.Parameter(torch.LongTensor(
            get_tail_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])), requires_grad=False)
        
        self.linear3 = nn.Linear(
            in_features=p['embedding_dim'] * 2 * p['layer_number']+2*p['gat_entity_embedding_dim'], out_features=n_out)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.entity_embedding_module = EntityEmbedding(p['char_embed_dim']+embeddings.shape[1], p['hidden_dim_ent'], p['num_entEmb_layers'], p['is_bidirectional_ent'], p['drop_out_rate_ent'], 
          p['entity_embed_dim'], p['conv_filter_size'], p['entity_conv_filter_size'], self.word_embedding, p['char_embed_dim'], p['max_char_len'], char_vocab, p['char_feature_size'])

    def forward(self, sentence_input, entity_markers, num_entities, unique_entites, entity_indices, context_words, context_chars, context_mask, entities_position, max_occurred_entity_in_batch_pos, gat_entity_embeddings):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len) ? edge markers?
        num_entities: (batch_size,) a list of number of entities of each instance in the batch
        
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the sentences for MAX_EDGES_PER_GRAPH times. As we will need it to be encoded differently
        # with difference target entity pairs.
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, max_sent_len) => (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        
        # import time
        # s1=time.time()
        entity_embeddings = self.entity_embedding_module(context_words, context_chars, context_mask)
        # s2=time.time()
        # print('t1',s2-s1)
        start_embedding = context_utils.make_start_entity_embeddings(entity_embeddings, entities_position, unique_entites, self.p['embedding_dim'], max_occurred_entity_in_batch_pos, self.start_embedding, max_num_nodes=self.p['max_num_nodes'])
        # s3=time.time()
        # print('t2',s3-s2)

        expanded_sentence_input = torch.transpose(
            sentence_input.expand(self.MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        
        # Word and position embeddings lookup.
        # shape: batch, self.MAX_EDGES_PER_GRAPH, max_sent_len, wordemb_dim
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        # Merge them together!
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)
        merged_embeddings = merged_embeddings.view(-1, self.max_sent_len, merged_embeddings.size()[-1])

        # Encode the setntences with LSTM. 
        # NOTE that the api of LSTM, GRU and RNN are different, the following only works for LSTM
        rnn_output, _ = self.rnn1(merged_embeddings)
        # rnn_output shape: batch * self.MAX_EDGES_PER_GRAPH, max_sent_len, hidden
        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1).view(sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, -1)
        rnn_result = self.dropout2(rnn_result)


        if(self.p['layer_number'] == 1 or self.p['projection_style'] == 'tie'):
            # 1 layer case or tied-matrices cases
            rnn_result = self.representation_to_adj(rnn_result).view(sentence_input.size()[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # magic number here
            if(self.p['non-linear'] != "linear"):
                try:
                    rnn_result = getattr(F, self.p['non-linear'])(rnn_result)
                except:
                    raise NotImplementedError
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * 8, 1).view(sentence_input.size()[0], 8, 1, (self.p['embedding_dim'] * 2) ** 2)
            rnn_result = torch.cat([identity_stuffing, rnn_result], dim=2).view(sentence_input.size()[0], 80, (self.p['embedding_dim'] * 2) ** 2)
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
            adjecent_matrix = torch.cat([rnn_result, identity_stuffing], dim=1).view(sentence_input.size()[0], 9, 9, self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                     * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)

            # adjecent_matrix = torch.matmul(adjecent_matrix, block_matrix).view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            adjecent_matrix = adjecent_matrix.view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            if(self.p['layer_number'] == 1):
                layer_1 = torch.matmul(adjecent_matrix, self.start_embedding).view(
                    sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_1 = getattr(F, self.p['non-linear1'])(layer_1)
                    except:
                        raise NotImplementedError

                heads = torch.gather(layer_1, 2, self.head_indices)
                tails = torch.gather(layer_1, 2, self.tail_indices)
                relation = heads * tails
                main_output = self.linear3(relation).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
            else:
                layer_tmp = self.start_embedding
                relation_list = []
                for i in range(self.p['layer_number']):
                    layer_tmp = torch.matmul(adjecent_matrix, layer_tmp)
                    if(self.p['non-linear1'] != 'linear'):
                        try:
                            layer_tmp = getattr(
                                F, self.p['non-linear1'])(layer_tmp)
                        except:
                            raise NotImplementedError
                    layer_result = layer_tmp.view(
                        sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                    
                    heads = torch.gather(layer_result, 2, self.head_indices)
                    tails = torch.gather(layer_result, 2, self.tail_indices)
                    relation = heads * tails
                    relation_list.append(relation)
                main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
        else:
            adjecent_matrix = []
            relation_list = []
            for i in range(self.p['layer_number']):
                rnn_result_tmp = self.representation_to_adj[i](rnn_result).view(sentence_input.size(
                )[0], self.p['max_num_nodes']-1, self.p['max_num_nodes'], (self.p['embedding_dim'] * 2) ** 2)  # 9 is the num of node, 8 * 9 is the edge_num
                
                if(self.p['non-linear1'] != "linear"):
                    try:
                        rnn_result_tmp = getattr(
                            F, self.p['non-linear1'])(rnn_result_tmp)
                    except:
                        raise NotImplementedError
                        
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * (self.p['max_num_nodes']-1), 1).view(sentence_input.size()[0], (self.p['max_num_nodes']-1), 1, (self.p['embedding_dim'] * 2) ** 2)
                
                rnn_result_tmp = torch.cat([identity_stuffing, rnn_result_tmp], dim=2).view(sentence_input.size()[0], self.p['max_num_nodes']**2-1, (self.p['embedding_dim'] * 2) ** 2)
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
                adjecent_matrix.append(None)
                adjecent_matrix[i] = torch.cat([rnn_result_tmp, identity_stuffing], dim=1).view(sentence_input.size()[0], self.p['max_num_nodes'], self.p['max_num_nodes'], self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                                * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * self.p['max_num_nodes'] * 2, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)

                adjecent_matrix[i] = adjecent_matrix[i].view(sentence_input.size()[0], 1, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)
            layer_tmp = start_embedding
            for i in range(self.p['layer_number']):
                layer_tmp = torch.matmul(adjecent_matrix[i], layer_tmp)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_tmp = getattr(
                            F, self.p['non-linear1'])(layer_tmp)
                    except:
                        raise NotImplementedError
                layer_result = layer_tmp.view(
                    sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)

                heads = torch.gather(layer_result, 2, self.head_indices)
                tails = torch.gather(layer_result, 2, self.tail_indices)
                relation_list.append(heads * tails)

            propagation_output = torch.cat(relation_list, dim=-1)
            propagation_and_gat = torch.cat([propagation_output,gat_entity_embeddings], dim=-1)
            main_output = self.linear3(propagation_and_gat).view(
                sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
            main_output = main_output
            return main_output

class RECON(nn.Module):
    """
    Our model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out, char_vocab, gat_relation_embeddings, W_ent2rel_all_rels, idx2property, gat_relation2idx, MAX_EDGES_PER_GRAPH=MAX_EDGES_PER_GRAPH):
        super(RECON, self).__init__()

        print("Parameters:", p)
        self.p = p
        if self.p.get('max_num_nodes'):
          self.MAX_EDGES_PER_GRAPH = self.p['max_num_nodes']*(self.p['max_num_nodes']-1)
        else:
          self.MAX_EDGES_PER_GRAPH = MAX_EDGES_PER_GRAPH

        # Input shape: (max_sent_len,)
        # Input type: int
        self.max_sent_len = max_sent_len

        self.word_embedding = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(p=p['dropout1'])

        self.pos_embedding = nn.Embedding(4, p['position_emb'], padding_idx=0)
        nn.init.orthogonal_(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(batch_first=True, input_size=embeddings.shape[1] + int(p['position_emb']),
                                           hidden_size=int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), bidirectional=bool(p['bidirectional']))

        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal_(parameter)

        self.dropout2 = nn.Dropout(p=p['dropout1'])

        if(p['layer_number'] == 1 or p['projection_style'] == 'tie'):
            self.representation_to_adj = nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2)
            nn.init.xavier_uniform_(self.representation_to_adj.weight)
        else:
            self.representation_to_adj = nn.ModuleList([nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2) for i in range(p['layer_number'])])
            for i in self.representation_to_adj:
                nn.init.xavier_uniform_(i.weight)
        
        self.identity_transformation = nn.Parameter(
            torch.eye(p['embedding_dim'] * 2), requires_grad=True)
        
        self.start_embedding = nn.Parameter(torch.from_numpy(
            make_start_embedding(p['max_num_nodes'], p['embedding_dim'])).float(), requires_grad=False)
        
        if torch.cuda.is_available():
          self.head_indices = torch.LongTensor(
              get_head_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])).cuda()
          self.tail_indices = torch.LongTensor(
              get_tail_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size'])).cuda()
        else:
          self.head_indices = torch.LongTensor(
              get_head_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size']))
          self.tail_indices = torch.LongTensor(
              get_tail_indices(p['max_num_nodes'], p['embedding_dim'], bs=p['batch_size']))
        
        # self.linear3 = nn.Linear(
        #     in_features=p['embedding_dim'] * 2 * p['layer_number']+2*p['gat_entity_embedding_dim'] + n_out*p['gat_entity_embedding_dim'], out_features=n_out)
        self.linear3 = nn.Linear(
            in_features=p['embedding_dim'] * 2 * p['layer_number']+2*p['gat_entity_embedding_dim'] + n_out, out_features=n_out)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.entity_embedding_module = EntityEmbedding(p['char_embed_dim']+embeddings.shape[1], p['hidden_dim_ent'], p['num_entEmb_layers'], p['is_bidirectional_ent'], p['drop_out_rate_ent'], 
          p['entity_embed_dim'], p['conv_filter_size'], p['entity_conv_filter_size'], self.word_embedding, p['char_embed_dim'], p['max_char_len'], char_vocab, p['char_feature_size'])

        init_gat_relation_embeddings = nn.Parameter(torch.zeros([n_out,len(gat_relation_embeddings["0"])]), requires_grad=True)
        for i in range(n_out):
          rel = idx2property[i]
          gat_idx = gat_relation2idx.get(rel,None)
          if gat_idx is not None:
            init_gat_relation_embeddings[i,:] = torch.tensor(gat_relation_embeddings[gat_idx])
        self.gat_relation_embeddings = nn.Parameter(init_gat_relation_embeddings, requires_grad=True)

        init_W_ent2rel = torch.zeros([n_out,W_ent2rel_all_rels.shape[1],W_ent2rel_all_rels.shape[2]])
        for i in range(n_out):
          rel = idx2property[i]
          gat_idx = gat_relation2idx.get(rel,None)
          if gat_idx is not None:
            init_W_ent2rel[i,:,:] = torch.tensor(W_ent2rel_all_rels[int(gat_idx)])
        self.W_ent2rel = nn.Parameter(init_W_ent2rel, requires_grad=False)


    def forward(self, sentence_input, entity_markers, num_entities, unique_entites, entity_indices, context_words, context_chars, context_mask, entities_position, max_occurred_entity_in_batch_pos, nonzero_gat_entity_embeddings, nonzero_entity_pos, gat_entity_embeddings):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len) ? edge markers?
        num_entities: (batch_size,) a list of number of entities of each instance in the batch
        
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the sentences for MAX_EDGES_PER_GRAPH times. As we will need it to be encoded differently
        # with difference target entity pairs.
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, max_sent_len) => (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        
        # import time
        # s1=time.time()
        entity_embeddings = self.entity_embedding_module(context_words, context_chars, context_mask)
        # s2=time.time()
        # print('t1',s2-s1)
        start_embedding = context_utils.make_start_entity_embeddings(entity_embeddings, entities_position, unique_entites, self.p['embedding_dim'], max_occurred_entity_in_batch_pos, self.start_embedding, max_num_nodes=self.p['max_num_nodes'])
        # s3=time.time()
        # print('t2',s3-s2)

        expanded_sentence_input = torch.transpose(
            sentence_input.expand(self.MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        
        # Word and position embeddings lookup.
        # shape: batch, self.MAX_EDGES_PER_GRAPH, max_sent_len, wordemb_dim
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        # Merge them together!
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)
        merged_embeddings = merged_embeddings.view(-1, self.max_sent_len, merged_embeddings.size()[-1])

        # Encode the setntences with LSTM. 
        # NOTE that the api of LSTM, GRU and RNN are different, the following only works for LSTM
        rnn_output, _ = self.rnn1(merged_embeddings)
        # rnn_output shape: batch * self.MAX_EDGES_PER_GRAPH, max_sent_len, hidden
        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1).view(sentence_input.size()[0], self.MAX_EDGES_PER_GRAPH, -1)
        rnn_result = self.dropout2(rnn_result)


        if(self.p['layer_number'] == 1 or self.p['projection_style'] == 'tie'):
            # 1 layer case or tied-matrices cases
            rnn_result = self.representation_to_adj(rnn_result).view(sentence_input.size()[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # magic number here
            if(self.p['non-linear'] != "linear"):
                try:
                    rnn_result = getattr(F, self.p['non-linear'])(rnn_result)
                except:
                    raise NotImplementedError
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * 8, 1).view(sentence_input.size()[0], 8, 1, (self.p['embedding_dim'] * 2) ** 2)
            rnn_result = torch.cat([identity_stuffing, rnn_result], dim=2).view(sentence_input.size()[0], 80, (self.p['embedding_dim'] * 2) ** 2)
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
            adjecent_matrix = torch.cat([rnn_result, identity_stuffing], dim=1).view(sentence_input.size()[0], 9, 9, self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                     * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)

            # adjecent_matrix = torch.matmul(adjecent_matrix, block_matrix).view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            adjecent_matrix = adjecent_matrix.view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            if(self.p['layer_number'] == 1):
                layer_1 = torch.matmul(adjecent_matrix, self.start_embedding).view(
                    sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_1 = getattr(F, self.p['non-linear1'])(layer_1)
                    except:
                        raise NotImplementedError

                heads = torch.gather(layer_1, 2, self.head_indices)
                tails = torch.gather(layer_1, 2, self.tail_indices)
                relation = heads * tails
                main_output = self.linear3(relation).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
            else:
                layer_tmp = self.start_embedding
                relation_list = []
                for i in range(self.p['layer_number']):
                    layer_tmp = torch.matmul(adjecent_matrix, layer_tmp)
                    if(self.p['non-linear1'] != 'linear'):
                        try:
                            layer_tmp = getattr(
                                F, self.p['non-linear1'])(layer_tmp)
                        except:
                            raise NotImplementedError
                    layer_result = layer_tmp.view(
                        sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                    
                    heads = torch.gather(layer_result, 2, self.head_indices)
                    tails = torch.gather(layer_result, 2, self.tail_indices)
                    relation = heads * tails
                    relation_list.append(relation)
                main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                    sentence_input.size()[0] * self.MAX_EDGES_PER_GRAPH, -1)
                return main_output
        else:
            adjecent_matrix = []
            relation_list = []
            for i in range(self.p['layer_number']):
                rnn_result_tmp = self.representation_to_adj[i](rnn_result).view(sentence_input.size(
                )[0], self.p['max_num_nodes']-1, self.p['max_num_nodes'], (self.p['embedding_dim'] * 2) ** 2)  # 9 is the num of node, 8 * 9 is the edge_num
                
                if(self.p['non-linear1'] != "linear"):
                    try:
                        rnn_result_tmp = getattr(
                            F, self.p['non-linear1'])(rnn_result_tmp)
                    except:
                        raise NotImplementedError
                        
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * (self.p['max_num_nodes']-1), 1).view(sentence_input.size()[0], (self.p['max_num_nodes']-1), 1, (self.p['embedding_dim'] * 2) ** 2)
                
                rnn_result_tmp = torch.cat([identity_stuffing, rnn_result_tmp], dim=2).view(sentence_input.size()[0], self.p['max_num_nodes']**2-1, (self.p['embedding_dim'] * 2) ** 2)
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
                adjecent_matrix.append(None)
                adjecent_matrix[i] = torch.cat([rnn_result_tmp, identity_stuffing], dim=1).view(sentence_input.size()[0], self.p['max_num_nodes'], self.p['max_num_nodes'], self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                                * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * self.p['max_num_nodes'] * 2, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)

                adjecent_matrix[i] = adjecent_matrix[i].view(sentence_input.size()[0], 1, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2, self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)
            layer_tmp = start_embedding
            for i in range(self.p['layer_number']):
                layer_tmp = torch.matmul(adjecent_matrix[i], layer_tmp)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_tmp = getattr(
                            F, self.p['non-linear1'])(layer_tmp)
                    except:
                        raise NotImplementedError
                layer_result = layer_tmp.view(
                    sentence_input.size()[0], self.p['max_num_nodes']*(self.p['max_num_nodes']-1), self.p['embedding_dim'] * self.p['max_num_nodes'] * 2)

                heads = torch.gather(layer_result, 2, self.head_indices)
                tails = torch.gather(layer_result, 2, self.tail_indices)
                relation_list.append(heads * tails)

            propagation_output = torch.cat(relation_list, dim=-1)
            propagation_output = propagation_output.view(-1,propagation_output.shape[-1])
            gat_entity_embeddings = gat_entity_embeddings.view(-1,gat_entity_embeddings.shape[-1])

            if CUDA:
              nonzero_gat_entity_embeddings_head = nonzero_gat_entity_embeddings[:,:nonzero_gat_entity_embeddings.shape[-1]//2].cuda()
              nonzero_gat_entity_embeddings_tail = nonzero_gat_entity_embeddings[:,nonzero_gat_entity_embeddings.shape[-1]//2:].cuda()
            else:
              nonzero_gat_entity_embeddings_head = nonzero_gat_entity_embeddings[:,:nonzero_gat_entity_embeddings.shape[-1]//2]
              nonzero_gat_entity_embeddings_tail = nonzero_gat_entity_embeddings[:,nonzero_gat_entity_embeddings.shape[-1]//2:]              
            nonzero_gat_entity_embeddings_head_relspace = torch.tanh(torch.matmul(\
                          nonzero_gat_entity_embeddings_head.unsqueeze(1).unsqueeze(1), \
                          self.W_ent2rel.unsqueeze(0))).squeeze()
            nonzero_gat_entity_embeddings_tail_relspace = torch.tanh(torch.matmul(\
                          nonzero_gat_entity_embeddings_tail.unsqueeze(1).unsqueeze(1), \
                          self.W_ent2rel.unsqueeze(0))).squeeze()

            translation_diff = nonzero_gat_entity_embeddings_head_relspace + self.gat_relation_embeddings.unsqueeze(0).repeat([nonzero_gat_entity_embeddings_head_relspace.shape[0],1,1]) - nonzero_gat_entity_embeddings_tail_relspace
            # if CUDA:
            #   translation_diff_all = torch.zeros([entity_indices.shape[0]*entity_indices.shape[1], self.W_ent2rel.shape[0], nonzero_gat_entity_embeddings.shape[-1]//2]).cuda()
            # else:
            #   translation_diff_all = torch.zeros([entity_indices.shape[0]*entity_indices.shape[1], self.W_ent2rel.shape[0], nonzero_gat_entity_embeddings.shape[-1]//2])
            # translation_diff_all[nonzero_entity_pos,:,:] = translation_diff
            # translation_diff_all = translation_diff_all.view(translation_diff_all.shape[0],-1)            
            translation_diff = torch.norm(translation_diff, p=1, dim=-1).squeeze()
            if CUDA:
              translation_diff_all = torch.zeros([entity_indices.shape[0]*entity_indices.shape[1], self.W_ent2rel.shape[0]]).cuda()
            else:
              translation_diff_all = torch.zeros([entity_indices.shape[0]*entity_indices.shape[1], self.W_ent2rel.shape[0]])
            translation_diff_all[nonzero_entity_pos,:] = translation_diff

            propagation_and_gat = torch.cat([propagation_output,gat_entity_embeddings,translation_diff_all], dim=-1)
            main_output = self.linear3(propagation_and_gat)
            main_output = main_output
            return main_output
