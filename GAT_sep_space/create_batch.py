import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
from torch.autograd import Variable
import pickle

MAX_CTX_LEN = 32
MAX_CTX_WORDS_LEN = 32

class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, unique_entities_test, 
                 entities_context_data, word_vocab, char_vocab,
                 get_2hop=False,get_1hop=True):
        
        self.entities_context_data = entities_context_data
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]
        # Converting to sparse tensor
        test_adj_indices = torch.LongTensor(
            [test_data[1][0], test_data[1][1]])  # rows and columns
        test_adj_values = torch.LongTensor(test_data[1][2])
        self.test_adj_matrix = (test_adj_indices, test_adj_values)

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        if get_1hop or get_2hop:
          self.graph = self.get_graph(Train=True)
          self.graph_test = self.get_graph(Train=False)
        if(get_2hop):
            self.node_neighbors_2hop = self.get_further_neighbors(nbd_size=2,Train=True)
            self.node_neighbors_2hop_test = self.get_further_neighbors(nbd_size=2,Train=False)
        if(get_1hop):
            self.node_neighbors_1hop = self.get_further_neighbors(nbd_size=1,Train=True)
            self.node_neighbors_1hop_test = self.get_further_neighbors(nbd_size=1,Train=False)

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]
        self.unique_entities_test = [self.entity2id[i]
                                      for i in unique_entities_test]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                           len(self.validation_indices), len(self.test_indices)))

        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

        self.entities_per_batch = int(np.ceil(len(self.train_indices)/self.batch_size))
        self.entities_per_batch = args.entities_per_batch
        # self.entity_triples = {}
        # for src in self.graph.keys()
        #   for rel, target in self.node_neighbors_1hop:
        #     if self.entity_triples.get(src,None):
        #       self.entity_triples[src].append([])
        #     else:
        #       self.entity_triples[src] = [[]]

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio*2 + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio*2 + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)
                random_relations = np.random.randint(
                    0, len(self.relation2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio*2, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio*2, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio):
                        current_index = last_index * \
                            self.invalid_valid_ratio + \
                            (i * self.invalid_valid_ratio + j)
                        current_rel_index = i * self.invalid_valid_ratio + j

                        rel_count = 0
                        while (self.batch_indices[last_index + current_index, 0], random_relations[current_rel_index], 
                          self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_relations[current_rel_index] = np.random.randint( 
                                0, len(self.relation2id))
                            rel_count += 1
                            if rel_count>=len(self.relation2id):
                              break

                        if rel_count<len(self.relation2id):
                          self.batch_indices[last_index + current_index,
                                             1] = random_relations[current_rel_index]
                          self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio*2 + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio*2 + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)
                random_relations = np.random.randint(
                    0, len(self.relation2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio*2, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio*2, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio):
                        current_index = last_index * \
                            self.invalid_valid_ratio + \
                            (i * self.invalid_valid_ratio + j)
                        current_rel_index = i * self.invalid_valid_ratio + j

                        rel_count = 0
                        while (self.batch_indices[last_index + current_index, 0], random_relations[current_rel_index], 
                          self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_relations[current_rel_index] = np.random.randint(
                                0, len(self.relation2id))
                            rel_count += 1
                            if rel_count>=len(self.relation2id):
                              break

                        if rel_count<len(self.relation2id):
                          self.batch_indices[last_index + current_index,
                                             1] = random_relations[current_rel_index]
                          self.batch_values[last_index + current_index, :] = [-1]


                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_iteration_triples_batch(self, batch_entities, invalid_valid_ratio=None):
        # TODO: consider inward relations as well
        if invalid_valid_ratio is None:
          invalid_valid_ratio = self.invalid_valid_ratio

        cur_entity_triples = []
        for ent in batch_entities:
          # import pdb; pdb.set_trace()
          for _tuple in self.node_neighbors_1hop.get(ent,{1:[]})[1]:
            rels = _tuple[0][0]
            for rel in rels:
              cur_entity_triples.append([ent, rel, _tuple[1][0]])

        cur_entity_triples = np.array(cur_entity_triples)
        cur_batch_size = cur_entity_triples.shape[0]
        self.batch_indices = np.empty(
            (cur_batch_size * (invalid_valid_ratio*2 + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (cur_batch_size * (invalid_valid_ratio*2 + 1), 1)).astype(np.float32)

        indices = range(0,cur_batch_size)

        self.batch_indices[:cur_batch_size,
                           :] = cur_entity_triples[indices, :]
        self.batch_values[:cur_batch_size,
                          :] = np.array(
            [[1]] * cur_batch_size).astype(np.float32)

        last_index = cur_batch_size

        if invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)
            random_relations = np.random.randint(
                0, len(self.relation2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio*2, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio*2 + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio*2, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                           self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, len(self.entity2id))
                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [-1]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                           random_entities[current_index]) in self.valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, len(self.entity2id))
                    self.batch_indices[last_index + current_index,
                                       2] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [-1]

                for j in range(self.invalid_valid_ratio):
                    current_index = last_index * \
                        self.invalid_valid_ratio + \
                        (i * self.invalid_valid_ratio + j)
                    current_rel_index = i * self.invalid_valid_ratio + j

                    rel_count = 0
                    while (self.batch_indices[last_index + current_index, 0], random_relations[current_rel_index], 
                      self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                        random_relations[current_rel_index] = np.random.randint(
                            0, len(self.relation2id))
                        rel_count += 1
                        if rel_count>=len(self.relation2id):
                          break

                    if rel_count<len(self.relation2id):
                      self.batch_indices[last_index + current_index,
                                         1] = random_relations[current_rel_index]
                      self.batch_values[last_index + current_index, :] = [-1]


        return self.batch_indices, self.batch_values


    def get_iteration_entities_batch(self, batch_entities, invalid_valid_ratio):

        cur_entities = []
        for ent in batch_entities:
          cur_entities.append(ent)

        cur_entities = np.array(cur_entities)
        cur_batch_size = cur_entities.shape[0]
        self.batch_indices = np.empty(
            (cur_batch_size * (invalid_valid_ratio + 1),)).astype(np.int32)
        self.batch_values = np.empty(
            (cur_batch_size * (invalid_valid_ratio + 1),)).astype(np.float32)

        indices = range(0,cur_batch_size)

        self.batch_indices[:cur_batch_size] = cur_entities[indices]
        self.batch_values[:cur_batch_size] = np.array(
            [1] * cur_batch_size).astype(np.float32)

        last_index = cur_batch_size

        if invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * invalid_valid_ratio)

            for i in range(last_index):
                for j in range(invalid_valid_ratio):
                    current_index = i * (invalid_valid_ratio) + j

                    while random_entities[current_index] == cur_entities[i]:
                        random_entities[current_index] = np.random.randint(
                            0, len(self.entity2id))
                    self.batch_indices[last_index + current_index] = random_entities[current_index]
                    self.batch_values[last_index + current_index] = -1

        return self.batch_indices, self.batch_values

    def get_batch_adj_data(self, iter_num, unique_entities_train=[], start_idx=None, end_idx=None):
        if len(unique_entities_train)==0:
          unique_entities_train = self.unique_entities_train
        batch_adj_indices = []
        trgts = []
        srcs = []
        batch_adj_values = []
        current_batch_entities = {
          'source': set(),
          'target': set(),
          'all': set()
        }
        # batch_adj_indices = torch.LongTensor(
        #     [train_data[1][0], train_data[1][1]])  # rows and columns
        # batch_adj_values = torch.LongTensor(train_data[1][2])
        # self.train_adj_matrix = (adj_indices, adj_values)

        if (start_idx is None or end_idx is None) and iter_num is not None:
          start_idx = iter_num*self.entities_per_batch
          end_idx = min((iter_num+1)*self.entities_per_batch,len(unique_entities_train)) 
          
        # self.batch_indices = np.empty((0,3)).astype(np.int32)
        for idx in range(start_idx,end_idx): 
          # self.batch_indices = np.concatenate(
          #   (self.batch_indices,
          #   np.array().astype(np.int32)),
          #   dim=0)
          current_batch_entities['source'].add(unique_entities_train[idx])
          current_batch_entities['all'].add(unique_entities_train[idx])
          # print(self.node_neighbors_1hop[unique_entities_train[idx]].keys())
          # import pdb; pdb.set_trace()
          for rel_tuple, target_tuple in self.node_neighbors_1hop.get(unique_entities_train[idx],{1:[]})[1]:
            target = target_tuple[0]
            rels = rel_tuple[0]
            current_batch_entities['target'].add(target)
            current_batch_entities['all'].add(target)
            for rel in rels:
              # batch_adj_indices.append([target,unique_entities_train[idx]])
              trgts.append(target)
              srcs.append(unique_entities_train[idx])
              batch_adj_values.append(rel)

        batch_adj_indices = torch.LongTensor([trgts,srcs])
        batch_adj_values = torch.LongTensor(batch_adj_values)
        train_adj_matrix = (batch_adj_indices, batch_adj_values)
        return train_adj_matrix, current_batch_entities

    def get_batch_adj_data_test(self, unique_entities_test=[]):
        if len(unique_entities_test)==0:
          unique_entities_test = self.unique_entities_test
        batch_adj_indices = []
        trgts = []
        srcs = []
        batch_adj_values = []
        current_batch_entities = {
          'source': set(),
          'target': set(),
          'all': set()
        }

        start_idx = 0
        end_idx = len(unique_entities_test)
        for idx in range(start_idx,end_idx): 
          current_batch_entities['source'].add(unique_entities_test[idx])
          current_batch_entities['all'].add(unique_entities_test[idx])
          for rel_tuple, target_tuple in self.node_neighbors_1hop_test.get(unique_entities_test[idx],{1:[]})[1]:
            target = target_tuple[0]
            rels = rel_tuple[0]
            current_batch_entities['target'].add(target)
            current_batch_entities['all'].add(target)
            for rel in rels:
              trgts.append(target)
              srcs.append(unique_entities_test[idx])
              batch_adj_values.append(rel)

        batch_adj_indices = torch.LongTensor([trgts,srcs])
        batch_adj_values = torch.LongTensor(batch_adj_values)
        train_adj_matrix = (batch_adj_indices, batch_adj_values)
        return train_adj_matrix, current_batch_entities


    def get_max_ctx_len(self, sample_batch, MAX_LEN=MAX_CTX_WORDS_LEN):
        src_max_ctx_len = 0
        for _, batch in sample_batch.items():
            if len(batch) > src_max_ctx_len:
                src_max_ctx_len = len(batch)

        return min(MAX_LEN,src_max_ctx_len)

    def get_max_src_ctx_len(self, sample_batch, MAX_LEN=MAX_CTX_LEN):
        src_max_len = 0
        for _, batch in sample_batch.items():
          for sample in batch:
            if len(sample) > src_max_len:
                src_max_len = len(sample)

        return min(MAX_LEN,src_max_len)

    def get_ctx_words_index_seq(self, words_batch, max_len, max_batch_len):
        batch_seq = list()
        words_batch = words_batch[:max_batch_len]
        for words in words_batch:
          seq = list()
          words = words[:max_len]
          for word in words:
              if word in self.word_vocab:
                  seq.append(self.word_vocab[word])
              else:
                  seq.append(self.word_vocab['<UNK>'])
          pad_len = max_len - len(words)
          for i in range(0, pad_len):
              seq.append(self.word_vocab['<PAD>'])      
          batch_seq.append(seq)

        pad_batch_len = max_batch_len - len(words_batch)
        for i in range(0, pad_batch_len):
            seq = list()
            for _ in range(max_len):
              seq.append(self.word_vocab['<PAD>'])
            batch_seq.append(seq)

        return batch_seq


    def get_ctx_char_seq(self, words_batch, max_len, max_batch_len, conv_filter_size, max_word_len):
        batch_char_seq = list()
        words_batch = words_batch[:max_batch_len]
        for words in words_batch:
          char_seq = list()
          words = words[:max_len]
          for i in range(0, conv_filter_size - 1):
              char_seq.append(self.char_vocab['<PAD>'])
          for word in words:
              for c in word[0:min(len(word), max_word_len)]:
                  if c in self.char_vocab:
                      char_seq.append(self.char_vocab[c])
                  else:
                      char_seq.append(self.char_vocab['<UNK>'])
              pad_len = max_word_len - len(word)
              for i in range(0, pad_len):
                  char_seq.append(self.char_vocab['<PAD>'])
              for i in range(0, conv_filter_size - 1):
                  char_seq.append(self.char_vocab['<PAD>'])

          pad_len = max_len - len(words)
          for i in range(0, pad_len):
              for i in range(0, max_word_len + conv_filter_size - 1):
                  char_seq.append(self.char_vocab['<PAD>'])

          batch_char_seq.append(char_seq)


        pad_len_batch = max_batch_len - len(words_batch)
        for i in range(0, pad_len_batch):
          char_seq = list()
          for i in range(0, conv_filter_size - 1):
              char_seq.append(self.char_vocab['<PAD>'])
          for i in range(0, max_len):
            for i in range(0, max_word_len + conv_filter_size - 1):
                char_seq.append(self.char_vocab['<PAD>'])
          batch_char_seq.append(char_seq)

        return batch_char_seq

    def get_padded_mask_ctx(self, words_batch, max_len, max_batch_len):
        mask_seq = list()
        words_batch = words_batch[:max_batch_len]
        for words in words_batch:
          mask_seq.append(0)

        pad_batch_len = max_batch_len - len(words_batch)
        for i in range(0, pad_batch_len):
          mask_seq.append(1)

        return mask_seq

    def get_ent_data(self, batch_ctx_data, conv_filter_size, max_word_len, is_training=False):
        """
        Returns the entity context data as numpy array
        """
        batch_ctx_max_len = self.get_max_ctx_len(batch_ctx_data)
        batch_ctx_src_max_len = self.get_max_src_ctx_len(batch_ctx_data)

        ent_data = {}
        entity_indices = []

        for entity_id, sample in batch_ctx_data.items():
            ctx_words = self.get_ctx_words_index_seq(sample, batch_ctx_src_max_len, batch_ctx_max_len)
            ctx_char = self.get_ctx_char_seq(sample, batch_ctx_src_max_len, batch_ctx_max_len, conv_filter_size, max_word_len)
            mask = self.get_padded_mask_ctx(sample, batch_ctx_src_max_len, batch_ctx_max_len)
            entity_indices.append(entity_id) 

            ent_data[entity_id] = {
              'ctx_words': np.array(ctx_words),
              'ctx_chars': np.array(ctx_char),
              'ctx_mask': np.array(mask)
            }

        return ent_data, entity_indices

    def get_batch_entities_ctx_data(self, batch_indices, conv_filter_size, max_word_len, nhop_indices=[], triple=False):
      batch_ctx_src_words = []
      batch_ctx_src_chars = []
      batch_ctx_mask = []

      unique_entities = set()
      if triple:
        for row in batch_indices:
          unique_entities.add(row[0])
          unique_entities.add(row[2])
        for row in nhop_indices:
          unique_entities.add(row[0])
          unique_entities.add(row[-1])
      else:
        for ent in batch_indices:
          unique_entities.add(ent)

      unique_entities = list(unique_entities)
      batch_ctx_data = {}
      for ent in unique_entities:
          batch_ctx_data[str(ent)] = []
          # import pdb; pdb.set_trace()
          # print(ent)
          # print(self.id2entity[ent])
          ent_ctx = self.entities_context_data.get(self.id2entity[ent], {
              'label': '',
              'desc': '',
              'instances': [],
              'aliases': []
            })


          batch_ctx_data[str(ent)] = [word_tokenize(ent_ctx['label']),word_tokenize(ent_ctx['desc'])]
          instanceof_arr = [word_tokenize('instance of {}'.format(i)) for i in ent_ctx['instances']]
          alias_arr = [word_tokenize('also known as {}'.format(i)) for i in ent_ctx['aliases']]
          if len(instanceof_arr)+len(alias_arr)>MAX_CTX_LEN-2:
            if len(instanceof_arr)>(MAX_CTX_LEN-2)//2 and \
               len(alias_arr)>(MAX_CTX_LEN-2)//2:
               instanceof_arr = instanceof_arr[:(MAX_CTX_LEN-2)//2]
               alias_arr = alias_arr[:(MAX_CTX_LEN-2)//2]
            elif len(instanceof_arr)>(MAX_CTX_LEN-2)//2:
              instanceof_arr = instanceof_arr[:(MAX_CTX_LEN-2-len(alias_arr))]
            elif len(alias_arr)>(MAX_CTX_LEN-2)//2:
              alias_arr = instanceof_arr[:(MAX_CTX_LEN-2-len(instanceof_arr))]
          batch_ctx_data[str(ent)].extend(instanceof_arr)
          batch_ctx_data[str(ent)].extend(alias_arr)


      ent_data, entity_indices = self.get_ent_data(batch_ctx_data,conv_filter_size,max_word_len)

      # for row in batch_indices:
      #   batch_ctx_src_words.append(ent_data[str(row[0])]['ctx_words'])
      #   batch_ctx_src_chars.append(ent_data[str(row[0])]['ctx_chars'])
      #   batch_ctx_mask.append(ent_data[str(row[0])]['ctx_mask'])

      #   batch_ctx_src_words.append(ent_data[str(row[2])]['ctx_words'])
      #   batch_ctx_src_chars.append(ent_data[str(row[2])]['ctx_chars'])
      #   batch_ctx_mask.append(ent_data[str(row[2])]['ctx_mask'])
      for entity, _ in ent_data.items():
        batch_ctx_src_words.append(ent_data[str(entity)]['ctx_words'])
        batch_ctx_src_chars.append(ent_data[str(entity)]['ctx_chars'])
        batch_ctx_mask.append(ent_data[str(entity)]['ctx_mask'])

      return {
        'ctx_words_list': np.array(batch_ctx_src_words),
        'ctx_char_seq': np.array(batch_ctx_src_chars),
        'ctx_mask': np.array(batch_ctx_mask),  
        'ctx_entity_indices': np.array(entity_indices)    
      }

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph(self, Train=True):
        graph = {}
        if Train:
          all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
              0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)
        else:
          all_tiples = torch.cat([self.test_adj_matrix[0].transpose(
              0, 1), self.test_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if graph.get(source,None) is None:
                graph[source] = {}
                graph[source][target] = [value]
            else:
              if graph[source].get(target,None) is None:
                graph[source][target] = [value]
              else:
                graph[source][target].append(value)

        print("Graph created")
        return graph

    # def bfs(self, graph, source, nbd_size=2):
    #     visit = {}
    #     distance = {}
    #     parent = {}
    #     distance_lengths = {}

    #     visit[source] = 1
    #     distance[source] = 0
    #     parent[source] = (-1, -1)

    #     q = queue.Queue()
    #     q.put((source, -1))

    #     while(not q.empty()):
    #         top = q.get()
    #         if top[0] in graph.keys():
    #             for target in graph[top[0]].keys():
    #                 if(target in visit.keys()):
    #                     continue
    #                 else:
    #                     q.put((target, graph[top[0]][target]))

    #                     distance[target] = distance[top[0]] + 1

    #                     visit[target] = 1
    #                     if distance[target] > 2:
    #                         continue
    #                     parent[target] = (top[0], graph[top[0]][target])

    #                     if distance[target] not in distance_lengths.keys():
    #                         distance_lengths[distance[target]] = 1

    #     neighbors = {}
    #     for target in visit.keys():
    #         if(distance[target] != nbd_size):
    #             continue
    #         edges = [-1, parent[target][1]]
    #         relations = []
    #         entities = [target]
    #         temp = target
    #         while(parent[temp] != (-1, -1)):
    #             relations.append(parent[temp][1])
    #             entities.append(parent[temp][0])
    #             temp = parent[temp][0]

    #         if(distance[target] in neighbors.keys()):
    #             neighbors[distance[target]].append(
    #                 (tuple(relations), tuple(entities[:-1])))
    #         else:
    #             neighbors[distance[target]] = [
    #                 (tuple(relations), tuple(entities[:-1]))]

    #     return neighbors

    def bfs(self, graph, source, nbd_size=1):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                      distance[target] = distance[top[0]] + 1

                      if distance[target] > nbd_size:
                          continue

                      q.put((target, graph[top[0]][target]))

                      visit[target] = 1
                        
                      parent[target] = (top[0], graph[top[0]][target])

                      if distance[target] not in distance_lengths.keys():
                          distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_further_neighbors(self, nbd_size=2, Train=True):
        neighbors = {}
        start_time = time.time()
        if Train:
          graph = self.graph
        else:
          graph = self.graph_test
        print("length of graph keys is ", len(graph.keys()))
        for source in tqdm(graph.keys()):
            # st_time = time.time()
            temp_neighbors = self.bfs(graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        # print("length of unique_entities ", len(batch_sources))
        count = 0
        # import pdb; pdb.set_trace()
        for source in batch_sources:
              # randomly select from the list of neighbors
              # if source in node_neighbors.keys():
              nhop_list = node_neighbors.get(source,{nbd_size: []})[nbd_size]

              # nhop_list = node_neighbors[source][nbd_size]

              for i, tup in enumerate(nhop_list):
                  if(args.partial_2hop and i >= 1):
                      break

                  count += 1
                  # TODO: permutation of the rels
                  batch_source_triples.append([source, nhop_list[i][0][-1][0], nhop_list[i][0][0][0],
                                               nhop_list[i][1][0]])

                  if len(batch_source_triples[-1])!=4:
                    print(source,i,nhop_list[i],len(batch_source_triples[-1]))
        # import pdb;  pdb.set_trace()
        return np.array(batch_source_triples).astype(np.int32)

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self, args, model, unique_entities):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in tqdm(range(batch_indices.shape[0])):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 4))

                    scores1_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                    #scores5_head, scores6_head, scores7_head, scores8_head,
                    # cores9_head, scores10_head], dim=0)
                else:
                    num_iters = int(np.ceil(new_x_batch_head.shape[0]/10000))
                    scores_head_arr = []
                    for batch_iter in tqdm(range(num_iters)):
                      start_idx = batch_iter*10000
                      end_idx = min((batch_iter+1)*10000,new_x_batch_head.shape[0])
                      cur_scores_head = model.batch_test(torch.LongTensor(
                          new_x_batch_head[start_idx:end_idx, :]).cuda())
                      scores_head_arr.append(cur_scores_head)
                      
                    scores_head = torch.cat(
                        scores_head_arr, dim=0)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 4))

                    scores1_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)
                    #     scores5_tail, scores6_tail, scores7_tail, scores8_tail,
                    #     scores9_tail, scores10_tail], dim=0)

                else:
                    num_iters = int(np.ceil(new_x_batch_tail.shape[0]/10000))
                    scores_tail_arr = []
                    for batch_iter in tqdm(range(num_iters)):
                      start_idx = batch_iter*10000
                      end_idx = min((batch_iter+1)*10000,new_x_batch_tail.shape[0])
                      cur_scores_tail = model.batch_test(torch.LongTensor(
                          new_x_batch_tail[start_idx:end_idx, :]).cuda())
                      scores_tail_arr.append(cur_scores_tail)
                      
                    scores_tail = torch.cat(
                        scores_tail_arr, dim=0)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                print("sample - ", ranks_head[-1], ranks_tail[-1])

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            print("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

    def write_test_pr(self, cnf_mats, outfile):
        writer = open(outfile, 'w')

        for r, cnf_mat in cnf_mats.items():
            if cnf_mat[1][1]+cnf_mat[0][1]>0:
              pr = cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[0][1])
            else:
              pr = 'NA'

            if cnf_mat[1][1]+cnf_mat[1][0]>0:
              rc = cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
            else:
              rc = 'NA'

            if pr=='NA' and rc=='NA':
              f = 'NA'
            elif (pr=='NA' and rc!='NA') or (pr!='NA' and rc=='NA'):
              f = 0
            else:
              f = 2*pr*rc/(pr+rc+1e-8)

            pr_count = cnf_mat[0][1]+cnf_mat[1][1]
            rc_count = cnf_mat[1][1]+cnf_mat[1][0]
            f_count = cnf_mat[1][1]+cnf_mat[1][0]+cnf_mat[0][1]

            writer.write('{} | {} | {} | {} | {} | {} | {}'.format(r,pr,rc,f,pr_count,rc_count,f_count) + '\n')
        writer.close()

    def write_test(self, actual_json, preds_json, out_file):
      file = open(out_file,'w')
      for entity_pair, actual_rels in actual_json.items():
        pred_rels_list = [int(i) for i in list(preds_json[entity_pair])]
        actual_rels_list = [int(i) for i in list(actual_rels)]
        e1, e2 = entity_pair.split('->')
        file.write('{} | {} | {} | {}\n'.format(e1,e2,pred_rels_list,actual_rels_list))
      file.close()

    def get_validation_cnfmat(self, args, model_gat, model_entity_embedding, model_conv, unique_entities_train, unique_entities_test, reuse=True, use_ent_emb_module=False, use_gat_for_getting_test_entities=True, gat_only=False):

        # Get the initial entity embeddings from the context
        CUDA = torch.cuda.is_available()
        test_batch = []

        if not reuse:
          if use_gat_for_getting_test_entities:
            unique_entities_train_set = set(unique_entities_train)
            unique_entities_test_set = set(unique_entities_test)
            unique_entities_test_not_in_train = unique_entities_test_set.difference(unique_entities_train_set)
            unique_entities_test_not_in_train = list(unique_entities_test_not_in_train)

            if use_ent_emb_module:
              entity_emb = []
              ent_batch_size = 100
              num_ent_iters = int(np.ceil(len(unique_entities_test_not_in_train)/ent_batch_size))
              print('== Getting entity embeddings for unique test entities')
              for idx in tqdm(range(num_ent_iters)):
                  start_idx = idx*ent_batch_size
                  end_idx = min(start_idx+ent_batch_size,len(unique_entities_test_not_in_train))
                  batch_entities = unique_entities_test_not_in_train[start_idx:end_idx]
                  batch_entities_ctx_data = self.get_batch_entities_ctx_data(batch_entities,args.conv_filter_size,args.max_word_len,triple=False)
                
                  ctx_words = torch.from_numpy(batch_entities_ctx_data['ctx_words_list'].astype('long'))
                  ctx_char_seq = torch.from_numpy(batch_entities_ctx_data['ctx_char_seq'].astype('long'))
                  ctx_mask = torch.from_numpy(batch_entities_ctx_data['ctx_mask'].astype('bool'))

                  if CUDA:
                      ctx_words = Variable(ctx_words).cuda() 
                      ctx_char_seq = Variable(ctx_char_seq).cuda()
                      ctx_mask = Variable(ctx_mask).cuda()
                  else: 
                      ctx_words = Variable(ctx_words) 
                      ctx_char_seq = Variable(ctx_char_seq)
                      ctx_mask = Variable(ctx_mask)

                  batch_entity_embeddings = model_entity_embedding(
                      ctx_words, ctx_char_seq, ctx_mask)            
                  entity_emb.append(batch_entity_embeddings)
                  
              entity_emb = torch.cat(entity_emb,axis=0)

              if CUDA:
                entity_embeddings = model_gat.entity_embeddings.detach().cuda()
              else:
                entity_embeddings = model_gat.entity_embeddings.detach()
              entity_embeddings[unique_entities_test_not_in_train, :] = entity_emb
            else:
              if CUDA:
                entity_embeddings = model_gat.entity_embeddings.detach().cuda()
              else:
                entity_embeddings = model_gat.entity_embeddings.detach()

            # Get the final embeddings for the entities unique to the test set
            if(args.use_2hop):
                print("Opening node_neighbors pickle object")
                file = args.data + "/2hop_test.pickle"
                with open(file, 'rb') as handle:
                    node_neighbors_2hop_test = pickle.load(handle)


            
            model_conv.final_entity_embeddings.data = model_gat.final_entity_embeddings.data
            model_conv.final_relation_embeddings.data = model_gat.final_relation_embeddings.data
            ent_batch_size = 100
            num_ent_iters = int(np.ceil(len(unique_entities_test_not_in_train)/ent_batch_size))
            print('== Getting entity embeddings for unique test entities from GAT')
            for iters in tqdm(range(num_ent_iters)):
              start_idx = iters*ent_batch_size
              end_idx = min((iters+1)*ent_batch_size,len(unique_entities_test_not_in_train))
              
              current_unique_entities = unique_entities_test_not_in_train[start_idx:end_idx]

              batch_test_adj_matrix, current_batch_entities_set = self.get_batch_adj_data_test(unique_entities_test=current_unique_entities)
              if batch_test_adj_matrix[0][0].shape[0]==0:
                continue

              test_indices, _ = self.get_iteration_entities_batch(list(current_batch_entities_set['source']),invalid_valid_ratio=0)

              if CUDA:
                  test_indices = Variable(
                      torch.LongTensor(test_indices)).cuda()
                  # test_values = Variable(torch.FloatTensor(test_values)).cuda()
              else:
                  test_indices = Variable(torch.LongTensor(test_indices))
                  # test_values = Variable(torch.FloatTensor(test_values))   


              if args.use_2hop:
                current_batch_2hop_indices = self.get_batch_nhop_neighbors_all(args,
                                                                          list(current_batch_entities_set['source']), node_neighbors_2hop_test)
                if current_batch_2hop_indices.shape[0]==0:
                  # current_batch_2hop_indices = np.empty((1,4))
                  current_batch_2hop_indices = torch.tensor([],dtype=torch.long)
                if CUDA:
                    current_batch_2hop_indices = Variable(
                        torch.LongTensor(current_batch_2hop_indices)).cuda()
                else:
                    current_batch_2hop_indices = Variable(
                        torch.LongTensor(current_batch_2hop_indices))

              entity_embed, relation_embed, mask = model_gat.batch_test(
                  self, torch.tensor(list(current_batch_entities_set['source']),dtype=torch.long), batch_test_adj_matrix, current_batch_2hop_indices, entity_embeddings)
              
              model_conv.final_entity_embeddings.data[current_unique_entities] = entity_embed.data[current_unique_entities]
              # model_conv.final_relation_embeddings.data = relation_embed.data

            del model_gat
            del model_entity_embedding
            del test_indices
            del current_batch_2hop_indices
            del batch_test_adj_matrix
            del node_neighbors_2hop_test
            del entity_embeddings
          else:
            model_conv.final_entity_embeddings.data = model_gat.final_entity_embeddings.data
            model_conv.final_relation_embeddings.data = model_gat.final_relation_embeddings.data            

          # Finally make the predictions
          for i in tqdm(range(self.test_indices.shape[0])):
              test_batch.append(self.test_indices[i])
          # for i in tqdm(range(1000)):
          #     test_batch.append(self.train_indices[i])
          test_batch = np.array(test_batch)

          num_rels = len(self.relation2id)
          rel_ids = range(num_rels)
          test_batch_pred = test_batch.copy()
          test_batch_pred = np.expand_dims(test_batch_pred,axis=1)
          test_batch_pred = np.tile(test_batch_pred,(1,num_rels,1))
          print('== Making final predictions')
          for rel_id in rel_ids:
            test_batch_pred[:,rel_id,1] = rel_id
          test_batch_pred = np.reshape(test_batch_pred,(-1,3))

          test_batch_size = 100
          num_iters = int(np.ceil(test_batch_pred.shape[0]/test_batch_size))
          scores_arr = torch.empty((0,1))
          if CUDA:
            scores_arr = scores_arr.cuda()
          for batch_iter in tqdm(range(num_iters)):
            start_idx = batch_iter*test_batch_size
            end_idx = min((batch_iter+1)*test_batch_size,test_batch_pred.shape[0])
            
            cur_scores = model_conv.batch_test(torch.LongTensor(
                test_batch_pred[start_idx:end_idx, :], model_gat).cuda())

            # print(torch.max(cur_scores),torch.min(cur_scores))
            # scores_arr.append(cur_scores)
            scores_arr = torch.cat((scores_arr,cur_scores),dim=0)
            
          # scores = torch.cat(
          #     scores_arr, dim=0)
          scores = scores_arr
          if gat_only:
            torch.save(scores,'scores_gatonly.pth')
          else:
            torch.save(scores,'scores.pth')
        else:
          for i in tqdm(range(self.test_indices.shape[0])):
            test_batch.append(self.test_indices[i])

          if gat_only:
            scores = torch.load('scores_gatonly.pth')
          else:
            scores = torch.load(scores,'scores.pth')

          num_rels = len(self.relation2id)

        # for i in tqdm(range(1000)):
        #     test_batch.append(self.train_indices[i])
        test_batch = np.array(test_batch)
        # scores_np = np.array(scores.cpu())
        # np.save('scores.npy',scores_np)
        # # scores = np.array(scores.cpu())

        # scores = np.reshape(scores,(-1,num_rels))
        scores = scores.view((-1,num_rels))
        sorted_scores, sorted_indices = torch.sort(
                    scores, dim=-1, descending=True)
        # pred_rels = np.argmax(scores,axis=-1)
        actual_json = {}
        cnf_mat_json = {}
        preds_json = {}
        for _, r in self.relation2id.items():
          cnf_mat_json[r] = [[0,0],[0,0]]

        for i in range(test_batch.shape[0]):
          e1 = test_batch[i][0]
          e2 = test_batch[i][2]
          r = test_batch[i][1]
          k = '{}->{}'.format(e1,e2)
          if actual_json.get(k,None) is None:
            actual_json[k] = set()
          actual_json[k].add(r)
        for i in range(sorted_scores.shape[0]):
          e1 = test_batch[i][0]
          e2 = test_batch[i][2]
          k = '{}->{}'.format(e1,e2)
          cur_entity_pair_num_rels = len(actual_json[k])
          pred_rels = [si.item() for si in sorted_indices[i][:max(cur_entity_pair_num_rels,10)]]
          k = '{}->{}'.format(e1,e2)
          if preds_json.get(k,None) is None:
            preds_json[k] = set(pred_rels)

        num_actual_rels = 0
        hits_at_10 = 0
        ranks = []
        reciprocal_ranks = []
        cur_triple_count = 0
        for entity_pair, actual_rels in tqdm(actual_json.items()):
          pred_rels = preds_json[entity_pair]

          true_positive_rels = actual_rels.intersection(pred_rels)
          false_positive_rels = pred_rels.difference(true_positive_rels)
          false_negative_rels = actual_rels.difference(true_positive_rels)

          hits_at_10 += len(true_positive_rels)
          for rel in actual_rels:
            match_ranks = (sorted_indices[i]==rel).nonzero()
            if len(match_ranks)>0:
              ranks.append(match_ranks[0].item()+1)
              reciprocal_ranks.append(1.0/(ranks[-1]))

          # # num_actual_rels += len(actual_rels)
          # # correct
          # for r in true_positive_rels:
          #   for _, p in self.relation2id.items():
          #       if p==r:
          #           cnf_mat_json[p][1][1] += 1
          #       else:
          #           cnf_mat_json[p][0][0] += 1          
          # for r in false_positive_rels:
          #   for _, p in self.relation2id.items():
          #       if p==r:
          #         cnf_mat_json[p][0][1] += 1
          #       if p in false_negative_rels:
          #         cnf_mat_json[p][1][0] += 1
          #       else:
          #         cnf_mat_json[p][0][0] += 1
          # # for r in false_negative_rels:
          # #   for p, _ in self.relation2id.items():
          # #       if p==r:
          # #         cnf_mat_json[p][1][0] += 1
          # #       if p in false_negative_rels:
          # #         cnf_mat_json[p][1][0] += 1
          # #       else:
          # #         cnf_mat_json[p][0][0] += 1

          cur_triple_count += 1

        print(hits_at_10)
        print(sum(reciprocal_ranks))
        average_hits_at_10 = hits_at_10*1.0/len(ranks)
        average_rank = sum(ranks)*1.0/len(ranks)
        average_recip_rank = sum(reciprocal_ranks)*1.0/len(reciprocal_ranks)
        print('average_hits_at_10: ',average_hits_at_10)
        print('average_rank: ',average_rank)
        print('average_recip_rank: ',average_recip_rank)
        self.write_test(actual_json,preds_json,'test.out')
        # self.write_test_pr(cnf_mat_json,'test_pr.out')


        
