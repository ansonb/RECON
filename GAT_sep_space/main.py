import matplotlib
matplotlib.use('Agg')

import torch

from models import SpKBGATModified, SpKBGATConvOnly, EntityEmbedding
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle
import json
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import datetime
import glob

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="../data/WikipediaWikidataDistantSupervisionAnnotations.v1.0", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-e_e", "--epochs_ent_emb", type=int,
                      default=20, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g1hop", "--get_1hop", type=bool, default=True)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-u1hop", "--use_1hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="../models/GAT_sep_space/WikipediaWikidataDistantSupervisionAnnotations/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=128, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float, 
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=64, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")
    
    # arguments for the entity embedding network
    args.add_argument("-ef", "--embedding_file", type=str,
                      default='../data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/glove.6B.50d.txt', help="The word2vec embedding file for initialising the entity embeddings")
    args.add_argument("-sv", "--save_vocab", type=str,
                      default='./vocab.pkl', help="The path of the file in which to save the vocab")
    args.add_argument("-ctx_file", "--entities_context_data_file", type=str,
                      default='../data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entities_context_augmented.json', help="The path of the file in which to save the vocab")
    args.add_argument("-wrd_emb_dim", "--word_embed_dim", type=int,
                      default=50, help="The embedding dimension of the word vector")
    args.add_argument("-word_min_freq", "--word_min_freq", type=int,
                      default=2, help="The minimum frequency for a word to occur to be considered in the vocabulary")
    args.add_argument("-cfs", "--conv_filter_size", type=int,
                      default=3, help="The 1d convolution window size for char embedd")
    args.add_argument("-mwl", "--max_word_len", type=int,
                      default=10, help="The maximum number of characters to consider in a word for learning character embeddings")
    args.add_argument("-epb", "--entities_per_batch", type=int,
                      default=128, help="The maximum number of entities to consider per iteration while passing through the model")
    args.add_argument("-neg_s_ent", "--valid_invalid_ratio_entity_embed", type=int,
                      default=4, help="The maximum number of entities to consider per iteration while passing through the model")


    args = args.parse_args()
    return args


args = parse_args()
embedding_file = args.embedding_file
save_vocab = args.save_vocab
entities_context_data_file = args.entities_context_data_file
word_embed_dim = args.word_embed_dim
word_min_freq = 2
USE_VOCAB = False
if not os.path.exists(args.output_folder):
  os.makedirs(args.output_folder)
# %%

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
          return obj.tolist()
        elif isinstance(obj, datetime.datetime):
          return str(obj)
        elif isinstance(obj, np.bool_):
          return bool(obj)
        else:
            return json.JSONEncoder.default(self, obj)

def load_word_embedding(embed_file, vocab):
    print('vocab length:', len(vocab))
    embed_vocab = OrderedDict()
    embed_matrix = list()

    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))

    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    word_idx = 2
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and vocab[word] >= word_min_freq:
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                embed_vocab[word] = word_idx
                word_idx += 1

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_min_freq:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1

    print('embed dictionary length:', len(embed_vocab))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(args, entities_context_data, save_vocab, embedding_file, use_vocab=USE_VOCAB, save_path=['./vocab.json','./char_v.json']):
    
    if use_vocab:
      print(save_path[0])
      with open(save_path[0],'r') as f:
        vocab = json.load(f)
      print(save_path[1])
      with open(save_path[1],'r') as f:
        char_v = json.load(f)
    else:
      train_data_path = os.path.join(args.data,'train.txt')
      unique_entities = set()
      with open(train_data_path, 'r') as f:
        train_triples = f.read()
        train_ents = []
        lines = train_triples.split('\n')
        for line in lines:
          ids = line.split(' ')
          if len(ids)!=3:
            continue
          unique_entities.add(ids[0])
          unique_entities.add(ids[2])
        unique_entities = list(unique_entities)

      vocab = OrderedDict()
      char_v = OrderedDict()
      char_v['<PAD>'] = 0
      char_v['<UNK>'] = 1
      char_idx = 2
      for ent in tqdm(unique_entities):
          sents = []

          ent_ctx = entities_context_data.get(ent, {
              'label': ent,
              'desc': '',
              'instances': [],
              'aliases': []
            })
          label_arr = [word_tokenize(ent_ctx['label'])]
          desc_arr = [word_tokenize(ent_ctx['desc'])]
          instanceof_arr = [word_tokenize('instance of {}'.format(i)) for i in ent_ctx['instances']]
          alias_arr = [word_tokenize('also known as {}'.format(i)) for i in ent_ctx['aliases']]
          sents.extend(label_arr)
          sents.extend(desc_arr)
          sents.extend(instanceof_arr)
          sents.extend(alias_arr)

          for sent in sents:
            for word in sent:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

                for c in word:
                    if c not in char_v:
                        char_v[c] = char_idx
                        char_idx += 1
      with open(save_path[0],'w') as f:
        json.dump(vocab, f)
      with open(save_path[1],'w') as f:
        json.dump(char_v, f)

    word_v, embed_matrix = load_word_embedding(embedding_file, vocab)
    output = open(save_vocab, 'wb')
    pickle.dump([word_v, char_v], output)
    output.close()
    return word_v, char_v, embed_matrix

def load_data(args, word_vocab, char_vocab, entities_context_data):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train, unique_entities_test = build_data(
        args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, unique_entities_test, 
                    entities_context_data, word_vocab, char_vocab, args.get_2hop, args.get_1hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)

with open(entities_context_data_file, 'r') as f:
  entities_context_data = json.load(f)
word_vocab, char_vocab, word_embed_matrix = build_vocab(args, entities_context_data, save_vocab, embedding_file)
Corpus_, entity_embeddings, relation_embeddings = load_data(args, word_vocab, char_vocab, entities_context_data)
initial_entity_emb_params = {
  'entity_embed_dim_in': 100,
  'hidden_dim_entity': 50,
  'num_encoder_layers_entity': 2,
  'is_bidirectional': True,
  'drop_out_rate': 0.5,
  'entity_embed_dim_out': 50,
  'entity_conv_filter_size': 1,
  'word_vocab': word_vocab,
  'word_embed_dim': 50,
  'char_embed_dim': 50,
  'word_embed_matrix': word_embed_matrix,
  'char_feature_size': 50,
  'conv_filter_size': args.conv_filter_size,
  'max_word_len_entity': 10,
  'char_vocab': char_vocab
}

if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/2hop_test.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop_test, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
if(args.get_1hop):
    file = args.data + "/1hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_1hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/1hop_test.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_1hop_test, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)
    file = args.data + "/2hop_test.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop_test = pickle.load(handle)
    Corpus_.node_neighbors_2hop = node_neighbors_2hop
    Corpus_.node_neighbors_2hop_test = node_neighbors_2hop_test
if(args.use_1hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/1hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_1hop = pickle.load(handle)
    file = args.data + "/1hop_test.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_1hop_test = pickle.load(handle)
    Corpus_.node_neighbors_1hop = node_neighbors_1hop
    Corpus_.node_neighbors_1hop_test = node_neighbors_1hop_test

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed, model_gat):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat)*2 + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat)*2, 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]
    # Convert entity to relation space per triple
    W_ent2rel = model_gat.W_ent2rel[pos_triples[:, 1]]
    source_embeds = source_embeds.unsqueeze(1)
    tail_embeds = tail_embeds.unsqueeze(1)
    source_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(source_embeds,W_ent2rel)).squeeze()
    tail_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(tail_embeds,W_ent2rel)).squeeze()

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]
    # Convert entity to relation space per triple
    W_ent2rel = model_gat.W_ent2rel[neg_triples[:, 1]]
    source_embeds = source_embeds.unsqueeze(1)
    tail_embeds = tail_embeds.unsqueeze(1)
    source_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(source_embeds,W_ent2rel)).squeeze()
    tail_embeds = model_gat.nonlinearity_ent2rel(torch.bmm(tail_embeds,W_ent2rel)).squeeze()

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    if CUDA:
      y = -torch.ones(int(args.valid_invalid_ratio_gat*2) * len_pos_triples).cuda()
    else:
      y = -torch.ones(int(args.valid_invalid_ratio_gat*2) * len_pos_triples)

    loss = gat_loss_func(pos_norm, neg_norm, y)

    assert not torch.isnan(loss).any()

    return loss


def batch_entity_embedding_loss(train_indices, train_values, entity_embed_gat, entity_embeddings):
    # Negative sampling loss (http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
    # ref: https://github.com/kefirski/pytorch_NEG_loss/blob/master/NEG_loss/neg.py
    # use nce(https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf) instead?

    len_pos_entities = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_entity_embed) + 1))
    pos_indices = train_indices[:len_pos_entities]
    # pos_indices = pos_indices.repeat(1+int(args.valid_invalid_ratio_entity_embed), 1)
    neg_indices = train_indices[len_pos_entities:]
    # all_indices = torch.cat((pos_indices[:len_pos_entities],neg_indices),dim=0)

    gat_embeddings_pos = entity_embed_gat[pos_indices]
    gat_embeddings_neg = entity_embed_gat[neg_indices]
    gat_embeddings_neg = gat_embeddings_neg.unsqueeze(dim=1).view(-1,args.valid_invalid_ratio_entity_embed,args.embedding_size)
    ctx_ent_emb = entity_embeddings

    log_target = (ctx_ent_emb * gat_embeddings_pos).sum(1).squeeze().sigmoid().log()
    ''' ∑[batch_size , num_sampled, embed_size] * [batch_size , embed_size, 1] ->
        ∑[batch_size, num_sampled, 1] -> [batch_size] '''
    sum_log_sampled = torch.bmm(gat_embeddings_neg.neg(), ctx_ent_emb.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

    loss = log_target + sum_log_sampled

    batch_size = len_pos_entities
    return -loss.sum() / batch_size 

def save_embed(embeddings, save_path):
  emb_data = {}
  # import pdb; pdb.set_trace()
  for idx in range(embeddings.shape[0]):
    emb_data[idx] = np.array(embeddings[idx].cpu().detach())

  with open(save_path,'w') as f:
    json.dump(emb_data,f,indent=4,cls=CustomEncoder)


def train_gat(args, word_embed_matrix, word_vocab, char_vocab):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    global initial_entity_emb_params
    global entity_embeddings

    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, initial_entity_emb_params)
    # if os.path.exists('{0}trained_{1}.pth'.format(args.output_folder, 0)):
    #   if CUDA:
    #     model_gat.load_state_dict(torch.load(
    #       os.path.join(args.output_folder,'trained_{}.pth'.format(0))))
    #   else:
    #     model_gat.load_state_dict(torch.load(
    #       os.path.join(args.output_folder,'trained_{}.pth'.format(0)),map_location=torch.device('cpu')))
    paths = glob.glob(os.path.join(args.output_folder, 'trained_*.pth'))
    if len(paths)>0:
      path = paths[0]
      start_epoch = int(path.split('/')[-1].split('_')[-1].split('.')[0]) + 1
      if CUDA:
        model_gat.load_state_dict(torch.load(
          path))
      else:
        model_gat.load_state_dict(torch.load(
          path),map_location=torch.device('cpu'))
    else:
      start_epoch = 0

    display_every = 100

    if CUDA:
        model_gat.cuda()

    # optimizer = torch.optim.Adam(
    #     model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    optimizer = torch.optim.SGD(
        model_gat.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)


    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(start_epoch, args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        num_iters_per_epoch = int(np.ceil(
                        len(Corpus_.unique_entities_train) / args.batch_size_gat))
        random_indices = np.arange(0,len(Corpus_.unique_entities_train))
        np.random.shuffle(random_indices)
        random_unique_entities_train = []
        for idx in range(len(random_indices)):
          random_unique_entities_train.append(Corpus_.unique_entities_train[random_indices[idx]])

        start_idx = end_idx = previous_end_idx = 0
        reduced_batch_size = None
        iters = 0
        # for iters in tqdm(range(num_iters_per_epoch)):
        pbar = tqdm(total = num_iters_per_epoch)
        while True:
          try: 
            if end_idx>=len(random_unique_entities_train):
              break

            if CUDA:
              torch.cuda.empty_cache()

            start_time_iter = time.time()

            start_idx = end_idx
            if reduced_batch_size:
              end_idx = min(len(random_unique_entities_train), start_idx + reduced_batch_size)
            else:
              end_idx = min(len(random_unique_entities_train), start_idx + args.entities_per_batch)

            batch_train_adj_matrix, current_batch_entities_set = Corpus_.get_batch_adj_data(iters,start_idx=start_idx,end_idx=end_idx,unique_entities_train=random_unique_entities_train)
            if batch_train_adj_matrix[0][0].shape[0]==0:
              continue

            train_indices, train_values = Corpus_.get_iteration_triples_batch(list(current_batch_entities_set['source']))

            if args.use_2hop:
              current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                        list(current_batch_entities_set['source']), node_neighbors_2hop)
              if current_batch_2hop_indices.shape[0]==0:
                # current_batch_2hop_indices = np.empty((1,4))
                current_batch_2hop_indices = torch.tensor([],dtype=torch.long)
            else:
              current_batch_2hop_indices = torch.tensor([],dtype=torch.long)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
                if args.use_2hop:
                  current_batch_2hop_indices = Variable(
                      torch.LongTensor(current_batch_2hop_indices)).cuda()
            else: 
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))
                if args.use_2hop: 
                  current_batch_2hop_indices = Variable(
                      torch.LongTensor(current_batch_2hop_indices))

            entity_embed, relation_embed, mask = model_gat(
                Corpus_, torch.tensor(list(current_batch_entities_set['source'])), batch_train_adj_matrix, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed, model_gat)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())

            pbar.update(1)
            end_time_iter = time.time()
            print("Iteration-> {0}/{1}  , Iteration_time-> {2:.4f} , Iteration_loss {3:.4f}".format(
              iters, num_iters_per_epoch, end_time_iter - start_time_iter, loss.data.item()))


            if iters%display_every==0:
              num_samples = 1000
              random_ent_indices = np.random.randint(0,entity_embed.shape[0],num_samples)
              sampled_entities = np.array(entity_embed[random_ent_indices].detach().cpu())
              mean_vector = np.mean(sampled_entities,axis=-1)
              norm_entities = np.sqrt(np.sum(np.square(sampled_entities),axis=-1))
              norm_mean = np.sqrt(np.sum(np.square(sampled_entities),axis=-1))
              den = norm_mean*norm_entities
              num = np.dot(mean_vector,norm_entities.transpose())
              cosine_dist = num/den

              mean_cosine_dist = np.mean(cosine_dist)
              median_cosine_dist = np.median(cosine_dist)
              min_norm = np.min(norm_entities)
              max_norm = np.max(norm_entities)

              print('mean_cosine_dist: ',mean_cosine_dist)
              print('median_cosine_dist: ',median_cosine_dist)
              print('min_norm: ',min_norm)
              print('max_norm: ',max_norm)

            if CUDA:
              del train_indices
              del train_values
              del current_batch_2hop_indices
              del batch_train_adj_matrix
              del entity_embed
              del relation_embed
              del mask
              torch.cuda.empty_cache()

            iters += 1
            reduced_batch_size = None
            previous_end_idx = end_idx
          except Exception as e:
            print('== Exception: {}'.format(e))
            if 'out of memory' in str(e):
              if reduced_batch_size==None:
                reduced_batch_size = args.batch_size_gat//2
              else:
                reduced_batch_size = reduced_batch_size//2
              if reduced_batch_size==0:
                raise e
            else:
              raise e

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        pbar.close()

        save_model(model_gat, args.data, epoch,
                   args.output_folder)
        if epoch>0:
          os.remove(os.path.join(args.output_folder,'trained_{}.pth'.format(epoch-1)))

def train_entity_embeddings(args, word_embed_matrix, word_vocab, char_vocab):

    global initial_entity_emb_params
    global entity_embeddings

    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, initial_entity_emb_params)
    print("Only Entity embedding model")
  
    model_entity_embedding =  EntityEmbedding(\
                                  initial_entity_emb_params['entity_embed_dim_in'], \
                                  initial_entity_emb_params['hidden_dim_entity'], \
                                  initial_entity_emb_params['num_encoder_layers_entity'], \
                                  initial_entity_emb_params['is_bidirectional'], \
                                  initial_entity_emb_params['drop_out_rate'], \
                                  initial_entity_emb_params['hidden_dim_entity'], \
                                  initial_entity_emb_params['entity_embed_dim_out'], \
                                  initial_entity_emb_params['entity_conv_filter_size'],
                                  initial_entity_emb_params['word_vocab'], \
                                  initial_entity_emb_params['word_embed_dim'], \
                                  initial_entity_emb_params['char_embed_dim'], \
                                  initial_entity_emb_params['word_embed_matrix'], \
                                  initial_entity_emb_params['char_feature_size'], \
                                  initial_entity_emb_params['conv_filter_size'], \
                                  initial_entity_emb_params['max_word_len_entity'], \
                                  initial_entity_emb_params['char_vocab'])    
    if CUDA:
        model_gat.cuda()
        model_entity_embedding.cuda()

    if CUDA:
      model_gat.load_state_dict(torch.load(
        os.path.join(args.output_folder,'trained_{}.pth'.format(0))))
    else:
      model_gat.load_state_dict(torch.load(
        os.path.join(args.output_folder,'trained_{}.pth'.format(0)),map_location=torch.device('cpu')))
    entity_embed_gat = model_gat.entity_embeddings


    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)     

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_ent_emb))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_entity_embedding.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        num_emb_iters_per_epoch = int(np.ceil(
                        len(Corpus_.unique_entities_train) / args.entities_per_batch))
        random_indices = np.arange(0,len(Corpus_.unique_entities_train))
        np.random.shuffle(random_indices)
        random_unique_entities_train = []
        for idx in range(len(random_indices)):
          random_unique_entities_train.append(Corpus_.unique_entities_train[random_indices[idx]])
        # for iters in tqdm(range(num_emb_iters_per_epoch)):

        start_idx = end_idx = previous_end_idx = 0
        reduced_batch_size = None
        pbar = tqdm(total = num_emb_iters_per_epoch)
        iters = 0
        while True:
          if end_idx>=len(random_unique_entities_train):
            break
          start_time_iter = time.time()

          try:

            iters += 1
            start_idx = end_idx
            if reduced_batch_size:
              end_idx = min(len(random_unique_entities_train), start_idx + reduced_batch_size)
            else:
              end_idx = min(len(random_unique_entities_train), start_idx + args.entities_per_batch)

            current_batch_entities = random_unique_entities_train[start_idx:end_idx]

            train_indices, train_values = Corpus_.get_iteration_entities_batch(current_batch_entities,args.valid_invalid_ratio_entity_embed)

            batch_entities_ctx_data = Corpus_.get_batch_entities_ctx_data(train_indices[:len(current_batch_entities)],args.conv_filter_size,args.max_word_len,triple=False)

            ctx_words = torch.from_numpy(batch_entities_ctx_data['ctx_words_list'].astype('long'))
            ctx_char_seq = torch.from_numpy(batch_entities_ctx_data['ctx_char_seq'].astype('long'))
            ctx_mask = torch.from_numpy(batch_entities_ctx_data['ctx_mask'].astype('bool'))

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
                ctx_words = Variable(ctx_words).cuda() 
                ctx_char_seq = Variable(ctx_char_seq).cuda()
                ctx_mask = Variable(ctx_mask).cuda()
            else: 
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))
                ctx_words = Variable(ctx_words) 
                ctx_char_seq = Variable(ctx_char_seq)
                ctx_mask = Variable(ctx_mask)


            entity_embeddings_batch = model_entity_embedding(ctx_words, ctx_char_seq, ctx_mask)

            optimizer.zero_grad()

            loss = batch_entity_embedding_loss(
                train_indices, train_values, entity_embed_gat, entity_embeddings_batch)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())

            
            reduced_batch_size = None
            previous_end_idx = end_idx
            pbar.update(1)
            end_time_iter = time.time()
            print("Iteration-> {0}/{1}  , Iteration_time-> {2:.4f} , Iteration_loss {3:.4f}".format(
              iters, num_emb_iters_per_epoch, end_time_iter - start_time_iter, loss.data.item()))

          except RuntimeError as e:
            raise e


        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_entity_embedding, args.data, 0,
                   args.output_folder+'entity_embeddings/')


def train_conv(args):

    # Creating convolution model here.
    ####################################

    global initial_entity_emb_params

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, initial_entity_emb_params)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    # model_gat.load_state_dict(torch.load(
    #     '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)))
    model_gat.load_state_dict(torch.load(
        os.path.join(args.output_folder,'trained_{}.pth'.format(199))))
    # if os.path.exists('{0}trained_{1}.pth'.format(args.output_folder + "conv/", 0)):
    #   model_conv.load_state_dict(torch.load(
    #       os.path.join(args.output_folder + "conv/",'trained_{}.pth'.format(0))))
    # else:
    #   model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    #   model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
    
    # import pdb; pdb.set_trace()
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
    
    model_conv.final_entity_embeddings.requires_grad = False
    model_conv.final_relation_embeddings.requires_grad = False

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))



    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)

        random.shuffle(Corpus_.train_triples)

        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in tqdm(range(num_iters_per_epoch)):
            # print(model_conv.final_entity_embeddings[0][:10],model_conv.final_entity_embeddings[50][:10],model_conv.final_entity_embeddings[100][:10])
            # print(model_conv.final_relation_embeddings[0][:10],model_conv.final_relation_embeddings[50][:10],model_conv.final_relation_embeddings[100][:10])
            start_time_iter = time.time()
            # train_indices, train_values = Corpus_.get_iteration_batch(iters)
            train_indices, train_values = Corpus_.get_iteration_batch(0)
            # print(train_indices.tolist())
            # print(train_indices[:3],train_indices[64:67],train_indices[128:131],train_indices[192:198])
            # print(train_values[:3],train_values[64:67],train_values[128:131],train_values[192:198])


            # # import pdb; pdb.set_trace()
            # sampled_entities = np.concatenate((np.array(model_conv.final_entity_embeddings[train_indices[:,0]].detach().cpu()),np.array(model_conv.final_entity_embeddings[train_indices[:,2]].detach().cpu())),axis=0)
            # mean_vector = np.mean(sampled_entities,axis=-1)
            # norm_entities = np.sqrt(np.sum(np.square(sampled_entities),axis=-1))
            # norm_mean = np.sqrt(np.sum(np.square(sampled_entities),axis=-1))
            # den = norm_mean*norm_entities
            # num = np.dot(mean_vector,norm_entities.transpose())
            # cosine_dist = num/den
            # mean_cosine_dist = np.mean(cosine_dist)
            # median_cosine_dist = np.median(cosine_dist)
            # min_norm = np.min(norm_entities)
            # max_norm = np.max(norm_entities)
            # print('mean_cosine_dist: ',mean_cosine_dist)
            # print('median_cosine_dist: ',median_cosine_dist)
            # print('min_norm: ',min_norm)
            # print('max_norm: ',max_norm)


            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices, model_gat)

            optimizer.zero_grad()

            # valid_preds = preds[:args.batch_size_conv]
            # valid_preds = valid_preds.repeat(int(args.valid_invalid_ratio_conv)*2, 1)
            # valid_values = torch.ones(valid_preds.shape)
            # if CUDA:
            #   valid_values = valid_values.cuda()
            # preds = torch.cat((preds,valid_preds),dim=0)
            # train_values = torch.cat((train_values,valid_values),dim=0)

            # loss = margin_loss(preds.view(-1), train_values.view(-1))
            train_values = train_values.view(-1)
            train_values = (train_values+1)/2
            train_values = train_values.float()
            preds = preds.view(-1)
            # import pdb; pdb.set_trace()
            print(preds)
            print(train_values)
            weights = train_values + (1-train_values)*1/(args.valid_invalid_ratio_conv*2)
            loss = bce_loss(preds,train_values.float(),weight=weights)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
            # break
        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        # save_model(model_conv, args.data, epoch,
        #            args.output_folder + "conv/")
        save_model(model_conv, args.data, 0,
                   args.output_folder + "conv/")


def evaluate_conv(args, unique_entities_train, unique_entities_test):

    global initial_entity_emb_params
    global entity_embeddings

    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, initial_entity_emb_params)
    model_gat.load_state_dict(torch.load(
        '{0}trained_{1}.pth'.format(args.output_folder, 0)))

    model_entity_embedding = EntityEmbedding(                                  initial_entity_emb_params['entity_embed_dim_in'], \
                                  initial_entity_emb_params['hidden_dim_entity'], \
                                  initial_entity_emb_params['num_encoder_layers_entity'], \
                                  initial_entity_emb_params['is_bidirectional'], \
                                  initial_entity_emb_params['drop_out_rate'], \
                                  initial_entity_emb_params['hidden_dim_entity'], \
                                  initial_entity_emb_params['entity_embed_dim_out'], \
                                  initial_entity_emb_params['entity_conv_filter_size'],
                                  initial_entity_emb_params['word_vocab'], \
                                  initial_entity_emb_params['word_embed_dim'], \
                                  initial_entity_emb_params['char_embed_dim'], \
                                  initial_entity_emb_params['word_embed_matrix'], \
                                  initial_entity_emb_params['char_feature_size'], \
                                  initial_entity_emb_params['conv_filter_size'], \
                                  initial_entity_emb_params['max_word_len_entity'], \
                                  initial_entity_emb_params['char_vocab'])
    model_entity_embedding.load_state_dict(torch.load(
        '{0}{1}/trained_{2}.pth'.format(args.output_folder,'entity_embeddings', 0)))

    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, 0)))

    if CUDA:
      model_conv.cuda()
      model_gat.cuda()
      model_entity_embedding.cuda()
    model_conv.eval()
    model_gat.eval()
    model_entity_embedding.eval()
    with torch.no_grad():
        Corpus_.get_validation_cnfmat(args, model_gat, model_entity_embedding, model_conv, unique_entities_train, unique_entities_test, reuse=False, gat_only=False)


def save_entity_relation_final_embeddings():
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, initial_entity_emb_params)
    model_gat.load_state_dict(torch.load(
        '{0}trained_{1}.pth'.format(args.output_folder, 0)))

    final_entity_embeddings = model_gat.final_entity_embeddings
    final_relation_embeddings = model_gat.final_relation_embeddings

    save_embed(final_entity_embeddings,os.path.join(args.output_folder,'final_entity_embeddings.json'))
    save_embed(final_relation_embeddings,os.path.join(args.output_folder,'final_relation_embeddings.json'))

    np.save(os.path.join(args.output_folder,'W_ent2rel.json'),np.array(model_gat.W_ent2rel.cpu().detach()))

train_gat(args, word_embed_matrix, word_vocab, char_vocab)
train_entity_embeddings(args, word_embed_matrix, word_vocab, char_vocab)
train_conv(args)

evaluate_conv(args, Corpus_.unique_entities_train, Corpus_.unique_entities_test)
evaluate_conv(args, Corpus_.unique_entities_train, [])
save_entity_relation_final_embeddings()
