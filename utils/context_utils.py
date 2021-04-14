from nltk import word_tokenize
from nltk import sent_tokenize
from collections import OrderedDict
import numpy as np
import torch
from utils import embedding_utils
import json
import datetime

all_zeroes = "ALL_ZERO"
unknown = "_UNKNOWN"
MAX_CTX_SENT_LEN = 32
MAX_NUM_CONTEXTS = 32
CUDA = torch.cuda.is_available()

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

def make_char_vocab(data):
    char_vocab = OrderedDict()
    char_vocab['<PAD>'] = 0
    char_vocab['<UNK>'] = 1
    char_idx = 2
    for d in data:
        for word in d['tokens']:
            for c in word:
                if c not in char_vocab:
                    char_vocab[c] = char_idx
                    char_idx += 1

    return char_vocab

def get_unique_entities(data):
    unique_entities = set()
    unique_entities.add(-1)
    
    entity_surface_forms = [['ALL_ZERO']]

    for d in data:
      for entity in d['vertexSet']:
        unique_entities.add(entity['kbID'])
        entity_surface_forms.append([d['tokens'][tp] for tp in entity['tokenpositions']])

    return list(unique_entities), entity_surface_forms

def get_batch_unique_entities(entity_indices, entity_surface_forms):
    unique_entities = {-1: ['ALL_ZERO']}
    unique_entities_count = {-1: 0}
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        for k in range(entity_indices.shape[2]):
          unique_entities[entity_indices[i,j,k]] = entity_surface_forms[i,j,k]
          unique_entities_count[entity_indices[i,j,k]] = unique_entities_count.get(entity_indices[i,j,k],0) + 1

    unique_entities_set = []
    unique_entities_surface_forms = []
    ent_occurrence = []
    ent_index = []
    for k, v in unique_entities.items():
      unique_entities_set.append(k)
      unique_entities_surface_forms.append(v)
      ent_occurrence.append(unique_entities_count[k])
      ent_index.append(k)
    max_entity_pos = np.argmax(ent_occurrence)
    max_occurred_ent = ent_index[max_entity_pos]
    max_occurred_ent_pos = unique_entities_set.index(max_occurred_ent)

    return np.array(unique_entities_set), unique_entities_surface_forms, max_occurred_ent_pos

def init_random(elements_to_embed, embedding_size, add_all_zeroes=False, add_unknown=False):
    """
    Initialize a random embedding matrix for a collection of elements. Elements are sorted in order to ensure
    the same mapping from indices to elements each time.

    :param elements_to_embed: collection of elements to construct the embedding matrix for
    :param embedding_size: size of the embedding
    :param add_all_zeroes: add a all_zero embedding at index 0
    :param add_unknown: add unknown embedding at the last index
    :return: an embedding matrix and a dictionary mapping elements to rows in the matrix
    """
    elements_to_embed = sorted(elements_to_embed)
    element2idx = {all_zeroes: -1} if add_all_zeroes else {}
    element2idx.update({el: idx for idx, el in enumerate(elements_to_embed)})
    if add_unknown:
        element2idx[unknown] = len(element2idx)

    embeddings = np.random.random((len(element2idx),embedding_size)).astype('f')
    if add_all_zeroes:
        embeddings[0] = np.zeros([embedding_size])

    return embeddings, element2idx

def get_vertex2Qid(data):
    vertex_to_qid = {}
    for v in data['vertexSet']:
      vertex_to_qid[str(v['tokenpositions'])] = v['kbID']

    return vertex_to_qid

def get_entityIdx_from_tokens(tokens, entity_tokens, vertex_to_qid, entity2idx):
    entity_Qid = vertex_to_qid[str(entity_tokens)]
    return entity2idx.get(entity_Qid,entity2idx[unknown])

def get_entity_context_data(entityIdx, surface_form_token, context_data, idx2entity, data='wikidata'):
    if data=='wikidata':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': '',
        'instances': [],
        'aliases': []
      })
      entity_context = [[i for i in surface_form_token]]
      if context['desc'] != '':
        entity_context.extend([word_tokenize(context['desc'])])
      instanceof_arr = [word_tokenize('instance of {}'.format(i['label'])) for i in context['instances'][:15]]
      alias_arr = [word_tokenize('also known as {}'.format(i)) for i in context['aliases'][:15]]
      entity_context.extend(instanceof_arr)
      entity_context.extend(alias_arr)
    elif data=='nyt':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': [],
        'en_instances': [],
        'alias': []
      })
      entity_context = [[i for i in surface_form_token]]
      if len(context['desc'])>0:
        entity_context.extend([word_tokenize(sent) for sent in sent_tokenize(context['desc'][0])[:1]])
      instanceof_arr = [word_tokenize('instance of {}'.format(i)) for i in context['en_instances'][:15]]
      alias_arr = [word_tokenize('also known as {}'.format(i)) for i in context['alias'][:15]]
      entity_context.extend(instanceof_arr)
      entity_context.extend(alias_arr)
    else:
      raise("data must be one of: {}, got: {}".format(['wikidata','nyt'],data))

    return entity_context

def get_entity_context_data_ablation(entityIdx, surface_form_token, context_data, idx2entity, context_to_use, data='wikidata'):
    if data=='wikidata':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': '',
        'instances': [],
        'aliases': []
      })
      if context_to_use=='surface_form':
        entity_context = [[i for i in surface_form_token]]
      elif context_to_use=='desc':    
        if context['desc'] != '':
          entity_context = [word_tokenize(context['desc'])]
        else:
          entity_context = [[i for i in surface_form_token]]
      elif context_to_use=='instanceof':
        instanceof_arr = [word_tokenize('instance of {}'.format(i['label'])) for i in context['instances'][:15]]
        if len(instanceof_arr)==0:
          instanceof_arr = [word_tokenize('instance of {}'.format(' '.join(surface_form_token).strip()))]
        entity_context = instanceof_arr
      elif context_to_use=='alias':
        alias_arr = [word_tokenize('also known as {}'.format(i)) for i in context['aliases'][:15]]
        if len(alias_arr)==0:
          alias_arr = [word_tokenize('also known as {}'.format(' '.join(surface_form_token).strip()))]
        entity_context = alias_arr
      else:
        raise("Context should be one of [surface_form, desc, instanceof, alias]")
    elif data=='nyt':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': [],
        'instances': [],
        'aliases': []
      })
      if context_to_use=='surface_form':
        entity_context = [[i for i in surface_form_token]]
      elif context_to_use=='desc':    
        if len(context['desc']) > 0:
          entity_context = [word_tokenize(sent) for sent in sent_tokenize(context['desc'][0])[:1]]
        else:
          entity_context = [[i for i in surface_form_token]]
      elif context_to_use=='instanceof':
        instanceof_arr = [word_tokenize('instance of {}'.format(i)) for i in context['en_instances'][:15]]
        if len(instanceof_arr)==0:
          instanceof_arr = [word_tokenize('instance of {}'.format(' '.join(surface_form_token).strip()))]
        entity_context = instanceof_arr
      elif context_to_use=='alias':
        alias_arr = [word_tokenize('also known as {}'.format(i)) for i in context['alias'][:15]]
        if len(alias_arr)==0:
          alias_arr = [word_tokenize('also known as {}'.format(' '.join(surface_form_token).strip()))]
        entity_context = alias_arr
      else:
        raise("Context should be one of [surface_form, desc, instanceof, alias]")
    else:
      raise("data must be one of: {}, got: {}".format(['wikidata','nyt'],data))
      
    return entity_context

def get_entity_context_data_ablation_incremental(entityIdx, surface_form_token, context_data, idx2entity, context_to_use, data='wikidata'):
    if data=='wikidata':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': '',
        'instances': [],
        'aliases': []
      })
      entity_context = []
      if 'surface_form' in context_to_use:
        entity_context.extend([[i for i in surface_form_token]])
      if 'desc' in context_to_use:    
        if context['desc'] != '':
          entity_context.extend([word_tokenize(context['desc'])])
      elif 'instanceof' in context_to_use:
        entity_context.extend( [word_tokenize('instance of {}'.format(i['label'])) for i in context['instances'][:15]])
      elif 'alias' in context_to_use:
        alias_arr.extend([word_tokenize('also known as {}'.format(i)) for i in context['aliases'][:15]])
      if len(entity_context)==0:
        entity_context = [[i for i in surface_form_token]]
    elif data=='nyt':
      entity_qid = idx2entity[entityIdx]
      context = context_data.get(entity_qid, {
        'desc': [],
        'instances': [],
        'aliases': []
      })
      entity_context = []
      if 'surface_form' in context_to_use:
        entity_context.extend([[i for i in surface_form_token]])
      if 'desc' in context_to_use:    
        if len(context['desc']) > 0:
          entity_context.extend([word_tokenize(sent) for sent in sent_tokenize(context['desc'][0])[:1]])
      elif 'instanceof' in context_to_use:
        entity_context.extend( [word_tokenize('instance of {}'.format(i)) for i in context['en_instances'][:15]])
      elif 'alias' in context_to_use:
        alias_arr.extend([word_tokenize('also known as {}'.format(i)) for i in context['alias'][:15]])
      if len(entity_context)==0:
        entity_context = [[i for i in surface_form_token]]
    else:
      raise("data must be one of: {}, got: {}".format(['wikidata','nyt'],data))
      
    return entity_context

def get_word_indices(entity_context, max_sent_len, max_num_contexts, word2idx):
    sentences_matrix = np.zeros((max_num_contexts,max_sent_len))
    for index, sent in enumerate(entity_context):
        token_ids = embedding_utils.get_idx_sequence(sent, word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
    return sentences_matrix

def get_char_indices(entity_context, max_sent_len, max_num_contexts, max_char_len, char2idx, conv_filter_size):
    sentences_matrix = np.ones( (max_num_contexts,
      conv_filter_size-1 + max_sent_len*(max_char_len+conv_filter_size-1) ) )
    sentences_matrix = sentences_matrix*char2idx['<PAD>']

    entity_context = entity_context[:max_num_contexts]
    for index_sent, sent in enumerate(entity_context):
      words = sent[:max_sent_len]
      cur_idx = conv_filter_size - 1

      for index_word, word in enumerate(words):
          for index_char, c in enumerate(word[0:min(len(word), max_char_len)]): 
              if char2idx.get(c,None) is not None:
                  sentences_matrix[index_sent, cur_idx+index_char] = char2idx[c]
              else:
                  sentences_matrix[index_sent, cur_idx+index_char] = char2idx['<UNK>']
          cur_idx += max_char_len + conv_filter_size-1

    return sentences_matrix

def get_mask(entity_context, max_sent_len, max_num_contexts):
    mask_matrix = np.ones( (max_num_contexts) )

    entity_context = entity_context[:max_num_contexts]
    mask_matrix[:len(entity_context)] = 0

    return mask_matrix

def get_entity_location_unique_entities(unique_entities, entity_indices):
    entity_pos = np.zeros(entity_indices.shape)
    unique_entities = unique_entities.tolist()
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        for k in range(entity_indices.shape[2]):
          entity_pos[i,j,k] = unique_entities.index(entity_indices[i,j,k])

    return entity_pos

def get_max_len(sample_batch, MAX_LEN=MAX_CTX_SENT_LEN):
    src_max_len = len(sample_batch[0][0])
    for idx in range(0, len(sample_batch)):
        for idx2 in range(0, len(sample_batch[idx])):
            if len(sample_batch[idx][idx2]) > src_max_len:
                src_max_len = len(sample_batch[idx][idx2])


    return min(MAX_LEN,src_max_len)

def get_max_ctx_len(sample_batch, MAX_LEN=MAX_NUM_CONTEXTS):
    src_max_ctx_len = len(sample_batch[0])
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx]) > src_max_ctx_len:
            src_max_ctx_len = len(sample_batch[idx])

    return min(MAX_LEN,src_max_ctx_len)

def get_context_indices_ablation_incremental(entity_indices, surface_form_tokens, context_data, idx2entity, word2idx, char2idx, conv_filter_size, context_to_use=['surface_form'], max_sent_len=MAX_CTX_SENT_LEN, max_num_contexts=MAX_NUM_CONTEXTS, max_char_len=10, data='wikidata'):

    entity_contexts = []
    for idx, entityIdx in enumerate(entity_indices):
        entity_contexts.append(get_entity_context_data_ablation_incremental(entityIdx, surface_form_tokens[idx], context_data, idx2entity, context_to_use, data=data))

    max_sent_len = get_max_len(entity_contexts, MAX_LEN=max_sent_len)
    max_num_contexts = get_max_ctx_len(entity_contexts, MAX_LEN=max_num_contexts)

    ctx_words_list = []
    ctx_char_seq = []
    ctx_mask = []
    for entity_context in entity_contexts:
      ctx_words_list.append(get_word_indices(entity_context, max_sent_len, max_num_contexts, word2idx))
      ctx_char_seq.append(get_char_indices(entity_context, max_sent_len, max_num_contexts, max_char_len, char2idx, conv_filter_size))
      ctx_mask.append(get_mask(entity_context, max_sent_len, max_num_contexts))
    ctx_words_list = np.array(ctx_words_list)
    ctx_char_seq = np.array(ctx_char_seq)
    ctx_mask = np.array(ctx_mask,dtype=bool)

    return ctx_words_list, ctx_char_seq, ctx_mask

def get_context_indices_ablation(entity_indices, surface_form_tokens, context_data, idx2entity, word2idx, char2idx, conv_filter_size, context_to_use='surface_form', max_sent_len=MAX_CTX_SENT_LEN, max_num_contexts=MAX_NUM_CONTEXTS, max_char_len=10, data='wikidata'):

    entity_contexts = []
    for idx, entityIdx in enumerate(entity_indices):
        entity_contexts.append(get_entity_context_data_ablation(entityIdx, surface_form_tokens[idx], context_data, idx2entity, context_to_use, data=data))

    max_sent_len = get_max_len(entity_contexts, MAX_LEN=max_sent_len)
    max_num_contexts = get_max_ctx_len(entity_contexts, MAX_LEN=max_num_contexts)

    ctx_words_list = []
    ctx_char_seq = []
    ctx_mask = []
    for entity_context in entity_contexts:
      ctx_words_list.append(get_word_indices(entity_context, max_sent_len, max_num_contexts, word2idx))
      ctx_char_seq.append(get_char_indices(entity_context, max_sent_len, max_num_contexts, max_char_len, char2idx, conv_filter_size))
      ctx_mask.append(get_mask(entity_context, max_sent_len, max_num_contexts))
    ctx_words_list = np.array(ctx_words_list)
    ctx_char_seq = np.array(ctx_char_seq)
    ctx_mask = np.array(ctx_mask,dtype=bool)

    return ctx_words_list, ctx_char_seq, ctx_mask

def get_context_indices(entity_indices, surface_form_tokens, context_data, idx2entity, word2idx, char2idx, conv_filter_size, max_sent_len=MAX_CTX_SENT_LEN, max_num_contexts=MAX_NUM_CONTEXTS, max_char_len=10, data='wikidata'):

    entity_contexts = []
    for idx, entityIdx in enumerate(entity_indices):
        entity_contexts.append(get_entity_context_data(entityIdx, surface_form_tokens[idx], context_data, idx2entity, data=data))

    max_sent_len = get_max_len(entity_contexts, MAX_LEN=max_sent_len)
    max_num_contexts = get_max_ctx_len(entity_contexts, MAX_LEN=max_num_contexts)

    ctx_words_list = []
    ctx_char_seq = []
    ctx_mask = []
    for entity_context in entity_contexts:
      ctx_words_list.append(get_word_indices(entity_context, max_sent_len, max_num_contexts, word2idx))
      ctx_char_seq.append(get_char_indices(entity_context, max_sent_len, max_num_contexts, max_char_len, char2idx, conv_filter_size))
      ctx_mask.append(get_mask(entity_context, max_sent_len, max_num_contexts))
    ctx_words_list = np.array(ctx_words_list)
    ctx_char_seq = np.array(ctx_char_seq)
    ctx_mask = np.array(ctx_mask,dtype=bool)

    return ctx_words_list, ctx_char_seq, ctx_mask

def make_start_entity_embeddings(entity_embeddings, entity_pos_indices, unique_entities, embedding_dim, max_occurred_entity_in_batch_pos, start_embedding_template, max_num_nodes=9):
    # import time
    # s1=time.time()
    # import pdb; pdb.set_trace()
    if torch.cuda.is_available():
      vector = torch.ones((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim*max_num_nodes,1)).cuda()
    else:
      vector = torch.ones((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim*max_num_nodes,1))
    
    max_occurring_ent_emb_reshaped = entity_embeddings[max_occurred_entity_in_batch_pos].unsqueeze(-1).repeat([2*max_num_nodes,1])
    vector = vector*max_occurring_ent_emb_reshaped

    # s2=time.time()
    # print('t3',s2-s1)
    # import pdb; pdb.set_trace()
    for idx in range(entity_pos_indices.shape[0]):
      # for idx2 in range(entity_pos_indices.shape[1]):
      #     vector[idx,idx2,2*(idx2//max_num_nodes)*embedding_dim:(2*(idx2//max_num_nodes)+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,0]]
      #     vector[idx,idx2,(2*(idx2//max_num_nodes)+1 + 2*(idx2%max_num_nodes)+1)*embedding_dim:2*(idx2%max_num_nodes+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,1]]
      idx2 = 0
      count = 0
      for i in range(max_num_nodes):
          for j in range(max_num_nodes):
            if i!=j:
                  for k in range(max_num_nodes):
                      if(k == i):
                          if entity_pos_indices[idx,idx2,0]!=max_occurred_entity_in_batch_pos:
                            vector[idx,idx2,2*i*embedding_dim:(2*i+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,0]]
                          count += 1
                      elif (k == j):
                          if entity_pos_indices[idx,idx2,1]!=max_occurred_entity_in_batch_pos:
                            vector[idx,idx2,(2*j+1)*embedding_dim:2*(j+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,1]]
                          count += 1
                      if count == 2:
                        count = 0
                        idx2 += 1
      
    vector = vector*start_embedding_template

    return vector      

def get_entity_embeddings(entity_embeddings, entity_pos_indices, unique_entities, embedding_dim, max_occurred_entity_in_batch_pos, start_embedding_template, max_num_nodes=9):
    # import time
    # s1=time.time()
    # import pdb; pdb.set_trace()
    if torch.cuda.is_available():
      vector = torch.zeros((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim)).cuda()
    else:
      vector = torch.zeros((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim))
    
    for idx1 in range(entity_pos_indices.shape[0]):
      for idx2 in range(entity_pos_indices.shape[0]):
        vector[idx1,idx2,:embedding_dim] = entity_embeddings[entity_pos_indices[idx1,idx2,0]]
        vector[idx1,idx2,embedding_dim:] = entity_embeddings[entity_pos_indices[idx1,idx2,1]]
      
    return vector  

def get_gat_entity_embeddings(entity_indices, entity2id, idx2entity, gat_entity2id, gat_embeddings):
    gat_entity_embedding_dim = len(gat_embeddings["0"])
    gat_entity_embeddings = np.zeros([entity_indices.shape[0],entity_indices.shape[1], entity_indices.shape[2]*gat_entity_embedding_dim])
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        for k in range(entity_indices.shape[2]):
          if entity_indices[i,j,k] not in [entity2id[all_zeroes],entity2id[unknown]]:
            idx = str(gat_entity2id[idx2entity[entity_indices[i,j,k]]])
            gat_entity_embeddings[i,j,k*gat_entity_embedding_dim:(k+1)*gat_entity_embedding_dim] = gat_embeddings[idx]
    return gat_entity_embeddings

def get_selected_gat_entity_embeddings(entity_indices, entity2id, idx2entity, gat_entity2id, gat_embeddings):
    gat_entity_embedding_dim = len(gat_embeddings["0"])
    gat_entity_embeddings = np.zeros([entity_indices.shape[0],entity_indices.shape[1], entity_indices.shape[2]*gat_entity_embedding_dim])
    nonzero_gat_entity_embeddings = []
    nonzero_entity_pos = []
    count = 0
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        cur_gat_embeddings = []
        for k in range(entity_indices.shape[2]):
          # if entity_indices[i,j,k] not in [entity2id[all_zeroes],entity2id[unknown]]:
          if entity_indices[i,j,k] not in [entity2id[all_zeroes]]:
            idx = str(gat_entity2id.get(idx2entity[entity_indices[i,j,k]], entity2id[unknown]))
            gat_entity_embeddings[i,j,k*gat_entity_embedding_dim:(k+1)*gat_entity_embedding_dim] = gat_embeddings[idx]
            cur_gat_embeddings.extend(gat_embeddings[idx])
            
        if len(cur_gat_embeddings)>0:
          nonzero_gat_entity_embeddings.append(cur_gat_embeddings)
          nonzero_entity_pos.append(count)
        count += 1
    return gat_entity_embeddings, np.array(nonzero_gat_entity_embeddings), nonzero_entity_pos

def get_selected_gat_entity_embeddings_v2(entity_indices, entity2id, idx2entity, gat_entity2id, gat_embeddings):
    gat_entity_embedding_dim = len(gat_embeddings[0])
    if CUDA:
      gat_entity_embeddings = torch.zeros([entity_indices.shape[0],entity_indices.shape[1], entity_indices.shape[2]*gat_entity_embedding_dim]).cuda()
      nonzero_gat_entity_embeddings = torch.empty((0,2*gat_entity_embedding_dim)).cuda()
    else:
      gat_entity_embeddings = torch.zeros([entity_indices.shape[0],entity_indices.shape[1], entity_indices.shape[2]*gat_entity_embedding_dim])
      nonzero_gat_entity_embeddings = torch.empty((0,2*gat_entity_embedding_dim))
    nonzero_entity_pos = []
    count = 0
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        if CUDA:
          cur_gat_embeddings = torch.empty((0)).cuda()
        else:
          cur_gat_embeddings = torch.empty((0))
        for k in range(entity_indices.shape[2]):
          # if entity_indices[i,j,k] not in [entity2id[all_zeroes],entity2id[unknown]]:
          if entity_indices[i,j,k] not in [entity2id[all_zeroes]]:
            # idx = str(gat_entity2id.get(idx2entity[entity_indices[i,j,k]], entity2id[unknown]))
            gat_entity_embeddings[i,j,k*gat_entity_embedding_dim:(k+1)*gat_entity_embedding_dim] = gat_embeddings[entity_indices[i,j,k]]
            cur_gat_embeddings = torch.cat((cur_gat_embeddings,gat_embeddings[entity_indices[i,j,k]]), dim=-1)
            
        if len(cur_gat_embeddings)>0:
          nonzero_gat_entity_embeddings = torch.cat((nonzero_gat_entity_embeddings,cur_gat_embeddings.unsqueeze(0)), dim=0)
          nonzero_entity_pos.append(count)
        count += 1
    return gat_entity_embeddings, nonzero_gat_entity_embeddings, nonzero_entity_pos


# def get_Went2rel(relations, gat_relation2idx, W_ent2rel_all_rels):
#     non_zero_indices = relations!=all_zeroes
#     # W_ent2rel = np.zeros((len(relations),W_ent2rel_all_rels.shape[1],W_ent2rel_all_rels.shape[2]),dtype=np.float32)
#     W_ent2rel = np.zeros((non_zero_indices,W_ent2rel_all_rels.shape[1],W_ent2rel_all_rels.shape[2]),dtype=np.float32)
#     for i, rel in enumerate(relations):
#       if rel==all_zeroes:
#         continue
#       gat_idx = gat_relation2idx.get(rel,None)
#       if gat_idx is not None:
#         W_ent2rel[i,:,:] = W_ent2rel_all_rels[gat_idx]

#     return W_ent2rel, non_zero_indices

def get_Went2rel(relations, gat_relation2idx, W_ent2rel_all_rels):
    non_zero_indices = np.array(relations)!=all_zeroes
    # print(non_zero_indices)
    # print(np.sum(non_zero_indices))
    W_ent2rel = np.zeros((np.sum(non_zero_indices),W_ent2rel_all_rels.shape[1],W_ent2rel_all_rels.shape[2]),dtype=np.float32)
    idx = 0
    for i, consider_rel_index in enumerate(non_zero_indices):
      if not consider_rel_index:
        continue
      rel = relations[i]
      gat_idx = gat_relation2idx.get(rel,None)
      if gat_idx is not None:
        W_ent2rel[idx,:,:] = W_ent2rel_all_rels[int(gat_idx)]
      idx += 1

    return W_ent2rel, non_zero_indices

def do_negative_sampling_and_get_relation_indices_and_probs(relations, idx2property, property2idx, negative_sampling, entity_indices, entity2idx, ignore_index=0, rels_to_ignore=[]):
    relation_indices = [property2idx.get(i,property2idx[unknown]) for i in relations]
    triple_probs = [1 for _ in range(len(relation_indices))]
    num_original_rels = len(relation_indices)
    num_rels = len(property2idx)
    entity_indices_reshaped = entity_indices.reshape([-1,2])

    relation_indices_all = np.zeros((num_original_rels*(negative_sampling+1)))
    relation_indices_all[:num_original_rels] = relation_indices
    triple_probs_all = np.zeros((num_original_rels*(negative_sampling+1)),dtype=np.float32)
    triple_probs_all[:num_original_rels] = triple_probs

    for idx in range(num_original_rels):
      if negative_sampling<num_rels-1:
        cur_negative_sampled_rels = np.random.randint(0,num_rels,negative_sampling)
        for idx2 in range(negative_sampling):
          # check: false negative samples in entire dataset?
          while cur_negative_sampled_rels[idx2]==relation_indices[idx] or idx2property[cur_negative_sampled_rels[idx2]] in [all_zeroes,unknown]:
            cur_negative_sampled_rels[idx2] = np.random.randint(0,num_rels)
      else:
        cur_negative_sampled_rels = [i for i in range(num_rels) if i!=relation_indices[idx]]

      for idx2 in range(negative_sampling):
        if entity_indices_reshaped[idx][0]==entity2idx[all_zeroes] or entity_indices_reshaped[idx][1]==entity2idx[all_zeroes]:
          relation_indices_all[(idx2+1)*num_original_rels+idx] = property2idx[all_zeroes]
          triple_probs_all[(idx2+1)*num_original_rels+idx] = 1
        else:
          relation_indices_all[(idx2+1)*num_original_rels+idx] = cur_negative_sampled_rels[idx2]
          triple_probs_all[(idx2+1)*num_original_rels+idx] = 0

    batch_relations = [idx2property[i] for i in relation_indices_all]

    # indices_to_consider = [i for i, br in enumerate(batch_relations) if br not in rels_to_ignore]
    # triple_probs_all = triple_probs_all[indices_to_consider]
    # relation_indices_all = relation_indices_all[indices_to_consider]
    # batch_relations = [batch_relations[i] for i in indices_to_consider]
    # original_indices_to_consider = [i for i, br in enumerate(relations) if br not in rels_to_ignore]
    # num_original_rels = len(original_indices_to_consider)
    # relations_to_consider = relations[original_indices_to_consider]

    loss_weights = np.ones(triple_probs_all.shape)
    # loss_weights = loss_weights*(1-(np.array(relation_indices_all)==ignore_index))
    indices_to_consider = 1-(np.array(relation_indices_all)==ignore_index)
    for r in rels_to_ignore:
      r = property2idx[r]
      indices_to_consider = indices_to_consider*(1-(np.array(relation_indices_all)==r))
    loss_weights = loss_weights*indices_to_consider
    negative_samples_inverse_weight = np.sum(loss_weights[num_original_rels:])/np.sum(loss_weights[:num_original_rels])
    loss_weights = (triple_probs_all.copy() + (1-triple_probs_all.copy())*1.0/negative_samples_inverse_weight) * loss_weights

    # return batch_relations, relation_indices_all, triple_probs_all, loss_weights, relations_to_consider, indices_to_consider, original_indices_to_consider
    return batch_relations, relation_indices_all, triple_probs_all, loss_weights

def get_relspace_embeddings(gat_entity_embeddings, W_ent2rel, non_zero_indices, batch_size, relations, negative_sampling):
    CUDA = torch.cuda.is_available()
    if CUDA:
      final_embeddings = torch.zeros([batch_size,gat_entity_embeddings.shape[-1]]).cuda()
    else:
      final_embeddings = torch.zeros([batch_size,gat_entity_embeddings.shape[-1]])
    gat_entity_embeddings = gat_entity_embeddings.reshape([gat_entity_embeddings.shape[0]*gat_entity_embeddings.shape[1],gat_entity_embeddings.shape[2]])

    gat_emb_head = gat_entity_embeddings[:,:gat_entity_embeddings.shape[-1]//2]
    gat_emb_tail = gat_entity_embeddings[:,gat_entity_embeddings.shape[-1]//2:]

    gat_emb_head = gat_emb_head.repeat([negative_sampling+1,1])
    gat_emb_tail = gat_emb_tail.repeat([negative_sampling+1,1])

    try:
      gat_emb_head = gat_emb_head[non_zero_indices].unsqueeze(1)
      gat_emb_tail = gat_emb_tail[non_zero_indices].unsqueeze(1)
    except Exception as e:
      print(e)
      import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    entity_emb_gat_head = torch.tanh(torch.bmm(gat_emb_head,W_ent2rel)).squeeze()
    entity_emb_gat_tail = torch.tanh(torch.bmm(gat_emb_tail,W_ent2rel)).squeeze()
    gat_entity_embeddings_relspace = torch.cat([entity_emb_gat_head,entity_emb_gat_tail], dim=-1)

    final_embeddings[non_zero_indices] = gat_entity_embeddings_relspace

    return final_embeddings

