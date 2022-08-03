from utils import evaluation_utils, embedding_utils, context_utils
from semanticgraph import io
from parsing import legacy_sp_models as sp_models
from models import baselines
import numpy as np
from sacred import Experiment
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
import ast
from models.factory import get_model
import argparse

import torch.nn.functional as F
try:
    from functools import reduce
except:
    pass

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

CUDA = torch.cuda.is_available()

def to_np(x):
    return x.data.cpu().numpy()

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("-sf", "--save_folder",
                      default="./models/RECON/", help="folder to save the models")
    args.add_argument("-rf", "--result_folder",
                      default="./result/", help="folder to save the results")
    args.add_argument("-b", "--batch_size", type=int,
                      default=50, help="Batch Size")
    args.add_argument("-tf", "--test_file",
                      default="", help="file containing test data")


    args = args.parse_args()
    return args

args = parse_args()

def test():

    """ Main Configurations """
    model_name = "RECON"
    load_model = "RECON-{}.out" # you should choose the proper model to load
    # device_id = 0
    dataset_name = 'wikidata' # Options: wikidata, nyt; any new data will need to be added in the loader in semantic_graph.io


    data_folder = "./data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/"
    save_folder = "./models/RECON/"
    result_folder = "result/"

    model_params = "model_params.json"
    word_embeddings = "./data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/glove.6B.50d.txt"

    test_set = "semantic-graphs-filtered-held-out.02_06.json"
    
    gat_embedding_file = None
    gat_relation_embedding_file = None
    if "RECON" in model_name:
        context_data_file = "./data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entities_context.json"
    if "KGGAT" in model_name:
        gat_embedding_file = './models/GAT/WikipediaWikidataDistantSupervisionAnnotations/final_entity_embeddings.json'
        gat_entity2id_file = './data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entity2id.txt'
    if model_name=="RECON":
        gat_embedding_file = './models/GAT_sep_space/WikipediaWikidataDistantSupervisionAnnotations/final_entity_embeddings.json'
        gat_entity2id_file = './data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entity2id.txt'
        gat_relation_embedding_file = './models/GAT_sep_space/WikipediaWikidataDistantSupervisionAnnotations/final_relation_embeddings.json'
        gat_relation2id_file = './data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/relation2id.txt'
        w_ent2rel_all_rels_file = './models/GAT_sep_space/WikipediaWikidataDistantSupervisionAnnotations/W_ent2rel.json.npy'

    use_char_vocab = False

    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None

    with open(model_params) as f:
        model_params = json.load(f)
    global args
    save_folder = args.save_folder
    if args.test_file != '':
        test_set = args.test_file
    result_folder = args.result_folder
    model_params['batch_size'] = args.batch_size
    if not os.path.exists(result_folder):
      os.makedirs(result_folder)

    char_vocab_file = os.path.join(save_folder,"char_vocab.json")

    sp_models.set_max_edges(model_params['max_num_nodes']*(model_params['max_num_nodes']-1), model_params['max_num_nodes'])


    if context_data_file:
      with open(context_data_file, 'r') as f:
          context_data = json.load(f)
    if gat_embedding_file:
      with open(gat_embedding_file, 'r') as f:
          gat_embeddings = json.load(f)
      with open(gat_entity2id_file, 'r') as f:
          gat_entity2idx = {}
          data = f.read()
          lines = data.split('\n')
          for line in lines:
            line_arr = line.split(' ')
            if len(line_arr)==2:
              gat_entity2idx[line_arr[0].strip()] = line_arr[1].strip()
    if gat_relation_embedding_file:
      with open(gat_relation_embedding_file, 'r') as f:
          gat_relation_embeddings = json.load(f)
      W_ent2rel_all_rels = np.load(w_ent2rel_all_rels_file)
      with open(gat_relation2id_file, 'r') as f:
          gat_relation2idx = {}
          data = f.read()
          lines = data.split('\n')
          for line in lines:
            line_arr = line.split(' ')
            if len(line_arr)==2:
              gat_relation2idx[line_arr[0].strip()] = line_arr[1].strip()

    embeddings, word2idx = embedding_utils.load(word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")


    print("Reading the property index")
    with open(os.path.join(save_folder, model_name + ".property2idx")) as f:
        property2idx = ast.literal_eval(f.read())
    idx2property = { v:k for k,v in property2idx.items() }
    print("Reading the entity index")
    with open(os.path.join(save_folder, model_name + ".entity2idx")) as f:
        entity2idx = ast.literal_eval(f.read())
    idx2entity = { v:k for k,v in entity2idx.items() }
    context_data['ALL_ZERO'] = {
      'desc': '',
      'label': 'ALL_ZERO',
      'instances': [],
      'aliases': []
    }

    with open(char_vocab_file, 'r') as f:
        char_vocab = json.load(f)

    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices_and_entity_pair
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask   
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON_EAC":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON_EAC_KGGAT":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)


    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    if "RECON" not in model_name:
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out)
    elif model_name=="RECON_EAC":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab)
    elif model_name=="RECON_EAC_KGGAT":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab)
    elif model_name=="RECON":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab, gat_relation_embeddings, W_ent2rel_all_rels, idx2property, gat_relation2idx)

    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(save_folder, load_model)))

    print("Testing")

    print("Results on the test set")
    test_set, _ = io.load_relation_graphs_from_file(data_folder + test_set, data=dataset_name)
    test_as_indices = list(graphs_to_indices(test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx, entity2idx=entity2idx))
    
    print("Start testing!")
    result_file = open(os.path.join(result_folder, "_" + model_name), "w")
    test_f1 = 0.0
    for i in tqdm(range(int(test_as_indices[0].shape[0] / model_params['batch_size']))):
        sentence_input = test_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        entity_markers = test_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        labels = test_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        if "RECON" in model_name:
          entity_indices = test_as_indices[4][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
          unique_entities, unique_entities_surface_forms, max_occurred_entity_in_batch_pos = context_utils.get_batch_unique_entities(test_as_indices[4][i * model_params['batch_size']: (i + 1) * model_params['batch_size']], test_as_indices[5][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
          unique_entities_context_indices = context_utils.get_context_indices(unique_entities, unique_entities_surface_forms, context_data, idx2entity, word2idx, char_vocab, model_params['conv_filter_size'], max_sent_len=32, max_num_contexts=32, max_char_len=10, data=dataset_name)
          entities_position = context_utils.get_entity_location_unique_entities(unique_entities, entity_indices)
        if model_name=="RECON-EAC-KGGAT":
          gat_entity_embeddings = context_utils.get_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)
        elif model_name=="RECON":
          gat_entity_embeddings, nonzero_gat_entity_embeddings, nonzero_entity_pos = context_utils.get_selected_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)

        with torch.no_grad():
            if model_name == "RECON":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']],
                                Variable(torch.from_numpy(unique_entities.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(entity_indices.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[0].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[1].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[2].astype(bool))).cuda(),
                                Variable(torch.from_numpy(entities_position.astype(int))).cuda(),
                                max_occurred_entity_in_batch_pos,
                                Variable(torch.from_numpy(nonzero_gat_entity_embeddings.astype(np.float32)), requires_grad=False).cuda(),
                                nonzero_entity_pos,
                                Variable(torch.from_numpy(gat_entity_embeddings.astype(np.float32)), requires_grad=False).cuda())
            elif model_name == "RECON-EAC-KGGAT":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']],
                                Variable(torch.from_numpy(unique_entities.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(entity_indices.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[0].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[1].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[2].astype(bool))).cuda(),
                                Variable(torch.from_numpy(entities_position.astype(int))).cuda(),
                                max_occurred_entity_in_batch_pos,
                                Variable(torch.from_numpy(gat_entity_embeddings.astype(np.float32)), requires_grad=False).cuda())
            elif model_name == "RECON-EAC":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']],
                                Variable(torch.from_numpy(unique_entities.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(entity_indices.astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[0].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[1].astype(np.long))).cuda(),
                                Variable(torch.from_numpy(unique_entities_context_indices[2].astype(bool))).cuda(),
                                Variable(torch.from_numpy(entities_position.astype(int))).cuda(),
                                max_occurred_entity_in_batch_pos)
            elif model_name == "GPGNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                Variable(torch.from_numpy(np.array(test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).cuda())
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda())

            _, predicted = torch.max(output, dim=1)
            labels_copy = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels_copy) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels_copy = np.array(labels_copy)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(
              predicted, labels_copy, empty_label=p0_index)
            test_f1 += add_f1

        score = F.softmax(output, dim=-1)
        score = to_np(score).reshape(-1, n_out)
        labels = labels.reshape(-1)
        p_indices = labels != 0
        score = score[p_indices].tolist()
        labels = labels[p_indices].tolist()
        pred_labels = r = np.argmax(score, axis=-1)
        indices = [i for i in range(len(p_indices)) if p_indices[i]]
        if(model_name != "LSTM" and model_name != "PCNN" and model_name != "CNN"):
            entity_pairs = test_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            entity_pairs = reduce(lambda x,y :x+y , entity_pairs)
        else:
            entity_pairs = test_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]    

        start_idx = i * model_params['batch_size']
        for index, (i, j, entity_pair) in enumerate(zip(score, labels, entity_pairs)):
            sent = ' ' .join(test_set[ start_idx + indices[index]//(model_params['max_num_nodes']*(model_params['max_num_nodes']-1)) ]['tokens']).strip()
            result_file.write("{} | {} | {} | {} | {} | {}\n".format(sent, entity_pair[0], entity_pair[1], idx2property[pred_labels[index]], idx2property[labels[index]], score[index][pred_labels[index]]))

    print("Test f1: ", test_f1 * 1.0 /
            (test_as_indices[0].shape[0] / model_params['batch_size']))
    result_file.close()

test()