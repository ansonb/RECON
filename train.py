from utils import evaluation_utils, embedding_utils, context_utils
from semanticgraph import io
from parsing import legacy_sp_models as sp_models
from models import baselines
import numpy as np
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
import ast
from models.factory import get_model
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("-sf", "--save_folder",
                      default="./models/RECON/", help="folder to save the models")
    args.add_argument("-e", "--epochs", type=int,
                      default=10, help="Number of epochs")
    args.add_argument("-b", "--batch_size", type=int,
                      default=50, help="Batch Size")

    args = args.parse_args()
    return args

args = parse_args()

def train():

    """ Main Configurations """
    model_name = "RECON"
    dataset_name = 'wikidata' # Options: wikidata, nyt; any new data will need to be added in the loader in semantic_graph.io
    data_folder = "./data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/"
    save_folder = "./models/RECON/"

    model_params = "model_params.json"
    word_embeddings = "glove.6B.50d.txt"
    train_set = "semantic-graphs-filtered-training.02_06.json"
    val_set = "semantic-graphs-filtered-validation.02_06.json"

    use_char_vocab = False

    gat_embedding_file = None
    gat_relation_embedding_file = None
    # Enter the appropriate file paths here
    if "RECON" in model_name:
        context_data_file = "./data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entities_context.json"
    if "KGGAT" in model_name:
        gat_embedding_file = './models/GAT/WikipediaWikidataDistantSupervisionAnnotations/final_entity_embeddings.json'
        gat_entity2id_file = './data/GAT/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entity2id.txt'
    if model_name=="RECON":
        # Point to the trained model/embedding/data files
        gat_relation_embedding_file = './models/GAT/WikipediaWikidataDistantSupervisionAnnotations/final_relation_embeddings.json'
        gat_relation2id_file = './data/GAT/WikipediaWikidataDistantSupervisionAnnotations.v1.0/relation2id.txt'
        w_ent2rel_all_rels_file = './models/GAT/WikipediaWikidataDistantSupervisionAnnotations/W_ent2rel.json.npy'


    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None
    learning_rate = 1e-3
    shuffle_data = True
    save_model = True
    grad_clip = 0.25
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    with open(model_params) as f:
        model_params = json.load(f)
    global args
    save_folder = args.save_folder
    model_params['batch_size'] = args.batch_size
    model_params['nb_epoch'] = args.epochs
    val_results_file = os.path.join(save_folder,'val_results.json')

    char_vocab_file = os.path.join(save_folder,"char_vocab.json")

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    sp_models.set_max_edges(model_params['max_num_nodes']*(model_params['max_num_nodes']-1), model_params['max_num_nodes'])

    if context_data_file:
      with open(context_data_file, 'r') as f:
          context_data = json.load(f)
    if gat_embedding_file:
      with open(gat_embedding_file, 'r') as f:
          gat_embeddings = json.load(f)
      with open(gat_relation_embedding_file, 'r') as f:
          gat_relation_embeddings = json.load(f)
    if gat_relation_embedding_file:
      W_ent2rel_all_rels = np.load(w_ent2rel_all_rels_file)
      with open(gat_entity2id_file, 'r') as f:
          gat_entity2idx = {}
          data = f.read()
          lines = data.split('\n')
          for line in lines:
            line_arr = line.split(' ')
            if len(line_arr)==2:
              gat_entity2idx[line_arr[0].strip()] = line_arr[1].strip()
      with open(gat_relation2id_file, 'r') as f:
          gat_relation2idx = {}
          data = f.read()
          lines = data.split('\n')
          for line in lines:
            line_arr = line.split(' ')
            if len(line_arr)==2:
              gat_relation2idx[line_arr[0].strip()] = line_arr[1].strip()


    embeddings, word2idx = embedding_utils.load(data_folder + word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")

    training_data, _ = io.load_relation_graphs_from_file(data_folder + train_set, load_vertices=True, data=dataset_name)
    if not use_char_vocab:
        char_vocab = context_utils.make_char_vocab(training_data)
        print("Save char vocab dictionary.")
        with open(char_vocab_file, 'w') as outfile:
            json.dump(char_vocab, outfile, indent=4)
    else:
        with open(char_vocab_file, 'r') as f:
            char_vocab = json.load(f)

    val_data, _ = io.load_relation_graphs_from_file(data_folder + val_set, load_vertices=True, data=dataset_name)

    check_data(training_data)
    check_data(val_data)

    if property_index:
        print("Reading the property index from parameter")
        with open(data_folder + args.property_index) as f:
            property2idx = ast.literal_eval(f.read())
        with open(data_folder + args.entity_index) as f:
            entity2idx = ast.literal_eval(f.read())
    else:
        _, property2idx = embedding_utils.init_random({e["kbID"] for g in training_data
                                                    for e in g["edgeSet"]} | {"P0"}, 1, add_all_zeroes=True, add_unknown=True)
        _, entity2idx = context_utils.init_random({kbID for kbID, _ in context_data.items()} ,
                                                     model_params['embedding_dim'], add_all_zeroes=True, add_unknown=True)
    idx2entity = {v:k for k,v in entity2idx.items()}
    context_data['ALL_ZERO'] = {
      'desc': '',
      'label': 'ALL_ZERO',
      'instances': [],
      'aliases': []
    }

    max_sent_len = max(len(g["tokens"]) for g in training_data)
    print("Max sentence length:", max_sent_len)

    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask   
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON-EAC":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON-EAC-KGGAT":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "RECON":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)

    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx, entity2idx=entity2idx))

    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx, entity2idx=entity2idx))
    val_data = None


    print("Save property dictionary.")
    with open(os.path.join(save_folder, model_name + ".property2idx"), 'w') as outfile:
        outfile.write(str(property2idx))
    print("Save entity dictionary.")
    with open(os.path.join(save_folder, model_name + ".entity2idx"), 'w') as outfile:
        outfile.write(str(entity2idx))

    print("Training the model")

    print("Initialize the model")

    if "RECON" not in model_name:
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out)
    elif model_name=="RECON-EAC":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab)
    elif model_name=="RECON-EAC-KGGAT":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab)
    elif model_name=="RECON":
      model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, char_vocab, gat_relation_embeddings, W_ent2rel_all_rels, idx2property, gat_relation2idx)

    model = model.cuda()
    loss_func = nn.CrossEntropyLoss(ignore_index=0).cuda()

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=model_params['weight_decay'])

    indices = np.arange(train_as_indices[0].shape[0])

    step = 0
    val_results = []
    for train_epoch in range(model_params['nb_epoch']):
        if(shuffle_data):
            np.random.shuffle(indices)
        f1 = 0
        for i in tqdm(range(int(train_as_indices[0].shape[0] / model_params['batch_size']))):
            opt.zero_grad()

            sentence_input = train_as_indices[0][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            entity_markers = train_as_indices[1][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            labels = train_as_indices[2][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            if "RECON" in model_name:
              entity_indices = train_as_indices[4][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
              unique_entities, unique_entities_surface_forms, max_occurred_entity_in_batch_pos = context_utils.get_batch_unique_entities(train_as_indices[4][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]], train_as_indices[5][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
              unique_entities_context_indices = context_utils.get_context_indices(unique_entities, unique_entities_surface_forms, context_data, idx2entity, word2idx, char_vocab, model_params['conv_filter_size'], max_sent_len=32, max_num_contexts=32, max_char_len=10, data=dataset_name)
              entities_position = context_utils.get_entity_location_unique_entities(unique_entities, entity_indices)
            if model_name=="RECON-EAC-KGGAT":
              gat_entity_embeddings = context_utils.get_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)
            elif model_name=="RECON":
              gat_entity_embeddings, nonzero_gat_entity_embeddings, nonzero_entity_pos = context_utils.get_selected_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)

            if model_name == "RECON":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                Variable(torch.from_numpy(np.array(train_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).cuda())
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda())
            
            loss = loss_func(output, Variable(torch.from_numpy(labels.astype(int))).view(-1).cuda())



            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            opt.step()

            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(predicted, labels, empty_label=p0_index)
            f1 += add_f1

            
        train_f1 = f1 / (train_as_indices[0].shape[0] / model_params['batch_size'])
        print("Train f1: ", train_f1)

        val_f1 = 0
        for i in tqdm(range(int(val_as_indices[0].shape[0] / model_params['batch_size']))):
            sentence_input = val_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            entity_markers = val_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            labels = val_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            if "RECON" in model_name:
              entity_indices = val_as_indices[4][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
              unique_entities, unique_entities_surface_forms, max_occurred_entity_in_batch_pos = context_utils.get_batch_unique_entities(val_as_indices[4][i * model_params['batch_size']: (i + 1) * model_params['batch_size']], val_as_indices[5][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
              unique_entities_context_indices = context_utils.get_context_indices(unique_entities, unique_entities_surface_forms, context_data, idx2entity, word2idx, char_vocab, model_params['conv_filter_size'], max_sent_len=32, max_num_contexts=32, max_char_len=10, data=dataset_name)
              entities_position = context_utils.get_entity_location_unique_entities(unique_entities, entity_indices)
            if model_name=='RECON-EAC-KGGAT':
              gat_entity_embeddings = context_utils.get_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)
            elif model_name=="RECON":
              gat_entity_embeddings, nonzero_gat_entity_embeddings, nonzero_entity_pos = context_utils.get_selected_gat_entity_embeddings(entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)

            if model_name == "RECON":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
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
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                Variable(torch.from_numpy(np.array(train_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).cuda())
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda())
            
            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(
                predicted, labels, empty_label=p0_index)
            val_f1 += add_f1

        val_f1 = val_f1 / (val_as_indices[0].shape[0] / model_params['batch_size'])
        print("Validation f1: ", val_f1)

        val_results.append({
          'train_f1': train_f1,
          'val_f1': val_f1
        })

        # save model
        if (train_epoch % 1 == 0 and save_model):
            torch.save(model.state_dict(), "{0}{1}-{2}.out".format(save_folder, model_name, str(train_epoch)))

        step = step + 1

        with open(val_results_file, 'w') as f:
          json.dump(val_results, f, indent=4, cls=context_utils.CustomEncoder)

train()