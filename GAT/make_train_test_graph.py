import os
import json
from tqdm import tqdm

data_dir = '../data/GAT_entCtx_augmented/WikipediaWikidataDistantSupervisionAnnotations.v1.0'
context_data_file = '../data/WikipediaWikidataDistantSupervisionAnnotations.v1.0/entities_context.json'

with open(context_data_file, 'r') as f:
  context_data = json.load(f)

with open(os.path.join(data_dir,'train.txt'), 'r') as f:
  data = f.read()
  lines_triples = data.split('\n')
  triples_json = {}
  entities_train_json = {}
  for line in tqdm(lines_triples):
    triples_json[line] = 1
    line_arr = line.strip().split(' ')
    if len(line_arr)==3:
      entities_train_json[line_arr[0]] = 1
      entities_train_json[line_arr[2]] = 1
with open(os.path.join(data_dir,'test.txt'), 'r') as f:
  data = f.read()
  lines_triples_test = data.split('\n')
  triples_test_json = {}
  entities_test_only_json = {}
  for line in tqdm(lines_triples_test):
    triples_json[line] = 1
    line_arr = line.strip().split(' ')
    if len(line_arr)==3:
      if entities_train_json.get(line_arr[0],None) is None:
        entities_test_only_json[line_arr[0]] = 1
      if entities_train_json.get(line_arr[2],None) is None:
        entities_test_only_json[line_arr[2]] = 1
with open(os.path.join(data_dir,'entity2id.txt'), 'r') as f:
  data_entities = f.read()
  lines_entities = data_entities.split('\n')
  entity2id = {}
  for line in tqdm(lines_entities):
    line_arr = line.strip().split(' ')
    if len(line_arr)==2:
      entity2id[line_arr[0]] = int(line_arr[1])

lines_triples.extend(lines_triples_test)
all_triples_json = {}
for triple in lines_triples:
  all_triples_json[triple] = 1

triples_to_add = []
for entity, _ in tqdm(entities_test_only_json.items()):
  if context_data.get(entity,None) is None:
    continue

  for instance_of in context_data[entity]['instances']:
    e1 = entity
    e2 = instance_of['kbID']
    r = 'P31'
    triple = ' '.join([e1,r,e2]).strip()

    if triples_json.get(triple,None) is None:
      lines_triples.append(triple)
    if entity2id.get(e2,None) is None:
      entity2id[e2] = len(entity2id)

with open(os.path.join(data_dir,'train_test.txt'), 'w') as f:
  triples_str = ''
  for line in tqdm(lines_triples):
    triples_str += '{}\n'.format(line)
  f.write(triples_str)
with open(os.path.join(data_dir,'entity2id.txt'), 'w') as f:
  entity2id_str = ''
  for entity, _id in tqdm(entity2id.items()):
    entity2id_str += '{} {}\n'.format(entity,_id)
  f.write(entity2id_str) 