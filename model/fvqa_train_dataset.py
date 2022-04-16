import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import json
import numpy as np
import pickle
import torch
from util.vocabulary import Vocabulary
from torch.utils.data import DataLoader
import dgl
from math import sqrt, atan2
import yaml
import h5py


class FvqaTrainDataset(Dataset):
    def __init__(self, config, overfit=False, in_memory=True):
        super().__init__()

        self.config = config
        self.overfit = overfit
        self.qids = []
        self.questions = []
        self.question_lengths = []
        self.image_ids = []

        self.fact_num_nodes = []
        self.fact_e1ids_list = []
        self.fact_e2ids_list = []
        self.fact_answer_ids = []
        self.fact_answer_list = []

        self.semantic_num_nodes = []
        self.semantic_e1ids_list = []
        self.semantic_e2ids_list = []

        self.image_features = []
        self.image_relations = []
        self.image_files = []

        fvqa_path = "/home/ifrahmaqsood/fvqa_dataset_mucko"
       
        print('loading dataset_opendomain_q_raw.json')
        with open(os.path.join(fvqa_path, 'question_raw_train_with_graphs.json'), 'r') as f:
        # with open(os.path.join(fvqa_path, 'question_raw_train_with_graphs_without_gt_fact.json'), 'r') as f:
            qa_raw = json.load(f)

        print('loading image_feature.npz')
        img_data = np.load(os.path.join(fvqa_path, 'train_image_feature.npz'))
        self.image_features = img_data['image_features']
        self.image_relations = img_data['image_relations']

        print('loading semantic_graph_feature.npz')
        # my changes in code (, allow_pickle=True added)
        sem_data = np.load(os.path.join(fvqa_path, 'semantic_graph_train_feature.npz'), allow_pickle=True)
        self.semantic_graph_node_features = sem_data['node_features']
        self.semantic_graph_edge_features = sem_data['edge_features']

        print('loading fact_graph_feature.npz')
        # my changes in code (, allow_pickle=True added)
        fact_data = np.load(os.path.join(fvqa_path, 'fact_graph_train_feature.npz'), allow_pickle=True)
        # fact_data = np.load(os.path.join(fvqa_path, 'fact_graph_train_feature_without_gt_fact.npz'), allow_pickle=True)
        self.fact_graph_node_features = fact_data['node_features']
        self.fact_graph_edge_features = fact_data['edge_features']

        for qid, qa_item in qa_raw.items():
            image_file = qa_item['image_id']
            self.qids.append(qid)
            self.questions.append(qa_item['question'])
            self.question_lengths.append(qa_item['question_length'])
            self.image_files.append(image_file)

            self.fact_num_nodes.append(qa_item['fact_graph']['num_node'])
            self.fact_e1ids_list.append(qa_item['fact_graph']['e1ids'])
            self.fact_e2ids_list.append(qa_item['fact_graph']['e2ids'])
            self.fact_answer_ids.append(qa_item['fact_graph']['answer_id'])

            self.semantic_num_nodes.append(qa_item['semantic_graph']['num_node'])
            self.semantic_e1ids_list.append(qa_item['semantic_graph']['e1ids'])
            self.semantic_e2ids_list.append(qa_item['semantic_graph']['e2ids'])

        if overfit:
            self.qids = self.qids[:100]
            self.questions = self.questions[:100]
            self.question_lengths = self.question_lengths[:100]

            self.fact_num_nodes = self.fact_num_nodes[:100]
            self.fact_e1ids_list = self.fact_e1ids_list[:100]
            self.fact_e2ids_list = self.fact_e2ids_list[:100]
            self.fact_answer_ids = self.fact_answer_ids[:100]
            self.fact_graph_node_features = self.fact_graph_node_features[:100]

            self.semantic_num_nodes = self.semantic_num_nodes[:100]
            self.semantic_e1ids_list = self.semantic_e1ids_list[:100]
            self.semantic_e2ids_list = self.semantic_e2ids_list[:100]
            self.semantic_graph_node_features = self.semantic_graph_node_features[:100]
            self.semantic_graph_edge_features = self.semantic_graph_edge_features[:100]

            self.image_features = self.image_features[:100]
            self.image_relations = self.image_relations[:100]

    def __getitem__(self, index):
        image_id = self.image_files[index]

        image_feature = torch.Tensor(self.image_features[index])
        # 图像归一化
        if self.config['dataset']["img_norm"]:
            image_feature = normalize(image_feature, dim=0, p=2)

        item = {}
        item['id'] = self.qids[index]
        item['question'] = torch.Tensor(self.questions[index]).long()
        item['question_length'] = self.question_lengths[index]
        item['img_features'] = image_feature
        item['img_relations'] = torch.Tensor(self.image_relations[index])

        item['facts_num_nodes'] = self.fact_num_nodes[index]
        item['facts_node_features'] = torch.Tensor(self.fact_graph_node_features[index])
        item['facts_edge_features'] = torch.Tensor(self.fact_graph_edge_features[index])
        item['facts_e1ids'] = torch.Tensor(self.fact_e1ids_list[index]).long()
        item['facts_e2ids'] = torch.Tensor(self.fact_e2ids_list[index]).long()
        item['facts_answer_id'] = self.fact_answer_ids[index]

        answer = np.zeros(self.fact_num_nodes[index])
        if self.fact_answer_ids[index] != -1:
            answer[self.fact_answer_ids[index]] = 1
        item['facts_answer'] = torch.Tensor(answer)

        item['semantic_num_nodes'] = self.semantic_num_nodes[index]
        item['semantic_node_features'] = torch.Tensor(self.semantic_graph_node_features[index])
        item['semantic_edge_features'] = torch.Tensor(self.semantic_graph_edge_features[index])
        item['semantic_e1ids'] = torch.Tensor(self.semantic_e1ids_list[index]).long()
        item['semantic_e2ids'] = torch.Tensor(self.semantic_e2ids_list[index]).long()

        return item

    def __len__(self):
        if (self.overfit):
            return 100
        else:
            return len(self.qids)
