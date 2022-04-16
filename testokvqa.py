import os
import json
import yaml
import argparse
import numpy as np

from math import log
import dgl
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint
from model.model_okvqa import CMGCNnet
from model.okvqa_train_dataset import OkvqaTrainDataset
from model.okvqa_test_dataset import OkvqaTestDataset
from model.fvqa_train_dataset_org import FvqaTrainDataset
from model.fvqa_test_dataset import FvqaTestDataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# model = TheModelClass(*args, **kwargs)


def cal_acc(answers, preds):
    all_num = len(preds)
    acc_num_1 = 0

    for i, answer_id in enumerate(answers):
        pred = preds[i]  # (num_nodes)
        try:

            _, idx_1 = torch.topk(pred, k=1)

        except RuntimeError:
            continue
        else:
            if idx_1.item() == answer_id:
                acc_num_1 = acc_num_1 + 1

    return acc_num_1 / all_num


def collate_fn(batch):
    res = {}
    qid_list = []
    question_list = []
    question_length_list = []
    img_features_list = []
    img_relations_list = []

    fact_num_nodes_list = []
    facts_node_features_list = []
    facts_e1ids_list = []
    facts_e2ids_list = []
    facts_answer_list = []
    facts_answer_id_list = []

    semantic_num_nodes_list = []
    semantic_node_features_list = []
    semantic_e1ids_list = []
    semantic_e2ids_list = []
    semantic_edge_features_list = []
    semantic_num_nodes_list = []

    for item in batch:
        # question
        qid = item['id']
        qid_list.append(qid)

        question = item['question']
        question_list.append(question)

        question_length = item['question_length']
        question_length_list.append(question_length)

        # image
        img_features = item['img_features']
        img_features_list.append(img_features)

        img_relations = item['img_relations']
        img_relations_list.append(img_relations)

        # fact
        fact_num_nodes = item['facts_num_nodes']
        fact_num_nodes_list.append(fact_num_nodes)

        facts_node_features = item['facts_node_features']
        facts_node_features_list.append(facts_node_features)

        facts_e1ids = item['facts_e1ids']
        facts_e1ids_list.append(facts_e1ids)

        facts_e2ids = item['facts_e2ids']
        facts_e2ids_list.append(facts_e2ids)

        facts_answer = item['facts_answer']
        facts_answer_list.append(facts_answer)

        facts_answer_id = item['facts_answer_id']
        facts_answer_id_list.append(facts_answer_id)

        # semantic
        semantic_num_nodes = item['semantic_num_nodes']
        semantic_num_nodes_list.append(semantic_num_nodes)

        semantic_node_features = item['semantic_node_features']
        semantic_node_features_list.append(semantic_node_features)

        semantic_e1ids = item['semantic_e1ids']
        semantic_e1ids_list.append(semantic_e1ids)

        semantic_e2ids = item['semantic_e2ids']
        semantic_e2ids_list.append(semantic_e2ids)

        semantic_edge_features = item['semantic_edge_features']
        semantic_edge_features_list.append(semantic_edge_features)

    res['id_list'] = qid_list
    res['question_list'] = question_list
    res['question_length_list'] = question_length_list
    res['features_list'] = img_features_list
    res['img_relations_list'] = img_relations_list
    res['facts_num_nodes_list'] = fact_num_nodes_list
    res['facts_node_features_list'] = facts_node_features_list
    res['facts_e1ids_list'] = facts_e1ids_list
    res['facts_e2ids_list'] = facts_e2ids_list
    res['facts_answer_list'] = facts_answer_list
    res['facts_answer_id_list'] = facts_answer_id_list
    res['semantic_node_features_list'] = semantic_node_features_list
    res['semantic_e1ids_list'] = semantic_e1ids_list
    res['semantic_e2ids_list'] = semantic_e2ids_list
    res['semantic_edge_features_list'] = semantic_edge_features_list
    res['semantic_num_nodes_list'] = semantic_num_nodes_list
    return res


def cal_batch_loss(fact_batch_graph, batch, device, pos_weight, neg_weight):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h']  # (n,2)
        answer = answers[i].long().to(device)
        weight = torch.FloatTensor([0.9, 0.1]).to(device)
        loss_fn=torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(pred, answer)
        batch_loss = batch_loss + loss

    return batch_loss / len(answers)


parser = argparse.ArgumentParser()

parser.add_argument("--cpu-workers", type=int, default=0, help="Number of CPU workers for dataloader.")
parser.add_argument("--overfit", action="store_true", help="Whether to validate on val split after every epoch.")
parser.add_argument("--validate", action="store_true", help="Whether to validate on val split after every epoch.")
parser.add_argument("--gpu-ids", nargs="+", type=int, default=0, help="List of ids of GPUs to use.")
parser.add_argument("--dataset", default="okvqa", help="dataset that model training on")

args = parser.parse_args()

config_path = 'model/config_okvqa.yml'
config = yaml.load(open(config_path))

# glove 相关
print('Loading glove...')
glovevocabulary = Vocabulary(config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"])
glove = np.load(config['dataset']['glove_vec_path'])
glove = torch.Tensor(glove)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = CMGCNnet(config,
                 que_vocabulary=glovevocabulary,
                 glove=glove,
                 device=device)


# model.load_state_dict(torch.load('exp_okvqa/checkpoint_8.pth'))
model_state, optimizer = load_checkpoint('exp_okvqa/checkpoint_8.pth')
model.load_state_dict(model_state)
model.eval()
answers = []  # [batch_answers,...]
preds = []  # [batch_preds,...]

print('Loading OKVQATestDataset...')
val_dataset = OkvqaTestDataset(config, overfit=args.overfit, in_memory=True)
# val_dataset.sliced_data(8)
# train_set, val_set = torch.utils.data.random_split(val_dataset, [3000, 2046])
# torch.save(train_set, "train.pt")
#
# torch.save(val_set, "val.pt")
# val_dataset = torch.load("val.pt")



val_dataloader = DataLoader(val_dataset,
                            batch_size=config['solver']['batch_size'],
                            num_workers=args.cpu_workers,
                            shuffle=True,
                            collate_fn=collate_fn)

print(f"Validation:")
for i, batch in enumerate(tqdm(val_dataloader)):
    with torch.no_grad():
        fact_batch_graph = model(batch)
    batch_loss = cal_batch_loss(fact_batch_graph,
                                batch,
                                device,
                                neg_weight=0.1,
                                pos_weight=0.9)

    fact_graphs = dgl.unbatch(fact_batch_graph)
    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h'].squeeze()  # (num_nodes,1)
        preds.append(pred[:, 1])  # [(num_nodes,)]
        answers.append(batch['facts_answer_id_list'][i])

acc_1 = cal_acc(answers, preds)

print("acc@1={:.2%} ".format(acc_1))

abc = 123



