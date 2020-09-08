import argparse
import uuid
import numpy as np
from sklearn.preprocessing import normalize
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm, trange
from collections import defaultdict, deque
import pickle
import torch
from code.data.data_utils import create_vocab,create_adj_list, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, \
    load_data_all_triples, read_graph_sparse
from scipy.sparse import vstack
from typing import *
import logging
import json
import sys
import wandb
import networkx as nx
import itertools
import torch.nn as nn
from code.models.path_encoder import PathScorer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class CBR(object):
    def __init__(self, args, full_map, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, adj_list, rel_ent_map):
        self.args = args
        self.run_id = uuid.uuid4().hex
        self.eval_map = eval_map
        self.train_map = train_map
        self.full_map = full_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.adj_list = adj_list
        self.rel_ent_map = rel_ent_map
        self.num_non_executable_programs = []
        self.index_adj_list, self.g, self.etypes = CBR.get_indexed_adj_list(adj_list,
                                                                     entity_vocab,
                                                                     rel_vocab)
        self.path_scorer = PathScorer(len(self.entity_vocab),
                                      len(self.rel_vocab), 128)
        self.path_scorer.cuda()

    @staticmethod
    def get_indexed_adj_list(adj_list, entity_vocab, rel_vocab):
        # Do not need multidigraph or digraph since we only
        # want paths
        g = nx.Graph()
        g.add_nodes_from([v for k, v in entity_vocab.items()])
        new_adj_list = {entity_vocab[k]: [(rel_vocab[r],entity_vocab[e]) for r,e in v] for k, v in adj_list.items()}
        for k, v in entity_vocab.items():
            if v not in new_adj_list:
                new_adj_list[v] = []
        etypes = defaultdict(list)
        for k, l in new_adj_list.items():
            g.add_edges_from([(k,e2) for r, e2 in l])
            for r, e2 in l:
                etypes[k, e2].append(r)

        return new_adj_list, g, etypes

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    def calc_sim(self, kg_file: Type[torch.Tensor], query_entities:
                 Type[torch.LongTensor], max_sim = 100) -> Type[torch.LongTensor]:
        """
        :param adj_mat: N X R
        :param query_entities: b is a batch of indices of query entities
        :return:
        """
        fname = os.path.join(self.args.output_dir, self.args.dataset_name,
                             "sim.pkl")
        if not os.path.exists(fname):
            logger.info('reading sparse mat')
            adj_mat = read_graph_sparse(kg_file, self.entity_vocab,
                                        self.rel_vocab, self.args.ngrams,
                                       self.args.use_entities)
            adj_mat = np.sqrt(adj_mat)
            logger.info('normalizing sparse mat')
            adj_mat = normalize(adj_mat)
            nbrs = NearestNeighbors(n_neighbors=max_sim, n_jobs=-1).fit(adj_mat)
            dist, ind = nbrs.kneighbors(adj_mat)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as f:
                pickle.dump(ind, f)
        else:
            with open(fname, 'rb') as f:
                ind = pickle.load(f)

        return ind

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> List[str]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.entity_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.full_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

    def get_all_path_variants(self, path):
        """ There may be multiedges"""
        if len(path) == 2:
            for x in self.etypes[path[0], path[1]]:
                yield[x]
        else:
            for r in self.etypes[path[0], path[1]]:
                for p in self.get_all_path_variants(path[1:]):
                    yield [r] + p

    def get_programs(self, e: str, ans: str, max_len:Optional[int]=3):
        """
        Given an entity and answer, get all paths? which end at that ans node in the subgraph surrounding e
        """

        start_node = self.entity_vocab[e]
        train_adj_list = self.index_adj_list
        all_programs = []
        pathset = set()
        targetset = set([self.entity_vocab[x] for x in ans])
        for i in range(self.args.n_paths):    
            curr_node = start_node
            path = []
            for l in range(max_len):
                outgoing_edges = train_adj_list[curr_node]
                if len(outgoing_edges) == 0:
                    break
                # pick one at random
                out_edge_idx = np.random.randint(len(outgoing_edges))
                # assign curr_node as the node of the selected edge
                r, curr_node = outgoing_edges[out_edge_idx]
                path.append((r,curr_node))
            pathset.add(tuple(path))
        for p in pathset:
            for i in range(len(p)):
                r,e = p[i]
                if e in targetset:
                    all_programs.append([self.rev_rel_vocab[r] for r,_ in \
                                         p[:i+1]])
        return all_programs


    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        if nearest_entities is None:
            self.all_num_ret_nn.append(0)
            return []
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.full_map.get((e, r), [])) > 0:
                nn_answers = self.full_map[(e, r)]
                all_programs.extend(self.get_programs(e, nn_answers))
            else:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
        return all_programs

    def rank_programs(self, list_programs: List[str]) -> List[str]:
        """
        Rank programs.
        """
        # Lets rank it simply by count:
        count_map = defaultdict(int)
        for p in list_programs:
            count_map[tuple(p)] += 1
        sorted_programs = sorted(count_map.items(), key=lambda kv: -kv[1])
        sorted_programs = [k for (k, v) in sorted_programs]
        return sorted_programs

    def execute_one_program(self, e: str, path: List[str], depth: int, max_branch: int):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        if depth == len(path):
            # reached end, return node
            return [e]
        next_rel = path[depth]
        next_entities = self.full_map[(e, path[depth])]
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        answers = []
        for e_next in next_entities:
            answers += self.execute_one_program(e_next, path, depth + 1, max_branch)
        return answers

    def execute_programs(self, e: str, path_list: List[List[str]], max_branch:
                         Optional[int] = 20) -> List[str]:

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for path in path_list:
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path, depth=0, max_branch=max_branch)
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans

        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    def rank_answers(self, list_answers: List[str]) -> List[str]:
        """
        Different ways to re-rank answers
        """
        # 1. rank based on occurrence, i.e. how many paths did end up at this entity?
        count_map = defaultdict(int)
        uniq_entities = set()
        for e in list_answers:
            count_map[e] += 1
            uniq_entities.add(e)
        sorted_entities_by_val = sorted(count_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        rank = 0
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1

    def get_hits(self, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) -> Tuple[float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred in list_answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    filtered_answers.append(pred)

            rank = CBR.get_rank_in_list(gold_answer, filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
        return hits_10, hits_5, hits_3, hits_1, rr

    @staticmethod
    def get_accuracy(gold_answers: List[str], list_answers: List[str]) -> List[float]:
        all_acc = []
        for gold_ans in gold_answers:
            if gold_ans in list_answers:
                all_acc.append(1.0)
            else:
                all_acc.append(0.0)
        return all_acc

    def execute_program_ents(self, ent,  program, max_branch=20):
        q = deque()
        solutions = defaultdict(list)
        q.append((ent, 0, []))
        while len(q):
            e1, depth, path = q.popleft()
            if depth == len(program):
                solutions[e1].append(path + [(self.entity_vocab[e1],
                                              len(self.rel_vocab))])
                continue
            rel = program[depth]
            next_entities = self.full_map[e1, rel]
            if len(next_entities) > max_branch:
                next_entities = np.random.choice(next_entities, max_branch,
                                                 replace=False)
            depth += 1
            for e2 in next_entities:
                q.append((e2, depth, path + [(self.entity_vocab[e1],
                                              self.rel_vocab[rel])]))
        return solutions

    def get_entity_programs(self, e1, programs):
        programs_to_entity = defaultdict(list)
        for p in programs:
            for ent, programs in self.execute_program_ents(e1, p).items():
                programs_to_entity[ent].extend(programs)
        return programs_to_entity

    def get_training_data(self, e1, programs, e2_list):
        e2_set = set(e2_list)
        programs_to_entity = self.get_entity_programs(e1, programs)
        keys = list(programs_to_entity.keys())
        programs = [programs_to_entity[k] for k in keys]
        labels = [k in e2_set for k in keys]
        return programs, labels


    def train_pathscorer(self, lr = 1e-4, n_epochs = 20):
        model_path = \
                 os.path.join(self.args.output_dir,self.run_id,self.args.dataset_name,'pathscorer')
        os.makedirs(model_path, exist_ok=True)
        with   \
            open(os.path.join(self.args.output_dir,self.run_id,self.args.dataset_name,
                          'config.pkl'), 'wb') as f:
            pickle.dump(vars(self.args), f)
        loss = nn.BCELoss()
        optimizer = torch.optim.Adam(self.path_scorer.parameters(),
                                     lr = lr)
        best_mrr = 0
        for epoch in trange(n_epochs, desc="epochs"):
            for i, ((e1, r), e2_list) in enumerate(tqdm((list(self.train_map.items())),
                                                  desc="Training")):
                if len(e2_list) == 0:
                    continue

                all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                        num_nn=self.args.k_adj)

                # filter the program if it is equal to the query relation
                temp = []
                for p in all_programs:
                    if len(p) == 1 and p[0] == r:
                        continue
                    temp.append(p)
                all_programs = temp
                all_uniq_programs = \
                    self.rank_programs(all_programs)[:self.args.max_num_programs]
                programs, labels = self.get_training_data(e1,
                                                          all_uniq_programs,
                                                          e2_list)
                if self.args.dropout > 0:
                    num_programs = int(len(all_uniq_programs)*(1-self.args.dropout))
                    selected_programs = np.random.choice(len(all_uniq_programs), num_programs)
                    all_uniq_programs = [all_uniq_programs[i] for i in
                                         selected_programs]
                if len(programs) == 0:
                    continue
                scores = self.path_scorer(programs, self.rel_vocab[r])
                labels = torch.tensor(labels).float().cuda()
                l = loss(scores, labels)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                if i %100 == 0:
                    print("Batch {}\t Loss = {}".format(i, l.detach().cpu().numpy()))                  
                if i%1000 == 0:
                    metrics = self.do_symbolic_case_based_reasoning()
                    print('Metrics after batch {} = {}'.format(i, metrics))
                    best_mrr = max(best_mrr, metrics['mrr'])
                    metrics['best_mrr'] = best_mrr
                    if self.args.use_wandb:
                        metrics['epoch'] = epoch
                        wandb.log(metrics)
            model_path = \
                 os.path.join(self.args.output_dir,self.run_id,self.args.dataset_name,'pathscorer',
                                      'model.pt')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.path_scorer.state_dict(), model_path)
            metrics = self.do_symbolic_case_based_reasoning()
            best_mrr = max(best_mrr, metrics['mrr'])
            logger.info('Metrics after epoch {} = {}'.format(epoch,metrics)) 
            # eval without dropout
            train_metrics = self.do_symbolic_case_based_reasoning(split= 'train')
            logger.info('Metrics on training set after epoch {} = {}'.format(epoch,
                                                                             train_metrics)) 
            metrics['train_metrics'] = train_metrics
            metrics['best_mrr'] = best_mrr
            if self.args.use_wandb:
                metrics['epoch'] = epoch
                wandb.log(metrics)



    def do_symbolic_case_based_reasoning(self, split='eval'):
        num_programs = []
        num_answers = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        model_path = os.path.join(self.args.output_dir, self.run_id,
                                  self.args.dataset_name, 'pathscorer',
                                  'model.pt')
        if os.path.exists(model_path):
            self.path_scorer.load_state_dict(torch.load(model_path))
        if split == 'eval':
            eval_map = self.eval_map
        else:
            eval_map = self.train_map

        for _, ((e1, r), e2_list) in enumerate(tqdm((eval_map.items()))):
            # if e2_list is in train list then remove them
            # Normally, this shouldnt happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.full_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.full_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.full_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.full_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.full_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=self.args.k_adj)

            if all_programs is None or len(all_programs) == 0:
                all_acc.append(0.0)
                continue

            # filter the program if it is equal to the query relation
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs = \
                self.rank_programs(all_programs)[:self.args.max_num_programs]

            for u_p in all_uniq_programs:
                learnt_programs[r][u_p] += 1

            num_programs.append(len(all_uniq_programs))
            # # Now execute the program
            # answers, not_executed_programs = self.execute_programs(e1, all_uniq_programs)

            # answers = self.rank_answers(answers)
            answers_cbr, not_executed_programs = self.execute_programs(e1, all_uniq_programs)
            answers_cbr = self.rank_answers(answers_cbr)


            entity_paths = self.get_entity_programs(e1, all_uniq_programs)
            answers = []
            if len(entity_paths) > 0:
                rel_id = self.rel_vocab[r]
                entities = [e for e, p in entity_paths.items()]
                paths = [p for e, p in entity_paths.items()]
                path_scores = self.path_scorer(paths,
                                               rel_id).detach().cpu().numpy()
                for e, score in zip(entities, path_scores):
                    answers.append((e, score))
                answers.sort(key = lambda x:-x[1])
            

            if len(answers) > 0:
                acc = self.get_accuracy(e2_list, [k[0] for k in answers])
                _10, _5, _3, _1, rr = self.get_hits([k[0] for k in answers], e2_list, query=(e1, r))
                _10cbr, _5cbr, _3cbr, _1cbr, rrcbr = self.get_hits([k[0] for k
                                                                    in
                                                                    answers_cbr], e2_list, query=(e1, r))
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                if args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_query_count[r] += len(e2_list)
            else:
                acc = [0.0] * len(e2_list)
            all_acc += acc
            num_answers.append(len(answers))
            # put it back
            self.full_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.full_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(args.output_dir, "per_relation_scores.json")
            fout = open(out_file_name, "w")
            logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        logger.info("Accuracy (Loose): {}".format(np.mean(all_acc)))
        logger.info("Hits@1 {}".format(hits_1 / total_examples))
        logger.info("Hits@3 {}".format(hits_3 / total_examples))
        logger.info("Hits@5 {}".format(hits_5 / total_examples))
        logger.info("Hits@10 {}".format(hits_10 / total_examples))
        logger.info("MRR {}".format(mrr / total_examples))
        logger.info("Avg number of nn, that do not have the query relation: {}".format(
            np.mean(self.all_zero_ctr)))
        logger.info("Avg num of returned nearest neighbors: {:2.4f}".format(np.mean(self.all_num_ret_nn)))
        logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                logger.info("query: {}".format(k))
                logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    logger.info((rel, learnt_programs[k][rel]))
                logger.info("=====" * 2)
        if args.parallelize:
            dirname = os.path.join(self.args.output_dir,\
                                   self.args.dataset_name)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(os.path.join(dirname,\
                    '{}.json'.format(self.args.splitid)), 'w') as f:
                results = {'hits_1': hits_1,\
                           'hits_3': hits_3,\
                           'hits_5': hits_5,\
                           'hits_10': hits_10,\
                           'MRR': mrr,\
                           'total' : total_examples}
                json.dump(results, f)
                
        if self.args.use_wandb and self.args.test:
            # Log all metrics
            wandb.log({'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': sum(self.all_zero_ctr), 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs), 'acc_loose': np.mean(all_acc)})

        return {'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': sum(self.all_zero_ctr), 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs), 'acc_loose': np.mean(all_acc)}


def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    kg_file = os.path.join(data_dir, "graph.txt")
    train_adj_list = create_adj_list(os.path.join(data_dir, 'graph.txt'))

    args.dev_file = os.path.join(data_dir, "dev.txt")
    args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
        else os.path.join(data_dir, args.test_file_name)
    if args.dataset_name == "FB122":
        args.test_file = os.path.join(data_dir, "testI.txt")

    args.train_file = os.path.join(data_dir, "train.txt")

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(kg_file)
    logger.info("Loading train map")
    full_map = load_data(kg_file)
    train_map = load_data(args.train_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    if args.parallelize:
        # delete old files
        dirname = os.path.join(args.output_dir,\
                                args.dataset_name)
        if os.path.exists(dirname):
            for fil in os.listdir(dirname):
                fid = int(fil.split('.')[0])
                if fid >= args.num_splits:
                    os.remove(os.path.join(dirname, fil))
                 
        keys = sorted(list(eval_map.keys()))
        splitsize = int(np.ceil(len(keys)/args.num_splits))
        start = splitsize * args.splitid
        end = start + splitsize
        selected_keys = keys[start:end]
        eval_map = {k:eval_map[k] for k in selected_keys}

    rel_ent_map = get_entities_group_by_relation(args.train_file)
    # Calculate nearest neighbors


    # get the unique entities in eval set, so that we can calculate similarity in advance.
    eval_entities = get_unique_entities(eval_file)
    eval_vocab, eval_rev_vocab = {}, {}
    query_ind = []

    e_ctr = 0
    for e in eval_entities:
        try:
            query_ind.append(entity_vocab[e])
        except KeyError:
            continue
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, args.dev_file, args.test_file)
    args.all_kg_map = all_kg_map

    symbolically_smart_agent = CBR(args, full_map, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                                                 rev_rel_vocab, eval_vocab,
                                   eval_rev_vocab, train_adj_list, rel_ent_map)

    # Calculate similarity
    logger.info('calculating similarity')
    nearest_neighbor_1_hop = symbolically_smart_agent.calc_sim(kg_file, 
                                          np.arange(len(entity_vocab)))  # n X N (n== size of dev_entities, N: size of all entities)
    print(nearest_neighbor_1_hop)
    print(nearest_neighbor_1_hop.shape)
    symbolically_smart_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

    logger.info("Loaded...")
    
    if args.train:
        symbolically_smart_agent.train_pathscorer()
    else:
        symbolically_smart_agent.do_symbolic_case_based_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CBR")
    parser.add_argument("--dataset_name", type=str, help="The dataset name. Replace with one of FB122 | WN18RR | NELL-995 to reproduce the results of the paper")
    parser.add_argument("--data_dir", type=str, default="./cbr-akbc-data/")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/nfs/work1/mccallum/dswarupogguv/cbr_lse/output/")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='')
    parser.add_argument("--max_num_programs", type=int, default=15, help="Max number of paths to consider")
    parser.add_argument("--splitid", type=int, default=0, help="Split number")
    parser.add_argument("--train", type=int, default=0, help="Train or test")
    parser.add_argument("--dropout", type=float, default=0, help="Fraction of paths to drop")
    parser.add_argument("--num_splits", type=int, default=20, help="Total "
                                                    "number of workers")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--ngrams", type=int, default=2,
                        help="Lengths of paths for similarity")
    parser.add_argument("--use_entities", type=int, default=0,
                        help="Use neighboring entities for similarity")
    parser.add_argument("--n_paths", type=int, default=1000,
                        help="Number of paths")
    parser.add_argument("--bsize", type=int, default=64,
                        help="Number of paths")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    parser.add_argument("--output_per_relation_scores", action="store_true")

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='case-based-reasoning')
    main(args)
