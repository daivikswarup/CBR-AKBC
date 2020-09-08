from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tempfile
from scipy.sparse import csr_matrix, hstack, eye
from typing import DefaultDict, List, Tuple, Dict, Set



def augment_kb_with_inv_edges(file_name: str) -> None:
    # Create temporary file read/write
    t = tempfile.NamedTemporaryFile(mode="r+")
    # Open input file read-only
    i = open(file_name, 'r')

    # Copy input file to temporary file, modifying as we go
    temp_list = []
    for line in i:
        t.write(line.strip() + "\n")
        e1, r, e2 = line.strip().split("\t")
        temp_list.append((e1, r, e2))
        temp_list.append((e2, "_" + r, e1))

    i.close()  # Close input file
    o = open(file_name, "w")  # Reopen input file writable
    # Overwriting original file with temporary file contents
    for (e1, r, e2) in temp_list:
        o.write("{}\t{}\t{}\n".format(e1, r, e2))
    t.close()  # Close temporary file, will cause it to be deleted
    o.close()


def create_adj_list(file_name: str) -> DefaultDict[str, Tuple[str, str]]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line_ctr, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[e1].append((r, e2))
    return out_map


def load_data(file_name: str) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    fin = open(file_name)

    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[(e1, r)].append(e2)

    return out_map

def load_data_all_triples(train_file: str, dev_file: str, test_file: str) -> DefaultDict[Tuple[str, str], list]:
    """
    Returns a map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    out_map = defaultdict(list)
    for file_name in [train_file, dev_file, test_file]:
        fin = open(file_name)
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("\t")
            out_map[(e1, r)].append(e2)
    return out_map



def create_vocab(kg_file: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    entity_vocab, rev_entity_vocab = {}, {}
    rel_vocab, rev_rel_vocab = {}, {}
    fin = open(kg_file)
    entity_ctr, rel_ctr = 0, 0
    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        if e1 not in entity_vocab:
            entity_vocab[e1] = entity_ctr
            rev_entity_vocab[entity_ctr] = e1
            entity_ctr += 1
        if e2 not in entity_vocab:
            entity_vocab[e2] = entity_ctr
            rev_entity_vocab[entity_ctr] = e2
            entity_ctr += 1
        if r not in rel_vocab:
            rel_vocab[r] = rel_ctr
            rev_rel_vocab[rel_ctr] = r
            rel_ctr += 1
    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab


def read_graph(file_name: str, entity_vocab: Dict[str, int], rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, r, _ = line.split("\t")
        adj_mat[entity_vocab[e1], rel_vocab[r]] = 1

    return adj_mat

def read_graph_sparse(filename, entity_vocab, rel_vocab, ngram_size, use_entity):
    adj_list = create_adj_list(filename)
    index_adj_list = {entity_vocab[e1]: [(rel_vocab[r], entity_vocab[e2]) for
                                         r, e2 in lis] for e1, lis in
                      adj_list.items()}
    return get_features(index_adj_list, len(entity_vocab), ngram_size, use_entity) 

def set_vectorizer(dic_set, num_entities):
    """Convert dictionary to sparse matrix

    :dic_set: Dict mapping Entity -> set of features
    :num_entities: Number of entities

    :returns: CSR matrix of Num_entities x num_features

    """
    all_feats = set()
    for key, st in dic_set.items():
        all_feats |= st
    num_features = len(all_feats)
    features = list(all_feats)
    feat2id = {f:i for i, f in enumerate(features)}
    rows, cols, data = [], [], []
    for key, st in dic_set.items():
        for feat in st:
            rows.append(key)
            cols.append(feat2id[feat])
            data.append(1)
    return csr_matrix((data, (rows, cols)), shape=(num_entities, num_features))


def get_entity_sparse(adj_list, num_entities):
    """ Returns sparse adjacency matrix of size n_entities x n_entities """
    entity_rows = []
    entity_cols = []
    for e1, lis in tqdm(adj_list.items()):
        for _, e2 in lis:
            entity_rows.append(e1)
            entity_cols.append(e2)
    entity_data = [1] * len(entity_rows) 
    entity_adj = csr_matrix((entity_data,(entity_rows, entity_cols)),
                            shape=(num_entities, num_entities))
    return entity_adj


def get_path_features(adj_list, num_entities, ngram_size=1):
    """ Returns list of sparse metrices each of size 
    num_entities x num_features_i where num_features_i is the number of
    distinct i length paths 
    """
    paths = [defaultdict(lambda: {()})]
    for i in range(ngram_size):
        nexthop = defaultdict(set)
        for e1, lis in tqdm(adj_list.items()):
            for r, e2 in lis:
                nexthop[e1] |= {(r,)+x for x in paths[-1][e2]}
        paths.append(nexthop)
    path_matrices = [set_vectorizer(p, num_entities) for p in paths[1:]]   
    return path_matrices

def get_multihop_entity_features(adj_list, num_entities, ngram_size=1):
    """ Returns list of sparse metrices each of size 
    num_entities x num_entities, each storing the neighborhood after i hops
    """
    entity_adj = get_entity_sparse(adj_list, num_entities)
    all_features = [entity_adj]
    for i in range(ngram_size-1):
        next_hop = entity_adj.dot(all_features[-1])
        all_features.append(next_hop)
    return all_features

def get_features(adj_list,num_entities, ngram_size=1, use_entity=False):
    """Returns sparse matrix of size n_entities x n_features including all the
    ngram path and entity features
    """
    all_features = get_path_features(adj_list, num_entities,  ngram_size)
    if use_entity:
        all_features += get_multihop_entity_features(adj_list, num_entities, ngram_size)
    return hstack(all_features)


def load_mid2str(mid2str_file: str) -> DefaultDict[str, str]:
    mid2str = defaultdict(str)
    with open(mid2str_file) as fin:
        for line in tqdm(fin):
            line = line.strip()
            try:
                mid, ent_name = line.split("\t")
            except ValueError:
                continue
            if mid not in mid2str:
                mid2str[mid] = ent_name
    return mid2str


def get_unique_entities(kg_file: str) -> Set[str]:
    unique_entities = set()
    fin = open(kg_file)
    for line in fin:
        e1, r, e2 = line.strip().split()
        unique_entities.add(e1)
        unique_entities.add(e2)
    fin.close()
    return unique_entities


def get_entities_group_by_relation(file_name: str) -> DefaultDict[str, List[str]]:
    rel_to_ent_map = defaultdict(list)
    fin = open(file_name)
    for line in fin:
        e1, r, e2 = line.strip().split()
        rel_to_ent_map[r].append(e1)
    return rel_to_ent_map


def get_inv_relation(r: str, dataset_name="nell") -> str:
    if dataset_name == "nell":
        if r[-4:] == "_inv":
            return r[:-4]
        else:
            return r + "_inv"
    else:
        if r[:2] == "__" or r[:2] == "_/":
            return r[1:]
        else:
            return "_" + r


def return_nearest_relation_str(sim_sorted_ind, rev_rel_vocab, rel, k=5):
    """
    helper method to print nearest relations
    :param sim_sorted_ind: sim matrix sorted wrt index
    :param rev_rel_vocab:
    :param rel: relation we want sim for
    :return:
    """
    print("====Query rel: {}====".format(rev_rel_vocab[rel]))
    nearest_rel_inds = sim_sorted_ind[rel, :k]
    return [rev_rel_vocab[i] for i in nearest_rel_inds]
