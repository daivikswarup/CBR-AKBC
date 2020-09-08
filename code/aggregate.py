import argparse
import logging
import os
import json


logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate scores from all "
                                     "workers")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/nfs/work1/mccallum/dswarupogguv/cbr_lse/output/")
    parser.add_argument("--dataset_name", type=str, help="The dataset name. Replace with one of FB122 | WN18RR | NELL-995 to reproduce the results of the paper")
    args = parser.parse_args()

    dirname = os.path.join(args.output_dir,\
                                args.dataset_name)
    hits_1 = hits_5 = hits_3 = hits_10 = mrr= total = 0
    for fil in os.listdir(dirname):
        fname = os.path.join(dirname, fil)
        with open(fname, 'r') as f:
            data = json.load(f)
            hits_1 += data['hits_1']
            hits_3 += data['hits_3']
            hits_5 += data['hits_5']
            hits_10 += data['hits_10']
            mrr += data['MRR']
            total += data['total']
    logger.info("Hits@1 {}".format(hits_1 / total))
    logger.info("Hits@3 {}".format(hits_3 / total))
    logger.info("Hits@5 {}".format(hits_5 / total))
    logger.info("Hits@10 {}".format(hits_10 / total))
    logger.info("MRR {}".format(mrr / total))



