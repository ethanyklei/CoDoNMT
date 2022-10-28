from collections import defaultdict
import os 
import json

import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--index-path", type=str)
    parser.add_argument("--save-dir", type=str)

    return parser.parse_args()
    


args = get_args()

with open(args.index_path, 'r') as fp:
    index_json = json.load(fp)

link_dict = defaultdict(dict)

for sample_id in index_json:
    align_pair = index_json[sample_id]
    for cur_indexs, ctx_indexs in align_pair:
        for cur_index in cur_indexs:
            if cur_index not in link_dict[sample_id]:
                link_dict[sample_id].update({
                    cur_index: []
                })
            for ctx_index in ctx_indexs:
                    link_dict[sample_id][cur_index].append(ctx_index)
    # 
    for cur_index in link_dict[sample_id]:
        link_dict[sample_id][cur_index] = list(set(link_dict[sample_id][cur_index]))

with open(os.path.join(args.save_dir, "link.json"), 'w') as fp:
    fp.write(json.dumps(link_dict))