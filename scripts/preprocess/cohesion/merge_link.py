import os 
import json
import argparse
from collections import defaultdict
import copy


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--link-dir", type=str, default="/home/yklei/projects/fairseq-dev/log/cohesion/Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--save-dir", type=str, default="/home/yklei/projects/fairseq-dev/log/cohesion/Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--split", type=str, default="train")



    return parser.parse_args()

def read_file(path, readlines: bool = True, load_json: bool = False):
    with open(path, 'r') as fp:
        if readlines:
            return fp.readlines()
        else:
            if load_json:
                return json.load(fp)
            else:
                return fp.read()


if __name__ == "__main__":
    args = get_args()
    
    lexical_link = read_file(os.path.join(args.link_dir, "lexical_cohesion", args.split, "link.json"), readlines=False, load_json=True)
    coreference_link = read_file(os.path.join(args.link_dir, "coreference", args.split, "link.json"), readlines=False, load_json=True)

    sample_ids = list(lexical_link.keys())
    sample_ids.extend(list(coreference_link.keys()))
    sample_ids = [int(id) for id in sample_ids]
    sample_ids = sorted(list(set(sample_ids)))
    

    merge_result = defaultdict(dict)
    for sample_id in sample_ids:
        if str(sample_id) in coreference_link and str(sample_id) in lexical_link:
            merge_result[sample_id] = copy.copy(coreference_link[str(sample_id)])
            for word_idx in lexical_link[str(sample_id)]:
                if word_idx in merge_result[sample_id]:
                    tmp = merge_result[sample_id][word_idx]
                    tmp.extend(lexical_link[str(sample_id)][word_idx])
                    tmp = list(set(tmp))
                    merge_result[sample_id][word_idx] = tmp
                else:
                    merge_result[sample_id][word_idx] = copy.copy(lexical_link[str(sample_id)][word_idx])
        elif str(sample_id) in coreference_link:
            merge_result[sample_id] = copy.copy(coreference_link[str(sample_id)])
        elif str(sample_id) in lexical_link:
            merge_result[sample_id] = copy.copy(lexical_link[str(sample_id)])
        else:
            print(f"unknow state {sample_id}")
    
    with open(os.path.join(args.save_dir, 'link.json'), 'w') as fp:
        json.dump(merge_result, fp)
        



