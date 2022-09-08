import argparse
import logging
import os

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("sample_builder")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/yklei/data/nfs/data-prep/SAC-IWSLT2017-sent/tmp")
    parser.add_argument("--save-dir", type=str, default="/home/yklei/data/nfs/data-prep/SAC-IWSLT2017-doc-3")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-sents", type=int, default=1000)
    parser.add_argument("--src-lang", type=str, default='en')
    parser.add_argument("--tgt-lang", type=str, default='de')
    parser.add_argument("--min-doc-len", type=int, default=1)
    parser.add_argument("--ctx-num", type=int, default=-1)
    parser.add_argument("--sent",action="store_true")


    return parser


def load_file(path, readline : bool = True):
    with open(path, 'r', encoding='utf-8') as fp:
        if readline:
            return [line.strip() for line in fp.readlines()]
        else:
            return fp.read()
    
def write_file(content, path):
    with open(path, 'w', encoding='utf-8') as fp:
        if isinstance(content, list):
            for line in content:
                fp.write(line.strip() + "\n")
        else:
            fp.write(content)

def _doc_to_sents(doc):
    sents = doc.split("</s>")
    sents = [sent.strip() for sent in sents]
    sents = list(filter(None, sents))
    return sents 

def _build_doc_sample(args, split):
    src_file = os.path.join(args.data_dir, f'{split}.{args.src_lang}')
    tgt_file = os.path.join(args.data_dir, f'{split}.{args.tgt_lang}')

    src_lines = load_file(src_file)
    tgt_lines = load_file(tgt_file)

    assert len(src_lines) == len(tgt_lines)

    src_data = []
    tgt_data = []
    for idx, (src_doc, tgt_doc) in enumerate(zip(src_lines, tgt_lines)):
        src_sents = _doc_to_sents(src_doc)
        tgt_sents = _doc_to_sents(tgt_doc)

        assert len(src_sents) == len(tgt_sents)
        if len(src_sents) < args.min_doc_len:
            logger.warning(f"Ingnoring too short document: split={split}, doc={idx}, sents={len(src_sents)}")
            continue

        segment = []
        if args.ctx_num != -1:
            # use fix previous context
            for idx in range(len(src_sents) - args.ctx_num - 1):
                segment.append([
                    0,
                    0,
                    src_sents[idx : idx + args.ctx_num + 1],
                    tgt_sents[idx : idx + args.ctx_num + 1]
                ])
        else:
            # doc2doc
            for idx, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
                assert len(src_sent) > 0 and len(tgt_sent) > 0
                max_tok = max(len(src_sent.split()), len(tgt_sent.split()))
                if len(segment) > 0 and (segment[-1][0] + max_tok < args.max_tokens) and (segment[-1][1] < args.max_sents):
                    segment[-1][0] += max_tok
                    segment[-1][1] += 1
                    segment[-1][2].append(src_sent)
                    segment[-1][3].append(tgt_sent)
                else:
                    segment.append([max_tok, 1, [src_sent], [tgt_sent]])
    
        for idx, seg in enumerate(segment):
            src_seg = " </s> ".join(segment[idx][2])
            tgt_seg = " </s> ".join(segment[idx][3])
            src_data.append(src_seg)
            tgt_data.append(tgt_seg)
    assert len(src_data) == len(tgt_data)
    return src_data, tgt_data   

def _build_sent_sample(args, split):
    src_file = os.path.join(args.data_dir, f'{split}.{args.src_lang}')
    tgt_file = os.path.join(args.data_dir, f'{split}.{args.tgt_lang}')

    src_lines = load_file(src_file)    
    tgt_lines = load_file(tgt_file)
    
    assert len(src_lines) == len(tgt_lines)

    src_data = []
    tgt_data = []
    for idx, (src_doc, tgt_doc) in enumerate(zip(src_lines, tgt_lines)):
        src_sents = _doc_to_sents(src_doc)
        tgt_sents = _doc_to_sents(tgt_doc)

        assert len(src_sents) == len(tgt_sents)

        src_data.extend(src_sents)
        tgt_data.extend(tgt_sents)

    return src_data, tgt_data



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logger.info(args)

    for split in ['train', 'test', 'valid']:
        if args.sent:
            src_data, tgt_data = _build_sent_sample(args, split )
        else: 
            src_data, tgt_data = _build_doc_sample(args, split)

        src_save_file = os.path.join(args.save_dir, f"{split}.{args.src_lang}")
        tgt_save_file = os.path.join(args.save_dir, f"{split}.{args.tgt_lang}")
        write_file(src_data, src_save_file)
        write_file(tgt_data, tgt_save_file)




