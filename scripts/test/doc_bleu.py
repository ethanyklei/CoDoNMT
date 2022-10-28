import sacrebleu
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--ref", type=str)
parser.add_argument("--gen", type=str)
parser.add_argument("--delimiter", type=str, default="</s>")
parser.add_argument("--save-dir", type=str)
parser.add_argument("--fix-ctx", action="store_true")

args = parser.parse_args()

def read_file(path, readlines : bool = True):
    with open(path, 'r', encoding='utf-8') as fp:
        if readlines:
            return [line.strip() for line in fp.readlines()]
        else:
            return fp.read()

save_fp = open(args.save_dir + ".sacrebleu", 'w', encoding='utf-8')

ref_corpus = read_file(args.ref)
gen_corpus = read_file(args.gen)

ref_corpus = [line.split(args.delimiter) for line in ref_corpus]
ref_corpus = [list(filter(None, ref)) for ref in ref_corpus]
gen_corpus = [line.split(args.delimiter) for line in gen_corpus]
gen_corpus = [list(filter(None, gen)) for gen in gen_corpus]

assert len(ref_corpus) == len(gen_corpus)

doc_ref_data = []
doc_gen_data = []

sent_ref_data = []
sent_gen_data = []
for i in range(len(ref_corpus)):
    assert len(ref_corpus[i]) == len(gen_corpus[i]), f"{i} \n {ref_corpus[i]} \n {gen_corpus[i]}"

    strip_ref_data = [line.strip() for line in ref_corpus[i]]
    strip_gen_data = [line.strip() for line in gen_corpus[i]]
    
    doc_ref_data.append(" ".join(strip_ref_data))
    doc_gen_data.append(" ".join(strip_gen_data))
    if args.fix_ctx:
        sent_ref_data.append(strip_ref_data[-1])
        sent_gen_data.append(strip_gen_data[-1])
    else:
        sent_ref_data.extend(strip_ref_data)
        sent_gen_data.extend(strip_gen_data)

print(f"Sentence-level BLEU: {sacrebleu.corpus_bleu(sent_gen_data, [sent_ref_data])}", file=save_fp)
print(f"Document-level BLEU: {sacrebleu.corpus_bleu(doc_gen_data, [doc_ref_data])}", file=save_fp)

save_fp.close()


