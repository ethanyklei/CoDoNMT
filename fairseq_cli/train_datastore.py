import argparse
from dataclasses import replace
import os
from random import random
from turtle import shape
import numpy as np
import faiss
import time
import logging

logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
logger = logging.getLogger("train datastore")


parser = argparse.ArgumentParser()
parser.add_argument('--dstore-dir', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore-size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
parser.add_argument('--faiss-index', type=str, help='file to write the faiss index')
parser.add_argument('--add-stride', default=500000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting-point', type=int, default=0, help='index to start adding keys at')

parser.add_argument('--concat-multiple-files', default=False, action='store_true')
parser.add_argument('--multiple-key-files', type=str, default=None)
parser.add_argument('--multiple-val-files', type=str, default=None)
parser.add_argument('--multiple-files-size', type=str, default=None)
parser.add_argument('--concat-file-path', type=str, default=None)

args = parser.parse_args()

logger.info(args)

res = faiss.StandardGpuResources()


# concat multiple memory map files
if args.dstore_fp16:
    if args.concat_multiple_files:
        assert args.multiple_key_files is not None and args.multiple_val_files is not None
        key_files = args.multiple_key_files.split(',')
        val_files = val_files = args.multiple_val_files.split(',')
        sizes = [int(size) for size in args.multiple_files_size.split(',')]

        key_list = [np.memmap(key_file, dtype=np.float16, mode='r', shape=(sizes[idx], args.dimension))
                    for idx, key_file in enumerate(key_files)]
        val_list = [np.memmap(val_file, dtype=np.int32, mode='r', shape=(sizes[idx], args.dimension))
                    for idx, val_file in enumerate(val_files)]
        concat_size = np.sum(sizes)

        keys = np.memmap(os.path.join(args.concat_file_path, 'keys.npy'), dtype=np.float16, 
                        mode='w+', shape=(concat_size, args.dimension))
        vals = np.memmap(os.path.join(args.concat_file_path, 'vals.npy'), dtype=np.int32,
                        mode="w+", shape=(concat_size, 1))

        cur_size = 0
        for idx, size in enumerate(sizes):
            logger.info(f'write {cur_size} to {cur_size + size}')
            keys[cur_size: cur_size + size, :] = key_list[idx][:, :]
            vals[cur_size: cur_size + size, :] = val_list[idx][:, :]
            cur_size += size
        
        logger.info("finish concat, exit program")
        exit()

# load keys and vals
if args.dstore_fp16:
    logger.info(f"[FP16] load dstore from {args.dstore_dir} (size={args.dstore_size}, dim={args.dimension})")
    keys = np.memmap(os.path.join(args.dstore_dir, 'keys.npy'), dtype=np.float16, mode='r',
                    shape=(args.dstore_size, args.dimension))
    vals = np.memmap(os.path.join(args.dstore_dir, 'vals.npy'), dtype=np.int32, mode='r',
                    shape=(args.dstore_size, 1)) 
else:
    logger.info(f"[FP32] load dstore from {args.dstore_dir} (size={args.dstore_size}, dim={args.dimension})")
    keys = np.memmap(os.path.join(args.dstore_dir, 'keys.npy'), dtype=np.float32, mode='r',
                    shape=(args.dstore_size, args.dimension))
    vals = np.memmap(os.path.join(args.dstore_dir, 'vals.npy'), dtype=np.int32, mode='r',
                    shape=(args.dstore_size, 1)) 


if not os.path.exists(args.faiss_index + ".trained"):
    # initialize faiss index
    quantizer = faiss.IndexFlatL2(args.dimension)
    index = faiss.IndexIVFPQ(quantizer, args.dimension, args.ncentroids, args.code_size, 8)

    index.nprobe = args.probe

    logger.info("Start put index to gpu")
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

    logger.info("Training index")
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    start_time = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    gpu_index.train(keys[random_sample].astype(np.float32))
    logger.info(f"Training took {time.time() - start_time}'s")

    logger.info("Writting index after training")
    start_time = time.time()
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index + ".trained")
    logger.info(f"Writing index took {time.time() - start_time}'s")

logger.info("Adding keys")
index = faiss.read_index(args.faiss_index + ".trained")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
start = args.starting_point
start_time = time.time()
while start < args.dstore_size:
    end = min(args.dstore_size, start + args.add_stride)
    logger.info(f"{start}-{end}")
    to_add = keys[start:end].copy()
    gpu_index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
    start += args.add_stride

    if (start % 1000000) == 0:
        logger.info(f"Added {start} tokens so far")
        logger.info(f"Writing index {start}")
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index)

logger.info(f"Add total {end} keys")
logger.info(f"Adding took {time.time() - start_time}'s")
logger.info("Writing Index")
start_time = time.time()
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index)
logger.info(f"Writing index took {time.time() - start_time}'s")