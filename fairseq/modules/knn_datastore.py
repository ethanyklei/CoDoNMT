import torch
import faiss
import numpy as np
from torch_scatter import scatter
import time
import math
import os
import faiss.contrib.torch_utils
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("knn_datastore")


class KNN_Datastore(object):

    def __init__(self, args, trg_vocab_size):

        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.use_gpu_to_search = args.use_gpu_to_search
        self.vocab_size = trg_vocab_size
        # self.only_use_max_idx = args.only_use_max_idx

        self.index = self.setup_faiss(args)
        
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        # set lambda
        self.__lambda_value = args.knn_lambda

        # set temperature
        self.__temperature = args.knn_temperature

        # set k
        self.__k = args.k

    def get_lambda(self):

        return self.__lambda_value

    def get_temperature(self):

        return self.__temperature

    def get_k(self):

        return self.__k

    def setup_faiss(self, args):

        if not args.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(os.path.join(args.dstore_dir, 'knn_index'), faiss.IO_FLAG_ONDISK_SAME_DIR)
        if self.use_gpu_to_search:
            logger.info('put index from cpu to gpu')
            res = faiss.StandardGpuResources()
            self.res = res
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        logger.info(f'Reading datastore took {time.time() - start} s')
        logger.info(f'The datastore is {args.dstore_dir}, size is {self.dstore_size}, and dim is {self.dimension}')

        index.nprobe = args.probe

        if args.dstore_fp16:
            logger.info('Keys are fp16 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(os.path.join(args.dstore_dir, 'keys.npy'), dtype=np.float16, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(os.path.join(args.dstore_dir, 'vals.npy'), dtype=np.int32, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            logger.info('Keys are fp32 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(os.path.join(args.dstore_dir, 'keys.npy'), dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))

            self.vals = np.memmap(os.path.join(args.dstore_dir, 'vals.npy'), dtype=np.int32, mode='r',
                                  shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(os.path.join(args.dstore_dir, 'keys.npy'),
                                                  dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(os.path.join(args.dstore_dir, 'vals.npy'),
                                              dtype=np.int32, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int)

            if self.use_gpu_to_search:
                self.vals = torch.from_numpy(self.vals)
                if torch.cuda.is_available():
                    logger.info('put vals to gpu')
                    self.vals = self.vals.cuda()

            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return index

    def dist_func(self, d, k, q, function=None):

        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.__k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2
            return d

        if function == 'dot':
            qsize = q.shape
            return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    def get_knns(self, queries):

        # move query to numpy, if faiss version < 1.6.5
        # numpy_queries = queries.detach().cpu().float().numpy()

        dists, knns = self.index.search(queries, self.__k)

        return dists, knns

    def get_only_max_index(self, prob):
        pass
        # if we do not need a distribution, only use the max prob result
        # if self.only_use_max_idx:
        #     _, max_idx = prob.max(dim=-1)
        #     prob = prob.zero_().scatter_(dim=-1, index=max_idx.unsqueeze(-1), value=1)

    def retrieve(self, queries):

        # queries  are [Batch, seq len, Hid Size]

        # retrieve
        bsz = queries.size(0)
        seq_len = queries.size(1)

        dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)))  # [Batch * seq len, K]
        # move retireval results to torch tensor from numpy, if faiss version < 1.6.5
        # knns = torch.from_numpy(knns).to(queries.device)
        # dists = torch.from_numpy(dists).to(queries.device)  # [Batch size * seq len, k]

        tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)  # [B, S, K]

        dists = dists.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
        knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}


    def calculate_knn_prob(self,
                           knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           ):

        bsz = queries.size(0)
        seq_len = queries.size(1)

        # update the dist and compute each neighbor weight, neg distance
        """
            softmax(-d(q,k)/tem)
        """
        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func)  # [B, S, K]

        scaled_dists = re_compute_dists / self.__temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        # set the target index for each neighbor
        knn_tgt_prob = torch.zeros(bsz, seq_len, self.__k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        # implemented with pytorch_scatter
        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)
        # knn_tgt_prob = knn_tgt_prob.scatter_(dim=-1, index=tgt_index, src=knn_weight.float())
        # print('set the target prob for each neighbor (need do scatter operation for {} tensor), took {} s'.
        #       format(knn_tgt_prob.size(), time.time() - start))

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        # reimplement this with scatter add
        # knn_tgt_prob = torch.zeros(bsz, seq_len, self.vocab_size).to(queries.device)  # [B, S, Vocab Size]
        # tgt_index = tgt_index  # [B, S, K]
        # knn_weight = knn_weight.squeeze(-1)
        # scatter(src=knn_weight, )

        return {'prob': prob}

    def update_get_knn_seq_prob(self, queries):

        knn_search_result = self.retrieve(queries)

        
        final_result = self.calculate_knn_prob(knn_index=knn_search_result['knn_index'],
                                                   tgt_index=knn_search_result['tgt_index'],
                                                   distance=knn_search_result['distance'],
                                                   queries=queries,
                                                   temperature=self.__temperature)

        return {'distance': knn_search_result['distance'],
                    'knn_index': knn_search_result['knn_index'],
                    'prob': final_result['prob'],
                    }
