import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank,size):
    print(f'rank:{rank}, size:{size}')

def init_process(rank,size,fn,backend='gloo'):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    dist.init_process_group(backend,rank=rank,world_size=size)
    fn(rank,size)


'''
1. spawn two process (which will setup the distributed environment)
2. initialize the process group
3. execute given run function
'''
if __name__ == '__main__':
    size = 2
    processes=[]
    mp.set_start_method('spawn')
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank,size,run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()