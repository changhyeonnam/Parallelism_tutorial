import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank,size):
    """
    Simple collective communication.
    collectives allow for communication patterns across all process in a group.
    In order to obtain the sum of all tensors at all processes,
    we can use the dist.all_reduce(tensor, op, group) collective.
    """
    group = dist.new_group([0,1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM,group=group)
    print(f'Rank {rank} has data {tensor[0]}\n')


def init_process(rank,size,fn,backend='gloo'):
    '''
    Initialize the distributed environment
    ensure that every process will be able to coordinate through master using ip,port.
    this function essentially allows processes to communicate with each other by sharing their locations.
    '''
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    dist.init_process_group(backend,rank=rank,world_size=size)
    fn(rank,size)



if __name__ == '__main__':
    '''
    1. spawn two process (which will setup the distributed environment)
    2. initialize the process group
    3. execute given run function
    '''
    size = 2
    processes=[]
    mp.set_start_method('spawn')
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank,size,run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()