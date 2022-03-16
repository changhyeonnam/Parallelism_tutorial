import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank,size):
    """ Distributed function to be implemented later. """
    print(f'rank:{rank}, size:{size}')



def run_point2point(rank,size,block=True):
    """
    point-to-point communication: transfer of data from one process to another.
    recv,send functions are blocking funciton. both process stop untial the communication is completed.
    on the other, irecv, isend are non-blocking.
    """
    tensor = torch.zeros(1)
    if block==True:
        if rank==0:
            tensor+=1
            # send the tensor to process 1
            print(f'#{rank} process sends tensor')
            dist.send(tensor=tensor,dst=1)
        else:
            # receive tensor from process 0
            dist.recv(tensor=tensor,src=0)
            print(f'#{rank} process receives tensor')
    else:
        if rank==0:
            tensor+=1
            # send the tensor to process 1
            print(f'#{rank} process sends tensor')
            req = dist.isend(tensor=tensor,dst=1)
        else:
            # receive tensor from process 0
            req = dist.irecv(tensor=tensor,src=0)
            print(f'#{rank} process receives tensor')
        req.wait()
    print(f'rank:{rank}, has data {tensor[0]}\n')


def init_process(rank,size,fn,block,backend='gloo'):
    '''
    Initialize the distributed environment
    ensure that every process will be able to coordinate through master using ip,port.
    this function essentially allows processes to communicate with each other by sharing their locations.
    '''
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    dist.init_process_group(backend,rank=rank,world_size=size)
    fn(rank,size,block)



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
        block=True
        p = mp.Process(target=init_process, args=(rank,size,run_point2point,block))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()