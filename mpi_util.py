from mpi4py import MPI
import tensorflow as tf
import os, subprocess, sys
import numpy as np
import random

"""

                                Parallelizing RL batch algorithms
                                        Gary McIntire

    There are multiple ways to do this, but the easiest one(shown here) is to have all processes do their rollouts and then gather
    those rollouts back to process 0. Let process 0 compute the weight updates, then broadcast those weight updates
    from rank0 to all the other processes.

    Another example where batches are 'not' accumulated to rank 0 is the trpo_mpi code of the openai baselines
    https://github.com/openai/baselines/tree/master/baselines/trpo_mpi  where the gradients are averaged together
    That method can require more knowledge of the actual algo whereas accumlating batches to rank0 works for
    almost all batch RL algorithms

1. Make the ConfigProto use as small an amount of memory as possible. It will grow this as needed. Add gpu_options like this
    self.sess = tf.Session(graph=self.g, config=mpi_util.tf_config)  # see tf_config in the code below

2. Add mpi_util
	import mpi_util

3. When nprocs is known shortly after program start, fork nprocs...
	if "parent" == mpi_util.mpi_fork(args.nprocs): os.exit()

4. Each process and environment will need random different seeds computed from
    mpi_util.set_global_seeds(seed+mpi_util.rank)
    env.seed(seed+mpi_util.rank)

5. print statements, env monitoring/rendering, etc may need to be conditional such that only one process does it
    if mpi_util.rank == 0: print(...
    it can be useful to prepend print statements with the rank   print( str(mpi_util.rank) + ... )

6. Accumulate the batches with something like
    d = mpi_util.rank0_accum_batches({'advantages': advantages, 'actions': actions, 'observes': observes, 'disc_sum_rew': disc_sum_rew})
    observes, actions, disc_sum_rew, advantages = d['observes'], d['actions'], d['disc_sum_rew'], d['advantages']

7. Since this accumulates the batches to rank0, you can avoid processing weight updates on the other processes
    if mpi_util.rank==0:
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function

8. After updating the weights on rank0, broadcast the weights from rank0 back to all the other processes
    mpi_util.rank0_bcast_wts(val_func.sess, val_func.g, 'val')
    mpi_util.rank0_bcast_wts(policy.sess, policy.g, 'policy')


    This code is for tensorflow, but a few alterations would allow it to work on theano, pytorch, etc

    The actual speedup from parallelizing is most dependent on the batch size. Larger batch sizes will get a closer to
    linear speedup, but even small batche sizes can triple or quadruple your wall clock speed with nprocs = 10

"""


batches_or_update = 1   # This is just a flag to indicate we are accumulating batches.  accum batches or update process wts in parallel

tf_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.05, allow_growth=True),
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)

nworkers=1      # defaults for a single process
rank= 0

def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def accum_batches(segs):
    combo = dict()
    for k in segs[0].keys():
        if type( segs[0][k ]) is list:
            # print(k,'is list')
            combo[k] = sum([segs[i][k] for i in range(len(segs))], [])
        elif type( segs[0][k ]) is np.float64:
            # print(k,'is float64')
            pass    # ignore nextvpred as it is not used in further code
        else:   # must be a numpy.ndarray
            # print(k,type(segs[0][k ]))
            combo[k] = np.concatenate([segs[i][k] for i in range(len(segs))])
    pass
    return combo

def oldrank0_accum_batches(segsin):
    segs = MPI.COMM_WORLD.gather(segsin)
    if rank != 0: return segsin
    return accum_batches(segs)

def mpitype(typ):
    if typ == np.float64:
        return MPI.DOUBLE
    elif typ == np.float32:
        return MPI.FLOAT
    else: return None

def rank0_accum_batches(segins):
    combo = dict()
    for k in sorted(segins):
        seq = segins[k]
        recbuf = None
        counts = MPI.COMM_WORLD.gather(seq.shape[0], root=0)
        if rank==0: Nrows = sum(counts)
        if rank==0:
            if len(seq.shape) == 2:
                recbuf = np.zeros((Nrows, seq.shape[-1]),dtype=seq.dtype)
            else:
                recbuf = np.zeros(Nrows,dtype=seq.dtype)
        if len(seq.shape) == 2:
            rowtype = mpitype(seq.dtype).Create_contiguous(seq.shape[-1])
        else:
            rowtype = mpitype(seq.dtype)
        rowtype.Commit()
        # print(str(rank)+'rank key',k,seq.shape, recbuf.shape if rank==0 else -1111)
        MPI.COMM_WORLD.Gatherv(sendbuf=[seq, mpitype(seq.dtype)] , recvbuf=[recbuf, (counts, None), rowtype], root=0)
        if rank==0: combo[k] = recbuf[:Nrows]
        pass
    if rank==0: return combo
    else: return segins

def all_accum_batches(segs):
    segs = MPI.COMM_WORLD.Allgather(segs)
    return accum_batches(segs)

def all_sum(x):
    assert isinstance(x, np.ndarray)
    out = np.empty_like(x)
    MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
    # MPI.COMM_WORLD.Barrier()
    return out

def all_mean(x):
    return all_sum(x) / nworkers

def bcast(x):
    return MPI.COMM_WORLD.bcast(x, root=0)

cache = {}
def set_wt(sess, graph, var, val):
    k = str(var)
    kc = str(sess)+str(graph)+str(var)
    if not kc in cache:
        with graph.as_default():
            ph = tf.placeholder(val.dtype, shape=val.shape)
            op = var.assign(ph)
            cache[k] = [ var, ph, op]
    _, ph, op = cache[k]
    sess.run(op, feed_dict={ph: val})

def rank0_bcast_wts(sess, graph, label=''):
    if nworkers == 1: return    # don't bother if only one process
    # for var in graph.get_collection('trainable_variables'):
    # chksum(sess, graph, label=label+'before'+str(rank))
    for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
        # print('rank:',rank,'label',label,'var',var)
        # tf.assign(var, all_mean(sess.run(var)))    suspected memory leaker
        wts = bcast(sess.run(var))
        if rank != 0:
            set_wt(sess, graph, var, wts)
            # sess.run(var, feed_dict={var: wts})
    # chksum(sess, graph, label=label+'after'+str(rank))
    # print('rank:',rank, 'nworkers',nworkers, 'label',label,'chksum', chksum)

def all_avg_wts(sess, graph, label=''):
    if nworkers == 1: return    # don't bother if only one process
    # for var in graph.get_collection('trainable_variables'):
    # print('label in all_avg_wts',label)
    chksum = 0
    for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
        # print('rank:',rank,'label',label,'var',var)
        # tf.assign(var, all_mean(sess.run(var)))    suspected memory leaker
        wts = all_mean(sess.run(var))
        chksum += wts.sum()
        set_wt(sess, graph, var, wts)
        # sess.run(var, feed_dict={var: wts})
    # print('rank:',rank, 'nworkers',nworkers, 'all_avg_wts label',label,'chksum', chksum)

def chksum(sess, graph, label=''):
    # if nworkers == 1: return    # don't bother if only one process
    # for var in graph.get_collection('trainable_variables'):
    print('label in chksum',label)
    chksum = 0
    for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
        # print('rank:',rank,'label',label,'var',var)
        # tf.assign(var, all_mean(sess.run(var)))    suspected memory leaker
        wts = sess.run(var)
        chksum += wts.sum()
        sess.run(var, feed_dict={var: wts})
    print('CHKSUM rank:',rank, 'nworkers',nworkers, 'label',label,'chksum', chksum)


