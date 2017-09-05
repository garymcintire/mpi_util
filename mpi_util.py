from mpi4py import MPI
import tensorflow as tf
import os, subprocess, sys
import numpy as np
import random
import time

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
	if "parent" == mpi_util.mpi_fork(args.nprocs): sys.exit()

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

9.  You should be able to use multiple gpu cards if you have them (untested))
    with tf.device('/gpu:'+str(mpi_util.rank)):   
        main()

10. You should be able to use multiple computers as well. See mpirun documentation on host files


    This code is for tensorflow, but a few alterations would allow it to work on theano, pytorch, etc

    The actual speedup from parallelizing is most dependent on the batch size. Larger batch sizes will get a closer to
    linear speedup, but even small batche sizes can triple or quadruple your wall clock speed with nprocs = 8


                        Timing
    python ../train.py Walker2d-v1 --nprocs 10 --gpu_pct 0.05  -n 2000

nprocs  steps_per_sec   reward_mean
1            546             641        # reward is highly variable because robot is highly variable
2           1040             544
3           1712             611
4           2014            1220
5           1950            1513
6           2100             860
7           2300             681
8           2400             452
9           2484             359
10          2339             367
11          2384             410
12          2169             336
"""


tf_config = None

nworkers=1      # defaults for a single process
rank= 0

def mpi_fork(n, gpu_pct=0.0):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    global tf_config
    if gpu_pct > 0:
        tf_config = tf.ConfigProto(
            gpu_options =tf.GPUOptions(
                                per_process_gpu_memory_fraction=gpu_pct,
                                # allow_growth=True
                                ),
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
    else:
        tf_config = tf.ConfigProto(
            gpu_options =tf.GPUOptions( allow_growth=True),
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
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
        subprocess.check_call(["mpirun", "-np", str(n),   sys.executable] +['-u']+ sys.argv, env=env)    # this mpirun makes bcast take more time with each iteration
        # subprocess.check_call(["/usr/bin/mpirun", "-np", str(n), '-mca', 'coll_tuned_bcast_algorithm', '0', sys.executable] +['-u']+ sys.argv, env=env)       # this mpirun is 1/3 the speed of the one above
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

def print0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

cumsum_steps=0
tstart = time.time()
def steps_sec(stepcnt):
    global tstart, cumsum_steps
    cumsum_steps += stepcnt
    print0('cumsum_steps', cumsum_steps)   # only for step counting
    etime = time.time()
    # print0(str(rank)+'_steps_sec', stepcnt, 1000*(etime-tstart), stepcnt/(etime-tstart))
    print0(str(rank)+'_steps_sec', stepcnt/(etime-tstart))
    tstart = etime

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
    if nworkers == 1: return segins   # don't bother if only one process
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
    segs = MPI.COMM_WORLD.allgather(segs)
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
    # print(str(rank)+'bcast sees',type(x),x.shape)
    origshape = x.shape
    if 1 or len(x.shape)==1:
        # print(str(rank)+'if True')
        MPI.COMM_WORLD.Bcast(x, root=0)
    else:
        print(str(rank)+'if False   prod',np.prod(x.shape))
        inv = np.reshape(x, (np.prod(x.shape)))
        print(str(rank)+'inv',type(inv),inv.shape,inv.dtype)
        MPI.COMM_WORLD.Bcast(inv, root=0)
        print(str(rank)+'outv',type(inv))
        outv = inv.reshape(origshape)
        print(str(rank)+'outv after reshape',type(outv),outv.shape)
    return x
def xxxbcast(x):
    comm = MPI.COMM_WORLD
    for i in range(1,nworkers):
        if rank == 0:
           comm.send(x, dest=i, tag=11)
        elif rank == i:
           x = comm.recv(source=0, tag=11)
    return x
def pt_to_pt_bcast(x):
    for i in range(1,nworkers):
        if rank == 0:
            MPI.COMM_WORLD.Send(x, dest=i)
        elif rank == i:
            MPI.COMM_WORLD.Recv(x, source=0)
    return x

cache = {}
def set_wt(sess, graph, var, val):
    k = str(var)
    kc = str(sess)+str(graph)+str(var)
    if not kc in cache:
        with graph.as_default():
            ph = tf.placeholder(val.dtype, shape=val.shape)
            op = var.assign(ph)
            cache[kc] = [ var, ph, op]
    _, ph, op = cache[kc]
    sess.run(op, feed_dict={ph: val})

def rank0_bcast_wts(sess, graph, label=''):
    if nworkers == 1: return    # don't bother if only one process
    # chksum(sess, graph, label=label+'before'+str(rank))
    for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
        # timeit(label+'______'+str(var))
        w = sess.run(var)
        # timeit(label+'sess.run'+str(var))
        wts = bcast(w)
        # print(str(rank)+'result type',type(wts))
        # timeit(label+'bcast'+str(var))
        if rank != 0:
            # print(label+'set_wts with type',type(wts), wts.shape, str(wts.__array_interface__['data']),var)
            set_wt(sess, graph, var, wts)
            pass
        # timeit(label+'set_wt'+str(var))
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
        # chksum += wts.sum()
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

lasttime = 0.
def timeit(label):
    global lasttime
    thistime = time.time()
    if rank==0: print(str(rank)+'timed-'+label,thistime-lasttime)
    lasttime = thistime


