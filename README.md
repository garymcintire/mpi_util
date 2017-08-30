                                Parallelizing RL batch algorithms
                                            Gary McIntire

    There are multiple ways to do this, but the easiest one(shown here) is to have all 
    processes do their rollouts and then gather those rollouts back to process 0. 
    Let process 0 compute the weight updates, then broadcast those weight updates 
    from rank0 to all the other processes.

    Another example where batches are 'not' accumulated to rank 0 is the trpo_mpi code of the openai 
    baselines https://github.com/openai/baselines/tree/master/baselines/trpo_mpi  where the gradients are 
    averaged together
    That method can require more knowledge of the actual algo whereas accumlating batches to rank0 works for
    almost all batch RL algorithms

1. Make the ConfigProto use as small an amount of memory as possible. It will grow this as needed. 
    Add gpu_options like this ...
    self.sess = tf.Session(graph=self.g, config=mpi_util.tf_config)  # see tf_config in the code below

2. Add mpi_util
	import mpi_util

3. When nprocs is known shortly after program start, fork nprocs...
	if "parent" == mpi_util.mpi_fork(args.nprocs): os.exit()

4. Each process and environment will need random different seeds computed from
    mpi_util.set_global_seeds(seed+mpi_util.rank)
    env.seed(seed+mpi_util.rank)

5. print statements, env monitoring/rendering, etc may need to be conditional such that only 
    one process does it
    if mpi_util.rank == 0: print(...
    it can be useful to prepend print statements with the rank   print( str(mpi_util.rank) + ... )

6. Accumulate the batches with something like
    d = mpi_util.rank0_accum_batches({'advantages': advantages, 'actions': actions, 'observes': observes, 'disc_sum_rew': disc_sum_rew})
    observes, actions, disc_sum_rew, advantages = d['observes'], d['actions'], d['disc_sum_rew'], d['advantages']

7. Since this accumulates the batches to rank0, you can avoid processing weight updates on 
    the other processes
    if mpi_util.rank==0:
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function

8. After updating the weights on rank0, broadcast the weights from rank0 back to all the other processes
    mpi_util.rank0_bcast_wts(val_func.sess, val_func.g, 'val')
    mpi_util.rank0_bcast_wts(policy.sess, policy.g, 'policy')


    This code is for tensorflow, but a few alterations would allow it to work on theano, pytorch, etc

    The actual speedup from parallelizing is most dependent on the batch size. Larger batch sizes 
    will get a closer to linear speedup, but even small batche sizes can triple or quadruple 
    your wall clock speed with nprocs = 10

