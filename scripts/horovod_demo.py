from mpi4py import MPI

comm = MPI.COMM_WORLD

import os
import tensorflow as tf
import horovod.tensorflow as hvd


# Split COMM_WORLD into subcommunicators
subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                               key=MPI.COMM_WORLD.rank)

# We use the first communicator in the array as global
# and subsequent comms are the various splits we need
comms = [comm, subcomm]

# Initialize Horovod
hvd.init(comm=comms)

print("Hello, all good from", hvd.rank())
r = hvd.rank()

# Setting GPU affinity
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(r+1)


# Let's try to reduce some tensors
a = (r+1)*tf.ones(16)

print("Output of reduction", hvd.alltoall(a, communicator_id=0))

print("Output of second reduction", hvd.alltoall(a, communicator_id=1))

print("get size of world", hvd.size(communicator_id=0))
print("get size of communicator", hvd.size(communicator_id=1))
