from mpi4py import MPI
import time
import os
# Pin only one GPU per horovod process
comm = MPI.COMM_WORLD
#os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(comm.rank+1) # This is specific to my machine
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl


# IDRIS allocation
"""
4 GPUs
salloc --nodes=1 --ntasks-per-node=4 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread -A ftb@gpu --reservation=hackathon_idr25
module load tensorflow-gpu/py3/2.4.1+nccl-2.8.3-1
srun --unbuffered --mpi=pmi2 /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u mtf_random.py
"""

# Define the input seed
input_seed = 0

# We create a small mesh
graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")
batch_dim = mtf.Dimension("batch", 2)
nx_dim = mtf.Dimension('nx_block', 4)
ny_dim = mtf.Dimension('ny_block', 4)
# Defines the mesh structure
mesh_shape = [ ("row", 2), ("col", 2)]
# layout_rules = [('batch', 'row'),  ("nx_block","col")]
layout_rules = [('batch', 'row')]

# mesh_shape = [ ("row", 2)]
# layout_rules = [('batch', 'row')]


# kwargs={'maxval':1, 'seed':input_seed}
kwargs={'maxval':1}
# data_same = mtf.random_uniform(mesh, [ny_dim], **kwargs)
data = mtf.random_uniform(mesh, [batch_dim, nx_dim, ny_dim], **kwargs)

mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                            mtf.convert_to_layout_rules(layout_rules))

lowering = mtf.Lowering(graph, {mesh:mesh_impl})

res = lowering.export_to_tf_tensor(data)
# res_same = lowering.export_to_tf_tensor(data_same)

# Execute and retrieve result
with tf.Session() as sess:
    r = sess.run(res)
    # r_same = sess.run(res_same)

print("Final shape", r.shape)
print("Final result", r)
# print("Final result_same", r_same)

