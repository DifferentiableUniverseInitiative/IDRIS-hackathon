import numpy as np
import os
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

tf.flags.DEFINE_integer("gpus_per_node", 4, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 4, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")
tf.flags.DEFINE_integer("nx", 2, "number of slices along x dim")
tf.flags.DEFINE_integer("ny", 2, "number of slices along x dim")
tf.flags.DEFINE_integer("nz", 1, "number of slices along z dim")
tf.flags.DEFINE_integer("nc", 8, "Size of data cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
FLAGS = tf.flags.FLAGS

def model_fn(nc=64, batch_size=1):
    """
    Example of function implementing a CNN and returning a value.
    """
    # Create the mesh TF graph
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    # Define the named dimensions
    n_block_x = FLAGS.nx #2
    n_block_y = FLAGS.ny #2
    n_block_z = FLAGS.nz #1
    batch_dim = mtf.Dimension("batch", batch_size)
    nx_dim = mtf.Dimension('nx_block', n_block_x)
    ny_dim = mtf.Dimension('ny_block', n_block_y)
    nz_dim = mtf.Dimension('nz_block', n_block_z)
    sx_dim = mtf.Dimension('sx_block', nc//n_block_x)
    sy_dim = mtf.Dimension('sy_block', nc//n_block_y)
    sz_dim = mtf.Dimension('sz_block', nc//n_block_z)

    nc_x_dim = mtf.Dimension('nc_x_dim', nc)
    nc_y_dim = mtf.Dimension('nc_y_dim', nc)
    nc_z_dim = mtf.Dimension('nc_z_dim', 1)

    # image_c_dim = mtf.Dimension('image_c', 3)
    hidden_dim  = mtf.Dimension('h', 4)
    
    # Create some input data
    """
    data = mtf.random_uniform(mesh, [batch_dim, nx_dim, ny_dim, nz_dim,
                                                sx_dim, sy_dim, sz_dim,
                                                image_c_dim])
    
    net = mtf.layers.conv3d_with_blocks(data, hidden_dim,
        filter_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
        d_blocks_dim=nx_dim, h_blocks_dim=ny_dim)
    """

    data = mtf.random_uniform(mesh, [batch_dim, nx_dim, ny_dim, nz_dim])

    net = mtf.layers.dense(data, hidden_dim)

    net = mtf.reduce_sum(net, output_shape=[batch_dim, hidden_dim] )
    return net


def main(_):
    """
    num_tasks = int(os.environ['SLURM_NTASKS'])
    print('num_tasks : ', num_tasks)
    # Resolve the cluster from SLURM environment
    cluster = tf.distribute.cluster_resolver.SlurmClusterResolver({"mesh": num_tasks},
                                                                port_base=8822,
                                                                gpus_per_node=FLAGS.gpus_per_node,
                                                                gpus_per_task=FLAGS.gpus_per_task,
                                                                tasks_per_node=FLAGS.tasks_per_node)
    cluster_spec = cluster.cluster_spec()
    print(cluster_spec)
    # Create a server for all mesh members
    server = tf.distribute.Server(cluster_spec, "mesh", cluster.task_id)
    print(server)
    if cluster.task_id >0:
      server.join()
    # Otherwise we are the main task, let's define the devices
    devices = ["/job:mesh/task:%d/device:GPU:%d"%(i,j) for i in range(cluster_spec.num_tasks("mesh")) for j in range(FLAGS.gpus_per_task)]
    print("List of devices", devices)
    """
    # Defines the mesh structure
    mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
    # layout_rules = [("nx_block","row"), ("ny_block","col")]
    layout_rules = [("nc_x_dim","row"), ("nc_y_dim","col")]
    
    #mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                                mtf.convert_to_layout_rules(layout_rules))
    # Create computational graphs
    net  = model_fn(nc=FLAGS.nc, batch_size=FLAGS.batch_size)
    # Lower mesh computation
    graph = net.graph
    mesh = net.mesh
    lowering = mtf.Lowering(graph, {mesh:mesh_impl})
    # Retrieve output of computation
    result = lowering.export_to_tf_tensor(net)
    # Perform some last processing in normal tensorflow
    out = tf.reduce_mean(result)

    print('Im about to enter the session..')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run(out)
    print("output of computation", r)
    exit(0)


if __name__ == "__main__":
  tf.app.run(main=main)



