# Guide to get started on the Hackathon

In this document, we list all of the steps for getting setup for the hackathon,
and how to run basic demos and scripts

## Setting up your environment

For this project we will use the following modules and TensorFlow environment
on the Jean-Zay machine:
```bash
module load cmake
module load tensorflow-gpu/py3/2.4.1+nccl-2.8.3-1
```

Executing these lines should load the correct environment in which we will
install additional dependencies as we need them.

### Compiling modified Horovod

We will be working on our [customized fork of Horovod](https://github.com/DifferentiableUniverseInitiative/horovod/tree/multiple_communicators) (see [here](https://github.com/DifferentiableUniverseInitiative/horovod/pull/2)
  for more details on what is different in this fork compared to upstream)

To compile this version of Horovod in the environment loaded above, you can
follow this procedure:

```bash
git clone --recursive https://github.com/DifferentiableUniverseInitiative/horovod.git
cd horovod
git checkout multiple_communicators
export HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_PYTORCH=1
pip install --user .
```

This procedure is if you want to install without modifying the code, if you want to install it in "develop"
mode, replace the last line by:
```bash
pip install --user -e .
```
This will fail, because the compile script of horovod actually doesnt like it if we dont compile mxnet and pytorch support...
To fix the issue, do the following:
```bash
mkdir -p build/lib.linux-x86_64-3.7/horovod/mxnet
touch build/lib.linux-x86_64-3.7/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so
mkdir -p build/lib.linux-x86_64-3.7/horovod/torch
touch build/lib.linux-x86_64-3.7/horovod/torch/mpi_lib_v2.cpython-37m-x86_64-linux-gnu.so
cd horovod
ln -s ../build/lib.linux-x86_64-3.7/horovod/metadata.json .
cd ..
```
you should be back in the root horovod folder at that point, and you can run again `pip install --user -e .` this time it should be faster
and not complain at the end.

The compilation itself should take a few minutes, and then horovod should be
accessible in your conda environment.

### Installing modified Mesh TensorFlow

We will be working on a [modified version of Mesh TensorFlow](https://github.com/DifferentiableUniverseInitiative/mesh/tree/hvd), which uses our modified version
of Horovod.

To install this fork of Mesh TensorFlow, follow this procedure:

```bash
git clone https://github.com/DifferentiableUniverseInitiative/mesh.git
cd mesh
git checkout hvd
pip install --user -e .
```

And that's it :-)

### Installing FlowPM

Finally, you can install FlowPM with the following:

```bash
git clone https://github.com/DifferentiableUniverseInitiative/flowpm.git
cd flowpm
git checkout mesh_update
pip install --user -e .
```

You should be all set.

## Testing your install with demos

After following the steps of the previous section, you should be able to run the
demos presented here, which show you a few minimal examples.

### Starting an interactive job with multiple GPUs

For debugging purposes, it may be very convenient to have multi-gpu interactive environment, you can
start one with:
```bash
$ salloc --nodes=1 --ntasks-per-node=4 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread -A ftb@gpu
```
But please don't let them run for hours idling, they burn GPU time.


### Running a minimal Horovod example

The [scripts/horovod_demo.py](scripts/horovod_demo.py) script contains a simple
example of using the  all2all collective in Horovod from TensorFlow. To execute
it on 4 GPUs in the environment declared above, you can use the following:
```bash
$ srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python horovod_demo.py 
```

**TODO**: Figure out how to get timelines with srun. 
This job will create a timeline file which you can inspect to see a little bit
the trace of the horovod communications. To see how to read these files, checkout
the horovod doc [here](https://horovod.readthedocs.io/en/stable/timeline_include.html).

Interesting environment variables for getting more info from NCCL and see if
everything is working well:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=nccl_log
export NCCL_DEBUG_SUBSYS=ALL
```
This will export a `nccl_log` file with traces of communications

You can look at the traffic between the GPUs on a single node by running the following command several times on that node:
```
nvidia-smi nvlink -gt d
```

That should display:
```
GPU 0: Tesla V100-SXM2-16GB (UUID: GPU-3bd00c3b-0777-ff8a-af75-76a125f9076f)
         Link 0: Data Tx: 5767281335 KiB
         Link 0: Data Rx: 5236186453 KiB
         Link 1: Data Tx: 5767301856 KiB
         Link 1: Data Rx: 5236363958 KiB
         Link 2: Data Tx: 5810435158 KiB
         Link 2: Data Rx: 5268354895 KiB
         Link 3: Data Tx: 6027545425 KiB
         Link 3: Data Rx: 5339549892 KiB
         Link 4: Data Tx: 5810695002 KiB
         Link 4: Data Rx: 5268581753 KiB
         Link 5: Data Tx: 6027408899 KiB
         Link 5: Data Rx: 5343167491 KiB
GPU 1: Tesla V100-SXM2-16GB (UUID: GPU-69aefc2f-736e-00f1-a539-c2eb78a6ff77)
         Link 0: Data Tx: 5339549892 KiB
         Link 0: Data Rx: 6027545425 KiB
         Link 1: Data Tx: 5470084557 KiB
         Link 1: Data Rx: 5345951114 KiB
         Link 2: Data Tx: 5313009161 KiB
         Link 2: Data Rx: 5309225642 KiB
         Link 3: Data Tx: 5313023346 KiB
         Link 3: Data Rx: 5309069286 KiB
         Link 4: Data Tx: 5343167491 KiB
         Link 4: Data Rx: 6027408899 KiB
         Link 5: Data Tx: 5470567305 KiB
         Link 5: Data Rx: 5342288442 KiB
etc. (each node has 6 links with its 3 neighbours in this example)
```

### Running 3D FFT benchmark

The [scripts/fft_benchmark.py](scripts/fft_benchmark.py) script contains a simple code that will run a series of back and forth 3D FFTs. The associate slurm script 
will run this code under nvprof to collect a trace of the execution that we can then
analyse.

To launch the script:
```bash
$ sbatch fft_benchmark.job
```
This will run a distributed FFT of size 512^3 over 8 GPUs on 2 nodes. 

To load the trace, downloaded to your local computer, and open it in nvvp.


### Running FlowPM simulation

The FlowPM repo contains an example script for running a full N-body simulation in mesh-tensorflow.
You can find it [here](https://github.com/DifferentiableUniverseInitiative/flowpm/tree/mesh_update/scripts). To run the following script
you need to be in the `flowpm/scripts` directory:
```bash
sbatch mesh_nbody_benchmark_idris.sh
```
by default, this should run a 512^3 simulation distributed on 4 nodes, and output a png file showing the initial
and final conditions of the simulation.

