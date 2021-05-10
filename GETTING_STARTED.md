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
This will fail, because the compile script of horovod actually doesnt like it
  if we dont compile mxnet and pytorch support... right now I have to manually create
  empty files `build/lib.linux-x86_64-XXXX/horovod/mxnet/mpi_lib.cpython-38-x86_64-linux-gnu.so`,
  `build/lib.linux-x86_64-XXXX/horovod/torch/mpi_lib_v2.cpython-38-x86_64-linux-gnu.so` which is a bit annoying,
  and also create a symlink of the metadata.json file found under `build/lib.linux-x86_64-XXXX/horovod`


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

### Running 3D FFT benchmark

**TODO**: add description here

### Running FlowPM simulation

**TODO**: add description here
