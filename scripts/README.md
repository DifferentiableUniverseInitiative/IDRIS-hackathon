# Demo and benchmark scripts


## Demos

Here are a few scripts you can use to test your install and see minimal examples
of how to use the various libraries we will be using.

 - [horovod_demo.py](horovod_demo.py): Demonstrates computing all2all with TensorFlow
 tensors over NCCL. To execute:
  ```bash
  $ srun python horovod_demo.py
  ```
 - [fft_benchmark.py](fft_benchmark.py): Runs a series of back and forth distributed FFTs, to get an nvprof trace. To execute:
  ```bash
  $ sbatch fft_benchmark.job
  ```
  You can adjust the size of the cube, how many GPUs to use, what mesh layout to use in the settings of the script. This script should output nvvp traces, that can be loaded into nvvp for analysis.
