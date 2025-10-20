from utils.gpu_utils import gpu_init, kill_interrupted_processes


if __name__ == '__main__':

    # sets GPU ids to use nvidia-smi ordering (CUDA_DEVICE_ORDER = PCI_BUS_ID)
    # finds the gpu with the most free utilization
    # hides all other GPUs so you only use this one (CUDA_VISIBLE_DEVICES = <gpu_id>)
    gpu_id = gpu_init(best_gpu_metric="util") # could also use "mem"

    # run lots of things here ...

    # if you interrupt a process using a GPU but find that, even though `nvidia-smi` no
    # longer shows the process, the memory is still being held, try this
    kill_interrupted_processes(sudo=False)