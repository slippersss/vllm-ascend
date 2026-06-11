from torch_npu.profiler.profiler import analyse


if __name__ == "__main__":
    analyse(profiler_path="/home/l00957369/data", max_process_number=128)
