import argparse
import random

import torch


def bench(nt, niter):
    # Warmup
    torch.softmax(nt, -1)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for iter in range(niter):
        torch.softmax(nt, -1)
    end_event.record()
    torch.cuda.synchronize()
    runtime = (start_event.elapsed_time(end_event) * 1.0e-3) / niter
    return runtime


def sweep_n(num_heads, ntensor, niter, dtype):
    print("num_heads, dtype, ntensor, runtime, runtime_t, speedup, padding")
    random.seed(123)
    max_sequence_len = 256
    seq_len_list = [max(0, min(max_sequence_len, int(random.gauss(max_sequence_len / 2, 10)))) for _ in range(ntensor)]
        
    nt = torch.nested.nested_tensor(
        [torch.randn(num_heads, s, s).to(dtype).cuda() for s in seq_len_list]
    )
    runtime = bench(nt, niter)
    t = nt.to_padded_tensor(0.)
    runtime_t = bench(t, niter)
    padding = nt.numel() / t.numel()
    print(num_heads, dtype, ntensor, runtime, runtime_t, runtime_t / runtime, padding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested Tensor BMM Benchmark")
    parser.add_argument("--niter", default="100", type=int)
    parser.add_argument("--ntensor", default="64", type=int)
    parser.add_argument("--num-heads", default="8", type=int)

    args = parser.parse_args()
    niter = args.niter
    ntensor = args.ntensor
    num_heads = args.num_heads

    sweep_n(num_heads, ntensor, niter, torch.float32)
    sweep_n(num_heads, ntensor, niter, torch.float16)
