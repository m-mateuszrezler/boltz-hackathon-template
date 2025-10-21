import gc
import os
import time

import torch

from boltz.model.layers.pairformer import PairformerLayer

# Disable auto-tuning
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

DEVICES = ("cpu", "mps")

# Set hyperparameters
C_S = 384
C_Z = 128
BATCH_SIZE = 1
INFERENCE = False
SEQ_LEN = (128, 256, 384, 512, 768)
COMPILE = False
torch.set_grad_enabled(not INFERENCE)


def fwd(
    model,
    s,
    z,
    mask,
    pair_mask,
    use_cuequiv_mul=False,
    use_cuequiv_attn=False,
):
    model(
        s,
        z,
        mask,
        pair_mask,
        use_cuequiv_mul=use_cuequiv_mul,
        use_cuequiv_attn=use_cuequiv_attn,
    )


def backward(
    model,
    s,
    z,
    mask,
    pair_mask,
    use_cuequiv_mul=False,
    use_cuequiv_attn=False,
):
    s, z = model(
        s,
        z,
        mask,
        pair_mask,
        use_cuequiv_mul=use_cuequiv_mul,
        use_cuequiv_attn=use_cuequiv_attn,
    )
    (s.sum() + z.sum()).backward()


def speed(func, its=10, warmup=10):
    for _ in range(warmup):
        func()
    start = time.time()
    for _ in range(its):
        func()
    time_a = time.time() - start
    time_a /= its
    return time_a


def benchmark(size, device):
    gc.collect()
    torch.mps.empty_cache()

    # Now run the benchmark
    s = torch.randn(
        (BATCH_SIZE, size, C_S),
        device=device,
        requires_grad=False,
    )
    z = torch.randn(
        (BATCH_SIZE, size, size, C_Z),
        device=device,
        requires_grad=False,
    )
    mask = torch.ones(
        (BATCH_SIZE, size),
        device=device,
        requires_grad=False,
    ).float()
    pair_mask = torch.ones(
        (BATCH_SIZE, size, size),
        device=device,
        requires_grad=False,
    ).float()

    fn = fwd if INFERENCE else backward
    ms = speed(
        lambda: fn(
            model,
            s,
            z,
            mask,
            pair_mask,
            use_cuequiv_mul=False,
            use_cuequiv_attn=False,
        )
    )

    # Compute throughput in sequences per second
    return ms / BATCH_SIZE


print("Speed")
for device in DEVICES:
    # Preload modules
    model = PairformerLayer(C_S, C_Z, v2=True)
    model = model.to(device)
    if COMPILE:
        model = torch.compile(model, fullgraph=True, dynamic=False)
    if INFERENCE:
        model.eval()
    print(f"Using device: {device}.")
    for size in SEQ_LEN:
        time_per_sequence = benchmark(size, device)
        print(f"Sequence size: {size}. Time per sequence: {time_per_sequence*1000:.2f} ms.")
