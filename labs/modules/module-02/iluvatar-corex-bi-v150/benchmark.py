"""Compare fused kernels with matching in-place storage and restored inputs."""

import argparse
import importlib
import inspect
import random
import statistics

import torch
import triton

from fused_rms_norm import (
    _add_kernel,
    _block_size,
    _fused_add_rms_norm_kernel,
    _rms_norm_kernel,
    torch_add_rms_norm,
)


def import_flaggems():
    """Import the installed FlagGems package; never substitute another kernel."""
    try:
        module = importlib.import_module("flag_gems")
    except Exception as exc:
        raise RuntimeError("Unable to import installed flag_gems; benchmarking stopped.") from exc
    if not hasattr(module, "fused_add_rms_norm"):
        raise RuntimeError("Installed flag_gems does not export fused_add_rms_norm.")
    print(f"Using installed FlagGems package: {module.__file__}")
    print(f"FlagGems implementation: {inspect.getsourcefile(module.fused_add_rms_norm)}")
    return module


def fused_inplace(x, residual, weight, eps):
    """Use the student's forward kernel with the same output aliasing as FlagGems."""
    m, n = x.shape
    _fused_add_rms_norm_kernel[(m,)](
        x, residual, x, residual, weight, n, eps, BLOCK_SIZE=_block_size(n)
    )
    return x, residual


def split_inplace(x, residual, weight, eps):
    """Two-kernel baseline; the intermediate sum retains the input storage dtype."""
    m, n = x.shape
    block_size = _block_size(n)
    _add_kernel[(m,)](x, residual, residual, n, BLOCK_SIZE=block_size)
    _rms_norm_kernel[(m,)](residual, x, weight, n, eps, BLOCK_SIZE=block_size)
    return x, residual


def benchmark(functions, x, residual, source_x, source_residual, warmup, iters, repeats):
    """Time device intervals after restoring inputs, in randomized interleaved order.

    Both fused implementations receive the same working addresses and values.
    Input copies and event initialization are outside the measured intervals.
    The PyTorch baseline can still allocate intermediates and returned outputs.
    """
    rng = random.Random(0)
    names = list(functions)
    times = {name: [] for name in names}

    def restore():
        x.copy_(source_x)
        residual.copy_(source_residual)

    # Compile once before warmup; even this call receives original input values.
    for fn in functions.values():
        restore()
        fn()
    for _ in range(warmup):
        rng.shuffle(names)
        for name in names:
            restore()
            functions[name]()
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iters * len(names))]
    # Initialize lazy device event resources before any measured call.
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()

    for _ in range(repeats):
        samples = {name: [] for name in names}
        scheduled = []
        index = 0
        for _ in range(iters):
            rng.shuffle(names)
            for name in names:
                restore()
                start, end = events[index]
                start.record()
                functions[name]()
                end.record()
                scheduled.append((name, start, end))
                index += 1
        torch.cuda.synchronize()
        for name, start, end in scheduled:
            samples[name].append(start.elapsed_time(end))
        for name in names:
            times[name].append(statistics.mean(samples[name]))
    return times


def throughput_gbs(shape, dtype, ms):
    # Shared logical traffic: two matrix reads, two writes, and weight reads.
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_moved = (4 * shape[0] * shape[1] + shape[1]) * element_size
    return bytes_moved / (ms * 1e-3) / 1e9


def run_case(shape, dtype, eps, warmup, iters, repeats=5):
    torch.manual_seed(0)
    source_x = torch.randn(shape, dtype=dtype, device="cuda")
    source_residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn((shape[-1],), dtype=dtype, device="cuda")
    x, residual = source_x.clone(), source_residual.clone()
    gems = import_flaggems()
    functions = {
        "torch": lambda: torch_add_rms_norm(x, residual, weight, eps),
        "split": lambda: split_inplace(x, residual, weight, eps),
        "fused": lambda: fused_inplace(x, residual, weight, eps),
        "gems": lambda: gems.fused_add_rms_norm(x, residual, (shape[-1],), weight, eps),
    }

    ref_out, ref_residual = torch_add_rms_norm(source_x, source_residual, weight, eps)
    errors = {}
    atol, rtol = (2e-5, 0.0) if dtype == torch.float32 else (0.0625, 0.015625)
    for name in ("fused", "gems"):
        x.copy_(source_x)
        residual.copy_(source_residual)
        out, updated = functions[name]()
        if out.data_ptr() != x.data_ptr() or updated.data_ptr() != residual.data_ptr():
            raise AssertionError(f"{name} must update the supplied buffers in place.")
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(updated, ref_residual, atol=0, rtol=0)
        errors[name] = {
            "out": (out.float() - ref_out.float()).abs().max().item(),
            "residual": (updated.float() - ref_residual.float()).abs().max().item(),
        }
    del ref_out, ref_residual

    times = benchmark(functions, x, residual, source_x, source_residual, warmup, iters, repeats)
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {
        "shape": shape, "dtype": str(dtype).replace("torch.", ""),
        "times_ms": times, "median_ms": medians, "errors": errors,
        "ratio": medians["gems"] / medians["fused"],
        "fused_gbs": throughput_gbs(shape, dtype, medians["fused"]),
        "gems_gbs": throughput_gbs(shape, dtype, medians["gems"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.m, args.n, args.iters, args.repeats) <= 0 or args.warmup < 0:
        parser.error("m, n, iters, and repeats must be positive; warmup must be nonnegative")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    result = run_case((args.m, args.n), dtype, args.eps, args.warmup, args.iters, args.repeats)
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__}")
    print(f"shape={result['shape']} dtype={result['dtype']} eps={args.eps}")
    print(f"warmup={args.warmup} iters={args.iters} repeats={args.repeats}")
    print("Timing: CUDA events; identical inputs restored before EVERY call; input copies excluded.")
    print("Fused comparison: both implementations update the SAME preallocated buffers in place.")
    labels = {
        "torch": "torch stepwise (allocating)", "split": "triton split (in-place)",
        "fused": "triton fused (in-place)", "gems": "FlagGems (in-place)",
    }
    for name, label in labels.items():
        ms = result["median_ms"][name]
        samples = result["times_ms"][name]
        print(f"{label:<28}: {ms:.4f} ms (repeat means {min(samples):.4f}..{max(samples):.4f})")
    print(f"Estimated logical throughput: fused={result['fused_gbs']:.2f}, FlagGems={result['gems_gbs']:.2f} GB/s")
    print(f"fused / FlagGems throughput ratio: {result['ratio'] * 100:.2f}%")
    print(f"Maximum absolute errors: {result['errors']}")
    if result["ratio"] < 0.90:
        raise SystemExit("FAIL: fused kernel is below 90% of FlagGems throughput.")
    print("PASS: numerical checks passed and fused throughput is at least 90% of FlagGems.")


if __name__ == "__main__":
    main()
