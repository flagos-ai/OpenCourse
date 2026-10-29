"""Validate the torch.library operator in eager mode and torch.compile."""

import argparse

import torch

import custom_op  # Registers torch.ops.lab.fused_add_rms_norm on import.
from fused_rms_norm import torch_add_rms_norm


class FusedAddRMSNormModule(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight):
        return torch.ops.lab.fused_add_rms_norm(x, residual, weight, self.eps)


def _run_forward_backward(module, x, residual, weight, grad_out, grad_updated):
    x = x.detach().clone().requires_grad_(True)
    residual = residual.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)

    out, updated = module(x, residual, weight)
    (out * grad_out + updated * grad_updated).sum().backward()
    return {
        "out": out.detach(),
        "updated": updated.detach(),
        "x_grad": x.grad.detach(),
        "residual_grad": residual.grad.detach(),
        "weight_grad": weight.grad.detach(),
    }


def _run_reference(x, residual, weight, grad_out, grad_updated, eps):
    x = x.detach().clone().requires_grad_(True)
    residual = residual.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)

    out, updated = torch_add_rms_norm(x, residual, weight, eps)
    (out * grad_out + updated * grad_updated).sum().backward()
    return {
        "out": out.detach(),
        "updated": updated.detach(),
        "x_grad": x.grad.detach(),
        "residual_grad": residual.grad.detach(),
        "weight_grad": weight.grad.detach(),
    }


def _max_errors(actual, expected):
    return {
        name: (actual[name].float() - expected[name].float()).abs().max().item()
        for name in actual
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--backend",
        choices=("aot_eager", "eager", "inductor"),
        default="eager",
        help="torch.compile backend; eager validates full-graph Dynamo capture.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (args.m, args.n)
    x = torch.randn(shape, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(args.n, device="cuda")
    grad_out = torch.randn_like(x)
    grad_updated = torch.randn_like(x)

    module = FusedAddRMSNormModule(args.eps).cuda()
    eager = _run_forward_backward(module, x, residual, weight, grad_out, grad_updated)
    reference = _run_reference(
        x, residual, weight, grad_out, grad_updated, args.eps
    )

    # fullgraph=True makes a graph break a test failure rather than silently
    # compiling only a fragment of this module.
    compiled_module = torch.compile(module, backend=args.backend, fullgraph=True)
    compiled = _run_forward_backward(
        compiled_module, x, residual, weight, grad_out, grad_updated
    )
    torch.cuda.synchronize()

    eager_errors = _max_errors(eager, reference)
    compiled_errors = _max_errors(compiled, reference)
    explanation = torch._dynamo.explain(module)(x, residual, weight)

    print(f"shape={shape} dtype=float32 eps={args.eps} backend={args.backend}")
    print(
        "Dynamo graph summary: "
        f"graphs={explanation.graph_count}, graph_breaks={explanation.graph_break_count}"
    )
    for name in eager_errors:
        print(
            f"{name:<14} eager_error={eager_errors[name]:.3e} "
            f"compiled_error={compiled_errors[name]:.3e}"
        )

    tolerance = 2e-5
    max_error = max(*eager_errors.values(), *compiled_errors.values())
    if explanation.graph_break_count != 0:
        raise AssertionError("torch.compile introduced a graph break")
    if max_error > tolerance:
        raise AssertionError(
            f"validation failed: {max_error:.3e} exceeds {tolerance:.1e}"
        )
    print("PASS: torch.library operator runs eagerly and under torch.compile.")


if __name__ == "__main__":
    main()
