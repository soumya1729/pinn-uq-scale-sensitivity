"""Frozen checkpoint model and raw residual definitions, copied verbatim.
Source: original n1_sensitivity.py; no external project import.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

def make_model(params, config, fourier=None):
    """The saved spatial/temporal subnetworks and fusion network in common.py."""
    theta, unpack = ravel_pytree(params)

    def mlp(layers, h):
        for layer in layers[:-1]:
            h = jnp.tanh(h @ layer["w"] + layer["b"])
        return h @ layers[-1]["w"] + layers[-1]["b"]

    def embed(z, key):
        if fourier is None:
            return jnp.atleast_1d(z)
        phase = 2 * jnp.pi * z * fourier[key].ravel()
        return jnp.concatenate((jnp.cos(phase), jnp.sin(phase)))

    def u(vector, x, t):
        p = unpack(vector)
        hx, ht = embed(x, "B_X"), embed(t, "B_T")
        features = [mlp(net, hx) for net in p["spatial"]]
        features += [mlp(net, ht) for net in p["temporal"]]
        return mlp(p["fusion"], jnp.concatenate(features))[0]

    return theta, u

def residual_blocks(u, config, data):
    """Raw training residuals, including the fixed IC and boundary weights."""
    c = config
    ux, ut = jax.grad(u, 1), jax.grad(u, 2)
    uxx = jax.grad(ux, 1)
    if c["pde"] == "burgers":
        def ic(a, x):
            return c["ic_weight"] * (u(a, x, 0.0) + jnp.sin(jnp.pi * x))

        def bc1(a, t):
            return c["bc_weight"] * u(a, -1.0, t)

        def bc2(a, t):
            return c["bc_weight"] * u(a, 1.0, t)

        def pde(a, x, t):
            return ut(a, x, t) + u(a, x, t) * ux(a, x, t) - c["nu"] * uxx(a, x, t)
    elif c["pde"] == "allen_cahn":
        def ic(a, x):
            return c["ic_weight"] * (u(a, x, 0.0) - x**2 * jnp.cos(jnp.pi * x))

        def bc1(a, t):
            return c["bcu_weight"] * (u(a, -1.0, t) - u(a, 1.0, t))

        def bc2(a, t):
            return c["bcx_weight"] * (ux(a, -1.0, t) - ux(a, 1.0, t))

        def pde(a, x, t):
            value = u(a, x, t)
            return ut(a, x, t) - c["eps2"] * uxx(a, x, t) - c["reac"] * value + c["reac"] * value**3
    else:
        raise ValueError("N1 supports the paper's admitted Burgers and Allen-Cahn settings.")
    return [(ic, (data["x_u"],)), (bc1, (data["t_b"],)),
            (bc2, (data["t_b"],)), (pde, (data["x_f"], data["t_f"]))]

def build_jacobian(theta, blocks, chunk):
    n = sum(len(points[0]) for _, points in blocks)
    jacobian = np.empty((n, len(theta)), dtype=np.float64)
    residual = np.empty(n, dtype=np.float64)
    start = 0
    for function, points in blocks:
        evaluate = jax.jit(jax.vmap(jax.value_and_grad(function),
                                  in_axes=(None,) + (0,) * len(points)))
        for q in range(0, len(points[0]), chunk):
            end = min(q + chunk, len(points[0]))
            values, rows = evaluate(theta, *[jnp.asarray(p[q:end]) for p in points])
            residual[start + q:start + end] = np.asarray(values)
            jacobian[start + q:start + end] = np.asarray(rows)
        start += len(points[0])
        print(f"Residual Jacobian: {start}/{n} rows", flush=True)
    return jacobian, residual
