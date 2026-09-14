"""Unit checks for weight-only fake-quant (models/quant.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models import quant

jax.config.update("jax_platform_name", "cpu")

SCHEMES = sorted(quant.WEIGHT_QUANT_DTYPES)


def _rand(shape, seed=0):
    return jax.random.normal(jax.random.key(seed), shape, dtype=jnp.float32)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_fake_quantize_is_a_fixed_point(scheme):
    dtype = quant.WEIGHT_QUANT_DTYPES[scheme]
    x = _rand((8, 64))
    once = quant.fake_quantize(x, dtype)
    twice = quant.fake_quantize(once, dtype)
    # Fixed point up to float32 ulp noise in the rescaling.
    np.testing.assert_allclose(np.asarray(once), np.asarray(twice), atol=1e-6)


@pytest.mark.parametrize("scheme,bound_scale", [("int8", 0.5 / 127), ("int4", 0.5 / 7)])
def test_per_channel_error_bound(scheme, bound_scale):
    dtype = quant.WEIGHT_QUANT_DTYPES[scheme]
    x = _rand((8, 64))
    err = jnp.abs(quant.fake_quantize(x, dtype) - x)
    bound = jnp.abs(x).max(axis=-1, keepdims=True) * bound_scale
    assert bool((err <= bound + 1e-7).all())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_absmax_is_exactly_representable(scheme):
    dtype = quant.WEIGHT_QUANT_DTYPES[scheme]
    x = _rand((4, 32))
    q = quant.fake_quantize(x, dtype)
    peak = jnp.abs(x).argmax(axis=-1)
    rows = jnp.arange(x.shape[0])
    np.testing.assert_allclose(
        np.asarray(q[rows, peak]), np.asarray(x[rows, peak]), rtol=1e-6)


def test_group_scales_contain_outlier_damage():
    # One huge outlier per row: per-channel scaling flattens everything else
    # to zero at int4; group scales confine the damage to the outlier's group.
    x = _rand((4, 256))
    x = x.at[:, 0].set(1e4)
    dtype = quant.WEIGHT_QUANT_DTYPES["int4"]
    per_channel_err = jnp.abs(quant.fake_quantize(x, dtype) - x)[:, 32:]
    grouped_err = jnp.abs(quant.fake_quantize(x, dtype, group_size=32) - x)[:, 32:]
    assert float(grouped_err.mean()) < 0.2 * float(per_channel_err.mean())


def test_group_size_must_divide_contraction():
    with pytest.raises(AssertionError):
        quant.fake_quantize(_rand((4, 60)), jnp.int8, group_size=32)


def test_int_bit_width_schemes_interpolate():
    # intN via weight_quant_spec: error strictly decreases with bit width and
    # int8-as-bits matches the int8 dtype path.
    x = _rand((8, 128))
    errs = [
        float(jnp.abs(quant.fake_quantize(x, quant.weight_quant_spec(f"int{b}")) - x).mean())
        for b in (4, 5, 6, 8)
    ]
    assert errs == sorted(errs, reverse=True)
    via_dtype = quant.fake_quantize(x, jnp.int8)
    via_bits = quant.fake_quantize(x, 8)
    np.testing.assert_allclose(np.asarray(via_dtype), np.asarray(via_bits), atol=1e-6)


@pytest.mark.parametrize("scheme", SCHEMES + ["int6", "int5"])
def test_fake_quantize_is_not_a_noop_under_jit(scheme):
    # Regression: XLA's GPU simplifier folds narrow-float astype round-trips
    # to the identity, silently disabling fp8/fp4 fake-quant inside jit.
    spec = quant.weight_quant_spec(scheme)
    x = _rand((8, 64)).astype(jnp.bfloat16)
    q = jax.jit(lambda v: quant.fake_quantize(v, spec, group_size=32))(x)
    err = float(jnp.abs(q.astype(jnp.float32) - x.astype(jnp.float32)).mean())
    assert err > 1e-3, f"{scheme} fake-quant is a no-op under jit"


def test_snap_float_matches_e2m1_grid():
    # Every snapped magnitude must land exactly on the e2m1 grid.
    x = _rand((4, 64))
    q = quant.fake_quantize(x, jnp.float4_e2m1fn)
    scale = jnp.abs(x).max(-1, keepdims=True) / 6.0
    codes = np.unique(np.round(np.abs(np.asarray(q / scale)), 4))
    assert set(codes).issubset({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}), codes


def test_weight_quant_spec_rejects_unknown():
    with pytest.raises(ValueError):
        quant.weight_quant_spec("int17")
    with pytest.raises(ValueError):
        quant.weight_quant_spec("fp6")


def test_wider_dtypes_are_more_accurate():
    x = _rand((8, 128))
    errs = {
        s: float(jnp.abs(quant.fake_quantize(x, quant.WEIGHT_QUANT_DTYPES[s]) - x).mean())
        for s in SCHEMES
    }
    assert errs["int8"] < errs["int4"] < errs["fp4"] or errs["int8"] < errs["fp4"]
    assert errs["fp8"] < errs["fp4"]


def test_fake_quantize_tree_skip_rules_and_shapes():
    weights = {
        "layers.0.q_proj.weight": _rand((64, 32), 1),
        "layers.0.o_proj.weight": _rand((4, 16, 32), 2),  # contracts trailing 2 axes
        "layers.0.mlp.experts.weight": _rand((4, 32, 16), 3),  # contracts middle axis
        "layers.0.norm.weight": _rand((32,), 4),
        "layers.0.q_proj.bias": _rand((64,), 5),
        "layers.0.mlp.gate.weight": _rand((32, 4), 6),
    }
    out = quant.fake_quantize_tree(weights, jnp.int8, group_size=16)
    for k, w in weights.items():
        assert out[k].shape == w.shape and out[k].dtype == w.dtype
        untouched = w.ndim <= 1 or "norm" in k or k.endswith("mlp.gate.weight")
        if untouched:
            np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(w))
        else:
            assert float(jnp.abs(out[k] - w).max()) > 0
    # experts quantize over their middle (contraction) axis: an outlier in one
    # output column must not disturb other columns' scales.
    e = weights["layers.0.mlp.experts.weight"].at[0, :, 0].set(1e4)
    qe = quant.fake_quantize_tree({**weights, "layers.0.mlp.experts.weight": e},
                                  jnp.int8)["layers.0.mlp.experts.weight"]
    err_other_cols = float(jnp.abs(qe[0, :, 1:] - e[0, :, 1:]).mean())
    assert err_other_cols < 1e-2
