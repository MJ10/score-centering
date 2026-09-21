import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P, reshard


RL_REWARD_MODES = (
    "plus_minus",
    "plus_zero",
    "zero_minus",
    "group_centered",
)


def get_batch_targets(tokens, mask):
    y = jnp.roll(tokens, -1, axis=1)
    mask = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    return tokens, y, mask


def cross_entropy(logits, labels):
    # Works with explicit axes, unlike the optax version.
    _, _, vocab_size = logits.shape
    log_softmax = jax.nn.log_softmax(logits.astype(jnp.float32))
    one_hot = jax.nn.one_hot(labels, vocab_size)
    return -jnp.sum(one_hot * log_softmax, axis=-1)


@partial(jax.jit, static_argnames="forward")
def loss_fn_ntp(forward, weights, batch):
    x, y, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits = forward(x, weights)
    loss = cross_entropy(logits, y)
    loss_sum = (loss * mask).sum()
    num_tokens = mask.sum()
    avg_loss = loss_sum / num_tokens
    return avg_loss, loss_sum, num_tokens


@partial(jax.jit, static_argnames="forward")
def loss_fn_kl(forward, weights_theta, weights_pi, batch):
    x, _, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits_theta = forward(x, weights_theta)
    logits_pi = forward(x, weights_pi)
    logprobs_pi = jax.nn.log_softmax(logits_pi.astype(jnp.float32))
    logprobs_theta = jax.nn.log_softmax(logits_theta.astype(jnp.float32))
    kl = optax.losses.kl_divergence_with_log_targets(logprobs_theta, logprobs_pi)
    loss_sum = (kl * mask).sum()
    num_tokens = mask.sum()
    avg_loss = loss_sum / num_tokens
    return avg_loss, loss_sum, num_tokens


def loss_fn_reg(forward, weights_theta, weights_pi, batch_train, batch_reg, reg, dtype=jnp.float32):
    if reg.method is None or reg.coeff == 0:
        return jnp.array(0.0, dtype=dtype)
    if reg.method == "ntp_pre":
        return loss_fn_ntp(forward, weights_theta, batch_reg)[0]
    if reg.method == "kl_dwn":
        return loss_fn_kl(forward, weights_theta, weights_pi, batch_train)[0]
    if reg.method == "kl_pre":
        return loss_fn_kl(forward, weights_theta, weights_pi, batch_reg)[0]
    raise ValueError(f"Unknown regularization method: {reg.method}")


def loss_fn_dpo(forward, weights_theta, weights_pi, batch, dpo):
    """Covers DPO, DPO-norm, SimPO."""
    xw, yw, mw = get_batch_targets(batch["chosen_tokens"], batch["chosen_mask"])
    xl, yl, ml = get_batch_targets(batch["rejected_tokens"], batch["rejected_mask"])

    lp_w_theta = -(cross_entropy(forward(xw, weights_theta), yw) * mw).sum(axis=1)
    lp_l_theta = -(cross_entropy(forward(xl, weights_theta), yl) * ml).sum(axis=1)

    if dpo.stop_grad_rejected:
        lp_l_theta = jax.lax.stop_gradient(lp_l_theta)

    if dpo.ref_weight > 0:
        lp_w_pi = -(cross_entropy(forward(xw, weights_pi), yw) * mw).sum(axis=1)
        lp_l_pi = -(cross_entropy(forward(xl, weights_pi), yl) * ml).sum(axis=1)
        reward_w = lp_w_theta - (dpo.ref_weight * lp_w_pi)
        reward_l = lp_l_theta - (dpo.ref_weight * lp_l_pi)
    else:
        reward_w = lp_w_theta
        reward_l = lp_l_theta

    if dpo.norm:
        reward_w = reward_w / mw.sum(axis=1)
        reward_l = reward_l / ml.sum(axis=1)

    logits = dpo.beta * (reward_w - reward_l) - dpo.margin
    preference_loss = -jax.nn.log_sigmoid(logits).mean()

    if dpo.sft_weight > 0:
        sft_loss = -(lp_w_theta / mw.sum(axis=1)).mean()
        preference_loss += dpo.sft_weight * sft_loss

    num_tokens = mw.sum() + ml.sum()
    loss_sum = preference_loss * num_tokens
    return preference_loss, loss_sum, num_tokens


def compute_advantages(rewards, group_size, reward_mode):
    solved = np.asarray(rewards) > 0
    match reward_mode:
        case "plus_minus" | "group_centered":
            advantages = np.where(solved, 1.0, -1.0)
        case "plus_zero":
            advantages = np.where(solved, 1.0, 0.0)
        case "zero_minus":
            advantages = np.where(solved, 0.0, -1.0)
        case _:
            raise ValueError(
                f"reward_mode must be one of {RL_REWARD_MODES}, "
                f"not {reward_mode!r}")
    advantages = advantages.astype(np.float32)
    if reward_mode == "group_centered":
        groups = advantages.reshape(-1, group_size)
        advantages = (groups - groups.mean(axis=1, keepdims=True)).reshape(-1)
    return advantages


def tb_advantages(rewards, seq_logp_theta, seq_logp_ref, group_size, beta):
    """Trajectory-balance (VarGrad) advantages for REINFORCE.

    Bartoldson et al. (2025, Appendix A, Eq. 12): the gradient of the VarGrad
    TB loss with reward R = pi_ref * exp(r / beta) equals, up to a positive
    constant, sum_j A_j grad log pi_theta(y_j) with

        A_j = (r_j - mean_j r) - beta * (l_j - mean_j l),
        l_j = log pi_theta(y_j | x) - log pi_ref(y_j | x),

    where the means run over the K responses to the same prompt (the batch
    estimate of log Z). Both log-probs are detached: the l term enters as a
    reward, not through the policy gradient. This is TBA without importance
    sampling. Rewards are mapped to +1/-1 like every other arm in this repo
    (compute_advantages), so beta = 0 reproduces reward_mode=group_centered
    exactly and the comparison isolates the KL term; beta therefore acts on
    a reward scale twice the TBA paper's 0/1 rewards.
    """
    r = np.where(np.asarray(rewards) > 0, 1.0, -1.0).astype(np.float32)
    r = r.reshape(-1, group_size)
    l = (np.asarray(seq_logp_theta, np.float32)
         - np.asarray(seq_logp_ref, np.float32)).reshape(-1, group_size)
    a = (r - r.mean(1, keepdims=True)) - beta * (l - l.mean(1, keepdims=True))
    return a.reshape(-1).astype(np.float32)


@partial(jax.jit, static_argnames=("forward", "head", "chunk", "pad_id"))
def sequence_logprobs(forward, head, weights, batch, pad_id, chunk=128):
    """Detached per-sequence sum of trainer-side log p(y_t | y_<t) over
    completion tokens, in rematerialized chunks like loss_fn_rl.
    `head(x, weights)` maps pre-head activations to logits."""
    _, target_tokens, target_mask = get_batch_targets(
        batch["tokens"], batch["mask"])
    hidden = forward(batch["tokens"], weights, logits=False,
                     live=batch["tokens"] != pad_id)
    seq_len = target_tokens.shape[1]

    # The carry takes the batch-axis sharding of the token logprobs so the
    # scan carry type is stable (see loss_fn_rl's per_seq accumulators).
    seq_spec = P(jax.typeof(hidden).sharding.spec[0])

    def chunk_sum(acc, start):
        slc = lambda x: jax.lax.dynamic_slice_in_dim(x, start, chunk, axis=1)
        vocab_logprobs = jax.nn.log_softmax(
            head(slc(hidden), weights).astype(jnp.float32))
        token_logprobs = take_vocab(
            vocab_logprobs, slc(target_tokens)[..., None])[..., 0]
        chunk_total = (token_logprobs * slc(target_mask)).sum(-1)
        return acc + reshard(chunk_total, seq_spec), None

    init = jnp.zeros(target_tokens.shape[0], jnp.float32, out_sharding=seq_spec)
    total, _ = jax.lax.scan(
        jax.checkpoint(chunk_sum), init, jnp.arange(0, seq_len, chunk))
    return jax.lax.stop_gradient(total)


IS_LEVELS = ("token", "seq_mean", "seq_sum")
IS_STATS = ("ratio", "tv")
IS_OUTSIDE = ("clamp", "drop")
IS_INSIDE = ("ratio", "one")
IS_SIGNS = ("both", "neg", "pos", "ppo")

DUAL_CLIP = 3.0  # PPO band top for A < 0 (arXiv:1912.09729); adopted, never tuned


class ISCorrection(NamedTuple):
    """One point on the IS-correction grid; the named methods (TIS/CISPO,
    MIS, IcePop, PPO/DAPO, GSPO, TOPR, DPPO) are spellings of it — table in
    README.md. Weights are always detached REINFORCE coefficients."""
    level: str = "token"   # token | seq_mean (mean log r) | seq_sum (sum log r)
    stat: str = "ratio"    # band statistic: the ratio r = p/q, or the binary
                           # TV divergence |q - p| (DPPO, arXiv:2602.04879)
    low: float = 0.0       # trust band; 0 / inf = open side (tv: high = delta)
    high: float = math.inf
    outside: str = "drop"  # clamp = pull weight to the nearest bound | drop = 0
    inside: str = "ratio"  # ratio = keep the IS weight | one = plain gradient
    sign: str = "both"     # both | neg | pos = advantages corrected (rest w=1)
                           # ppo = block only moves away from the sampler:
                           # ratio: pos (0, high], neg [low, DUAL_CLIP];
                           # tv: pos r>1, neg r<1, when tv > high (no dual clip)
    def in_band(self, r, positive, tv):
        if self.stat == "tv":
            if self.sign == "ppo":
                return jnp.where(positive, (r <= 1.0) | (tv <= self.high),
                                 (r >= 1.0) | (tv <= self.high))
            return tv <= self.high
        if self.sign == "ppo":
            return jnp.where(
                positive, r <= self.high, (r >= self.low) & (r <= DUAL_CLIP))
        return (r >= self.low) & (r <= self.high)


def is_weights(log_ratio, positive, c, tv=None):
    """Detached IS weight on r = exp(log_ratio), clamped to e**+-20. For
    level=seq_mean uses per-sequence means for log_ratio (and tv), while
    level=seq_sum uses the per-sequence sum (the full trajectory ratio).
    stat=tv gates the same weight on the tv statistic instead."""
    r = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    w = jnp.where(
        c.in_band(r, positive, tv),
        r if c.inside == "ratio" else 1.0,
        jnp.clip(r, c.low, c.high) if c.outside == "clamp" else 0.0)
    if c.sign == "neg":
        w = jnp.where(positive, 1.0, w)
    if c.sign == "pos":
        w = jnp.where(positive, w, 1.0)
    return jax.lax.stop_gradient(w)


def centering_weights(vocab_logprobs, sampling_logprobs, positive, c):
    """Effective centering mass q*w for a pointwise ratio correction.

    The calculation stays in log space, so the common in-band case
    q * (p/q) = p is exact and cannot overflow. The same helper operates on
    the full vocabulary, the exact top-k head, and one representative token
    for the modeled top-k tail.
    """
    lo = (sampling_logprobs + math.log(c.low)
          if c.low > 0 else -math.inf)
    hi = (sampling_logprobs + math.log(c.high)
          if not math.isinf(c.high) else math.inf)
    if c.sign == "ppo":
        in_band = jnp.where(
            positive[..., None], vocab_logprobs <= hi,
            (vocab_logprobs >= lo)
            & (vocab_logprobs <= sampling_logprobs + math.log(DUAL_CLIP)))
    else:
        in_band = (vocab_logprobs >= lo) & (vocab_logprobs <= hi)
    w = jnp.exp(jnp.where(
        in_band,
        vocab_logprobs if c.inside == "ratio" else sampling_logprobs,
        jnp.clip(vocab_logprobs, lo, hi) if c.outside == "clamp" else -jnp.inf))
    if c.sign == "neg":
        w = jnp.where(positive[..., None], jnp.exp(sampling_logprobs), w)
    if c.sign == "pos":
        w = jnp.where(positive[..., None], w, jnp.exp(sampling_logprobs))
    return w


def take_vocab(vocab_logprobs, ids):
    """take_along_axis on [B, C, V] whose vocab axis may be TP-sharded:
    the gather output drops that axis, so its sharding must be explicit."""
    spec = jax.typeof(vocab_logprobs).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(vocab_logprobs, ids, -1)
    b = jnp.arange(ids.shape[0])[:, None, None]
    c = jnp.arange(ids.shape[1])[None, :, None]
    return vocab_logprobs.at[b, c, ids].get(out_sharding=P(*spec[:-1], None))


def loss_fn_rl(hidden, head, batch, is_cfg=None, chunk=128,
               score_center=False):
    """REINFORCE and diagnostics using logprobs recorded during sampling.

    is_cfg (ISCorrection) reweights REINFORCE with detached IS weights on the
    p/q ratio: per token (level=token; the ppo shape is gradient-identical to
    the clipped surrogate), or one weight per sequence, applied after the
    scan. level=seq_mean uses the geometric-mean ratio; level=seq_sum uses the
    full trajectory ratio (the product of token ratios). Token-level ratio
    corrections compose with score centering by integrating under the
    effective mass q*w; TV corrections deliberately do not compose with score
    centering.

    head maps pre-head activations [B, C, D] to logits [B, C, V]. It runs
    together with the loss in rematerialized sequence chunks, so the
    full-vocabulary float32 intermediates (log-softmax, centering weights and
    their cotangents) only ever cover `chunk` positions per device at a time.

    When score_center is enabled, the batch determines the estimator. Full
    sampler logprobs give exact centering; otherwise the transported top-k
    head is exact and q's tail is modeled as proportional to the trainer's p.
    Every ratio correction is constant over that modeled tail, so its
    effective mass is alpha*p and the zero-expected-score identity leaves only
    top-k gathers in the centering backward pass.
    """
    token_c = is_cfg if is_cfg and is_cfg.level == "token" else None
    seq_c = (is_cfg if is_cfg and is_cfg.level in ("seq_mean", "seq_sum")
             else None)
    if is_cfg and is_cfg.level == "seq_sum" and is_cfg.stat != "ratio":
        raise ValueError("level=seq_sum requires stat=ratio")
    if score_center and is_cfg and is_cfg.stat != "ratio":
        raise ValueError("score centering composes only with stat=ratio")
    full_center = score_center and "sampling_vocab_logprobs" in batch
    topk_center = (
        score_center and not full_center and "center_ids" in batch)
    if score_center and not (full_center or topk_center):
        raise ValueError(
            "score centering requires full or top-k sampler vocab logprobs")
    _, target_tokens, target_mask = get_batch_targets(batch["tokens"], batch["mask"])
    seq_len = target_tokens.shape[1]
    chunk = next(c for c in range(min(chunk, seq_len), 0, -1) if seq_len % c == 0)

    def chunk_sums(sums, start):
        slc = lambda x: jax.lax.dynamic_slice_in_dim(x, start, chunk, axis=1)
        sg = jax.lax.stop_gradient
        logits = head(slc(hidden))
        vocab_logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
        token_logprobs = take_vocab(
            vocab_logprobs, slc(target_tokens)[..., None])[..., 0]
        log_ratio = token_logprobs - slc(batch["sampling_token_logprobs"])
        positive = batch["advantages"][:, None] >= 0
        tv = None
        if is_cfg and is_cfg.stat == "tv":
            tv = jnp.abs(jnp.exp(slc(batch["sampling_token_logprobs"]))
                         - jnp.exp(token_logprobs))
        importance_weights = 1.0
        if token_c:
            importance_weights = is_weights(log_ratio, positive, token_c, tv)
        score = importance_weights * token_logprobs
        # Negative entropy of the trainer distribution: a metric, and the
        # full-vocab factor of the pmodel tail. Detached — its logit gradient
        # is identically zero (E_p[grad log p] = 0), so backward never touches
        # the full vocabulary (see the paper's top-k derivation).
        plogp = sg((sg(jnp.exp(vocab_logprobs)) * vocab_logprobs).sum(-1))

        head_q = head_p = None
        if "center_ids" in batch:
            head_q = slc(batch["center_logprobs"])
            head_p = take_vocab(vocab_logprobs, slc(batch["center_ids"]))

        center = None
        if full_center:
            sampling_vocab_logprobs = slc(batch["sampling_vocab_logprobs"]).astype(jnp.float32)
            cw = (centering_weights(
                      vocab_logprobs, sampling_vocab_logprobs, positive, token_c)
                  if token_c else jnp.exp(sampling_vocab_logprobs))
            center = (sg(cw) * vocab_logprobs).sum(-1)
        if topk_center:
            # Apply the same correction to the exact transported head as the
            # full path applies to every vocabulary token.
            head_cw = (centering_weights(head_p, head_q, positive, token_c)
                       if token_c else jnp.exp(head_q))
            center = (sg(head_cw) * head_p).sum(-1)

            # Model q_tail = rho*p_tail, matching its transported mass. Since
            # p/q = 1/rho is constant there, every ratio correction produces
            # effective tail mass alpha*p. Evaluate alpha with one synthetic
            # p=1, q=rho token so band/sign semantics remain centralized in
            # centering_weights rather than being reimplemented here.
            tail_q_mass = jnp.maximum(1.0 - jnp.exp(head_q).sum(-1), 0.0)
            p_head = jnp.exp(head_p)
            rho = tail_q_mass / jnp.maximum(1.0 - p_head.sum(-1), 1e-6)
            alpha = rho
            if token_c:
                alpha = centering_weights(
                    jnp.zeros_like(rho)[..., None],
                    jnp.log(rho)[..., None], positive, token_c)[..., 0]
            # The head gather stays in the graph: only the full sum is
            # zero-gradient, a partial sum over the head is not.
            center += sg(alpha) * (
                plogp - (sg(p_head) * head_p).sum(-1))
        if center is not None:
            score = score - center

        mask = slc(target_mask)
        sums = dict(sums)
        if head_q is not None:
            # Head-based KL(q||p) with the tail modeled as q = r*p (exact when
            # q's tail is proportional to p's): tail contribution = tail * log r.
            q_head = jnp.exp(head_q)
            q_mass = q_head.sum(-1)
            r = ((1.0 - q_mass)
                 / jnp.maximum(1.0 - jnp.exp(head_p).sum(-1), 1e-6))
            kl_head = ((q_head * (head_q - head_p)).sum(-1)
                       + (1.0 - q_mass) * jnp.log(jnp.maximum(r, 1e-9)))
            sums["kl_head"] += (kl_head * mask).sum()
            sums["head_mass"] += (q_mass * mask).sum()
        sums["score"] += (score * mask).sum(-1)
        sums["entropy"] += (-plogp * mask).sum()
        sums["absdiff"] += (
            jnp.abs(slc(batch["sampling_token_logprobs"]) - token_logprobs)
            * mask
        ).sum()
        # Signed mean = E_q[log q - log p] on sampled tokens = KL(q||p) estimate;
        # matches molt's vllm_kl so mismatch is comparable across the two setups.
        sums["kl_samp_train"] += (
            (slc(batch["sampling_token_logprobs"]) - token_logprobs) * mask
        ).sum()
        if seq_c:
            sums["log_ratio"] += (sg(log_ratio) * mask).sum(-1)
            if tv is not None:
                sums["tv"] += (sg(tv) * mask).sum(-1)
        if token_c and token_c.outside == "drop":
            sums["masked"] += ((importance_weights == 0.0) * mask).sum()
        return sums, None

    scalar = jnp.zeros((), jnp.float32)
    # Per-sequence accumulators inherit the advantages' [B] sharding so the
    # scan carry type is stable.
    per_seq = jnp.zeros_like(batch["advantages"], dtype=jnp.float32)
    init = {"score": per_seq, "entropy": scalar, "absdiff": scalar,
            "kl_samp_train": scalar, "masked": scalar}
    if seq_c:
        init["log_ratio"] = per_seq
        if seq_c.stat == "tv":
            init["tv"] = per_seq
    if "center_ids" in batch:
        init.update(kl_head=scalar, head_mass=scalar)
    sums, _ = jax.lax.scan(
        jax.checkpoint(chunk_sums), init, jnp.arange(0, seq_len, chunk))
    count = jnp.maximum(target_mask.sum(), 1)
    weight = batch["advantages"]
    frac_masked = sums["masked"] / count
    if seq_c:
        seq_count = jnp.maximum(target_mask.sum(1), 1)
        seq_log_ratio = sums["log_ratio"]
        if seq_c.level == "seq_mean":
            seq_log_ratio = seq_log_ratio / seq_count
        w_seq = is_weights(
            seq_log_ratio, batch["advantages"] >= 0, seq_c,
            sums["tv"] / seq_count if seq_c.stat == "tv" else None)
        weight = weight * w_seq
        if seq_c.outside == "drop":
            frac_masked = (w_seq == 0.0).mean()  # fraction of sequences
    loss = -(weight * sums["score"]).sum() / count
    metrics = {
        "pol_entropy": sums["entropy"] / count,
        "logp_absdiff_samp_train": sums["absdiff"] / count,
        "kl_samp_train": sums["kl_samp_train"] / count,
        "frac_masked": frac_masked,
    }
    if "center_ids" in batch:
        metrics["kl_head"] = sums["kl_head"] / count
        metrics["head_mass"] = sums["head_mass"] / count
    return loss, metrics


# --- validation-loss evals: loops over the training batch pipelines, so they
# --- live with the losses they aggregate, not with the benchmarks (evals/)

@partial(jax.jit, static_argnames="forward")
def eval_preference_step(forward, weights, batch):
    xw, yw, mw = get_batch_targets(batch["chosen_tokens"], batch["chosen_mask"])
    xl, yl, ml = get_batch_targets(batch["rejected_tokens"], batch["rejected_mask"])
    lp_chosen = -(cross_entropy(forward(xw, weights), yw) * mw).sum(axis=1)
    lp_rejected = -(cross_entropy(forward(xl, weights), yl) * ml).sum(axis=1)
    lp_chosen_norm = lp_chosen / mw.sum(axis=1)
    lp_rejected_norm = lp_rejected / ml.sum(axis=1)
    return lp_chosen, lp_rejected, lp_chosen_norm, lp_rejected_norm


def eval_preference(model, make_valid, epoch):
    chosen_lps, rejected_lps = [], []
    chosen_lps_norm, rejected_lps_norm = [], []
    for batch in make_valid(epoch):
        lp_c, lp_r, lp_c_norm, lp_r_norm = eval_preference_step(model.forward, model.weights, batch)
        chosen_lps.append(lp_c)
        rejected_lps.append(lp_r)
        chosen_lps_norm.append(lp_c_norm)
        rejected_lps_norm.append(lp_r_norm)
    chosen_lps = jnp.concatenate(chosen_lps)
    rejected_lps = jnp.concatenate(rejected_lps)
    chosen_lps_norm = jnp.concatenate(chosen_lps_norm)
    rejected_lps_norm = jnp.concatenate(rejected_lps_norm)
    return {
        "preference/valid/chosen_logprob": chosen_lps.mean(),
        "preference/valid/rejected_logprob": rejected_lps.mean(),
        "preference/valid/accuracy": (chosen_lps > rejected_lps).mean(),
        "preference/valid/margin": (chosen_lps - rejected_lps).mean(),
        "preference/valid/chosen_logprob_norm": chosen_lps_norm.mean(),
        "preference/valid/rejected_logprob_norm": rejected_lps_norm.mean(),
        "preference/valid/accuracy_norm": (chosen_lps_norm > rejected_lps_norm).mean(),
        "preference/valid/margin_norm": (chosen_lps_norm - rejected_lps_norm).mean(),
    }


def eval_lm_losses(model, weights_pi, ds_eval, target_tokens=1_000_000):
    total_ntp_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_tokens = 0
    while total_tokens < target_tokens:
        batch = next(ds_eval)
        if "tokens" in batch and "mask" in batch:
            lm_batch = batch
        elif "chosen_tokens" in batch and "chosen_mask" in batch:
            lm_batch = {"tokens": batch["chosen_tokens"], "mask": batch["chosen_mask"]}
        else:
            raise ValueError(f"Batch is missing LM tokens/mask fields: {batch.keys()}")
        _, batch_ntp_loss_sum, batch_tokens = loss_fn_ntp(model.forward, model.weights, lm_batch)
        _, batch_kl_loss_sum, _ = loss_fn_kl(model.forward, model.weights, weights_pi, lm_batch)
        total_ntp_loss_sum += batch_ntp_loss_sum
        total_kl_loss_sum += batch_kl_loss_sum
        total_tokens += batch_tokens
    return total_ntp_loss_sum / total_tokens, total_kl_loss_sum / total_tokens


@partial(jax.jit, static_argnames="forward")
def eval_lm_valid_step(forward, weights, batch):
    x, y, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits = forward(x, weights)
    loss_sum = (cross_entropy(logits, y) * mask).sum()
    correct = jnp.logical_and(jnp.argmax(logits, -1) == y, mask).sum()
    return loss_sum, correct, mask.sum()


def eval_lm_valid(model, make_valid):
    total_loss = total_correct = total_tokens = 0
    for batch in make_valid(0):
        loss_sum, correct, n_tokens = eval_lm_valid_step(model.forward, model.weights, batch)
        total_loss += loss_sum
        total_correct += correct
        total_tokens += n_tokens
    return {"eval/valid/ntp": total_loss / total_tokens, "eval/valid/accuracy": total_correct / total_tokens}
