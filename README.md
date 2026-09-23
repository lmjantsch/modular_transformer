# modular-transformer

**Faithful, gradient-based attribution for HuggingFace LLMs, done by swapping the backward pass in place, without touching the forward pass.**

Say you want to know which input tokens, attention heads, or MLP neurons caused a language model's prediction. The obvious answer is `input × gradient`, but for transformers it is unreliable. Softmax, GELU/SiLU, LayerNorm/RMSNorm and multiplicative interactions (QKᵀ, attention·V, gated MLPs) all have local gradients that don't reflect how much each input actually contributed to the output. Relevance gets lost, created out of nothing, or counted twice.

This library fixes that at the autograd level. It walks a pretrained HuggingFace model, replaces each attention, MLP and norm block with a drop-in wrapper, and gives every nonlinear operation a custom backward rule. The rules are chosen so that relevance is **conserved** from layer to layer. The forward pass stays numerically identical, so the model still makes the same predictions. After patching, a normal `loss.backward()` returns attributions instead of raw gradients.

```python
from transformers import AutoModelForCausalLM
from modular_transformer import patch_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = patch_model(
    model,
    norm_approx="frozen",
    attn_act_fn="softmax",
    matmul_fn="bilinear_matmul",
    mul_fn="bilinear_mul",
    mlp_act_fn="secant_gelu_tanh",
)
# model is now an nnsight.NNsight wrapper: same logits, attribution-preserving gradients
```

## Why it matters

- **Interpretability at scale.** Attribution costs one forward and one backward pass, and it works on off-the-shelf checkpoints (GPT-2, Llama, Qwen2, Gemma-2) with no retraining and no model-specific rewrites.
- **Correctness you can check.** Conservation (`Σ x·∇x` is the same at every layer and equals the output score) is a property you can test. The test suite checks it per module and end to end.
- **Composable research tooling.** Each backward rule is a separate, named option. You can compare attribution methods (secant vs. integrated-gradients-style, uniform vs. dynamic bilinear splits) by changing a string argument instead of forking the model code.

## How it works

The method breaks a transformer down into four kinds of operations and gives each one a backward rule (derivations are in [`misc/math.md`](misc/math.md)):

| Rule | Applies to | Backward behaviour |
|---|---|---|
| 1. Linear | projections, RoPE, frozen norms | standard `Wᵀ t` |
| 2. Zero-preserving nonlinearity | SiLU, GELU | element-wise **secant** `σ(x)/x` in place of the derivative |
| 3. Non-zero-baseline nonlinearity | softmax | stabilised Jacobian `t − s·(1ᵀt)` (plus an integrated variant) |
| 4. Bilinear product | QKᵀ, A·V, gate ⊙ up | relevance **split** between operands (uniform 50/50 or a dynamic IG-midpoint split) |

Normalisation layers are handled by freezing their statistics (mean, variance or RMS) in the backward pass, which turns them into a linear per-element scaling.

## Engineering highlights

- **Non-invasive patching.** `patch_model` walks the `nn.Module` tree and swaps registered HF classes for `Modular*` wrappers. The wrappers **share the original parameters** rather than copying them, so there is no extra memory and weights can't drift. `dry_run=True` prints the patch plan, and `include`/`exclude` restrict what gets patched.
- **Extensible registry.** HF classes map to wrapper factories. HF imports are deferred, so a missing model family only fails when it's actually used. You can add new architectures with `register_module(hf_class, factory)`. Architectures with the same structure reuse existing wrappers (Qwen2 reuses the Llama wrappers, for example).
- **Custom `torch.autograd.Function`s.** Each rule is a small, self-contained op with a numerically careful implementation: fp32 upcasting inside the op, casting back to the model dtype, and epsilon/sign guards where a secant would divide by zero.
- **String-configurable backward.** Five kwargs (`norm_approx`, `attn_act_fn`, `matmul_fn`, `mul_fn`, `mlp_act_fn`) select the rule for each op type through lookup tables, so a new rule becomes available everywhere as soon as it's registered.
- **Activation access built in.** Patched models are wrapped in [nnsight](https://nnsight.net/), so you can read or intervene on the attributions of intermediate activations with the same API you'd use for forward activations.

## Supported architectures

| Family | Patched modules |
|---|---|
| GPT-2 | Attention, MLP, LayerNorm |
| Llama 2 / 3 | Attention (with RoPE), gated MLP, RMSNorm |
| Qwen2 / 2.5 | via the Llama wrappers |
| Gemma-2 | Attention (soft-capping, sliding window), gated MLP, RMSNorm |

## Verification

There are two layers of tests:

- **`tests/modules/`**: fast unit tests for every `autograd.Function`. They check that the forward pass matches the `torch.nn.functional` reference, that gradient shapes are correct, and that the op conserves relevance.
- **`tests/models/`**: integration tests that load real checkpoints, patch them, and run forward and backward passes over a set of prompts. They check:
  1. hidden-state identity with the unpatched model (relative L2 < 1e-2),
  2. next-token KL divergence < 5e-3,
  3. per-module conservation error < 0.1,
  4. total attribution at the embedding layer equals the output score.

```bash
pip install -e ".[dev]"
pytest tests/modules/                 # unit tests, no downloads
pytest tests/models/ --extended       # model tests with per-layer diagnostic tables
pytest tests/ --model gpt2 --mul_fn bilinear_mul   # override model / rule choices
```

## Project layout

```
src/modular_transformer/
  patching/   patch_model() entry point and HF → wrapper registry
  models/     Modular* wrappers per family (gpt2/, llama2/, gemma2/, generic/)
  modules/    autograd.Functions: secant activations, softmax rules, bilinear ops, norms
tests/        unit tests (modules/) and model-level identity/conservation tests (models/)
misc/math.md  derivation of the propagation rules
```

**Stack:** Python ≥ 3.10, PyTorch ≥ 2.0, HuggingFace Transformers, nnsight, pytest.
