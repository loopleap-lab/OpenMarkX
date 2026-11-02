# OpenMarkX FAQ

Short answers to the questions people ask when they start using OpenMarkX. Each answer links to the full
documentation.

- [The basics](#the-basics)
- [Installing and running](#installing-and-running)
- [Keys](#keys)
- [Detecting](#detecting)
- [Old names](#old-names)
- [Extending OpenMarkX](#extending-openmarkx)
- [Reproducing the paper](#reproducing-the-paper)
- [Troubleshooting](#troubleshooting)

## The basics

### What does OpenMarkX do?

It learns a watermark into the weights of an open-source model, so that what the model generates carries the mark
without any change to decoding. It then tests the mark under edits a downstream user may make (fine-tuning,
merging, pruning, quantization, rewording) and detects it with the owner's secret key. `openmarkx bench` compares
methods on a leaderboard of your own runs; [LEADERBOARD.md](../LEADERBOARD.md) holds the paper's results.

### Which models does it support?

| Model | Adapter |
|---|---|
| Llama-style causal language models | `text` |
| a ReAct agent around a text model | `agent` |
| image and text in, text out (MedGemma, Gemma 3 style) | `vlm` |
| Taming Transformers' image tokens (through WMAR) | `taming` |
| GeoPT physics fields on 3D points | `geopt` |

[Models](models.md) explains the adapters and how to bring your own.

### Which watermark methods are included?

OpenMark (the paper's method, in three settings), 12 MarkLLM methods distilled into the weights, GaussMark, `wmar`
for image tokens and `ripple` for fields. `openmarkx describe` prints the table; [Methods](methods.md) has the
details.

### What is the difference between a release and a key?

The release is the watermarked model you publish: weights and `release.json`. The secret key stays with you, in the
key store. The release holds only a commitment to the key. See [Keys](user-guide.md#keys).

## Installing and running

### How do I install it?

```bash
git clone https://github.com/UCF-ML-Research/OpenMarkX.git && cd OpenMarkX
pip install -e ".[test]"
bash scripts/install_deps.sh        # MarkLLM at a pinned commit, for the markllm-* methods
```

The [user guide](user-guide.md#installation) lists the other options.

### Do I need a GPU?

Training and sampling need a CUDA GPU. `openmarkx check` and the tests run on CPU. Without a GPU, the
[Colab notebook](https://colab.research.google.com/github/UCF-ML-Research/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb)
installs OpenMarkX on a free T4 and starts the web page.

### Do I need MarkLLM?

Only for the `markllm-*` methods. `scripts/install_deps.sh` clones it at the pinned commit. OpenMark ships with the
package.

### My run stopped. Do I start over?

No. Run the same command again. Each step of `openmarkx run` writes `<out>/<step>.json` and is skipped when that
file exists, and an interrupted imprint resumes toward the same key. See
[Running an experiment from a config](user-guide.md#running-an-experiment-from-a-config).

### How do I check a setup before a long run?

`openmarkx run CONFIG --smoke` trains at most 5 steps per phase on 4 samples and writes to `<out>_smoke`, so the
real run's directory stays empty.

### Can I run it on a remote GPU from my browser?

Yes. Start `openmarkx serve` on the GPU machine and forward the port with `ssh -L 7860:127.0.0.1:7860 gpu-server`.
See [Running from a web page](user-guide.md#running-from-a-web-page).

## Keys

### Where are my keys?

In the key store, `$OMX_KEYS`, by default `~/.openmarkx/runs/keys`. `openmarkx keys list` lists them, and
`openmarkx keys show KEY` prints a key's metadata without its key material.

### I lost my secret key. Can I recover it?

No. A release holds only a commitment to its key (a sha256 over a nonce, the scheme, its parameters and the plugin
source), and the key cannot be rebuilt from it. Without the key, the mark in that release cannot be detected.
Back up `$OMX_KEYS`, including its `blobs/` directory.

### Is the secret key inside the release I publish?

No. Key material goes to the key store, never to the release directory. For methods that write the weights
directly, `openmarkx check` tests that the release holds only model files (C7).

### `detect` warns that the key is public. What does that mean?

The key was imported with `port --public`: it was shipped with the checkpoint, such as a paper's released mapping.
Anyone can detect with it, and anyone can also remove or forge the mark.

## Detecting

### Should I use `detect` or `verify`?

`detect` gives one p-value per text you already have. `verify` samples a suspect model on prompts and gives one
verdict for the model. With `--sequential` it stops as soon as the evidence is decisive. See
[Detecting and verifying](user-guide.md#detecting-and-verifying).

### Do I have to calibrate a key before detecting?

No, but calibration gives p-values from a frozen null of the base model's own texts. Without it, `detect` uses the
scheme's closed-form null when there is one, or you can pass `--null wrong_key`, which scores each text under 200
wrong keys. `verify` needs a calibrated key unless the scheme has a closed-form null.

### Why does a text get `p_value: null`?

It is too short to be judged against any calibrated null. Such a text counts as no evidence either way, and
`verify` leaves it out.

### Why does a MarkLLM key refuse to detect on CPU?

The `markllm-*` keys drew their randomness on CUDA, and CPU and CUDA generators draw different numbers, so scoring
on CPU would use a different key. `detect` runs on the device type the key was made on and needs a GPU for these
keys.

### Can I detect images or physics fields?

Yes: `openmarkx detect images/ --model taming --key RELEASE` for a directory of PNG images, and
`openmarkx detect case.npz --model geopt --key RELEASE` for a field. See
[Images and fields](user-guide.md#images-and-fields).

## Old names

### I have code, keys or runs from MarkWeightX. Do they still work?

Yes. `import markweightx`, the `markweightx` command, `MWX_*` environment variables and `~/.markweightx` all keep
working. An unset `OMX_*` variable falls back to its `MWX_*` namesake, and `~/.markweightx` is used when
`~/.openmarkx` does not exist.

### My keys and configs say `pro`. Do I have to change them?

No. `pro`, `pro-coopt-c100`, `pro-full-b01` and the other PRO names are aliases of `openmark`,
`openmark-coopt-c100`, `openmark-full-b01`, ... A run directory made under the old name is resumed, not refused.

## Extending OpenMarkX

### How do I add a watermark method?

`openmarkx new-scheme my-mark` writes a one-file template; `openmarkx check --adapter my_mark.py` tests it on CPU.
See [Adding a method](methods.md#adding-a-method).

### How do I add an attack?

A weight edit or a text edit is one function with `@omx.register_edit` or `@omx.register_text_edit`. See
[Adding an edit](benchmark.md#adding-an-edit).

### How do I watermark my own model?

See [Models](models.md).

### Can I use my own dataset or another base model?

Yes. `exp/prepare_data.py --dataset ...` builds a corpus from a Hugging Face dataset or a `.jsonl` file, and
`--base` takes a Hugging Face repo id or a local directory. See [Your own dataset](user-guide.md#your-own-dataset)
and [Another base model](user-guide.md#another-base-model).

## Reproducing the paper

### Can I reproduce the paper's numbers?

Not with one command. The paper's results on LLaMA-3-8B are in [LEADERBOARD.md](../LEADERBOARD.md). The repository
does not ship the paper's trained weights, and its edit grid is not the paper's exact set of modifications.
`openmarkx bench configs/arena/1b.yaml` runs the same kind of benchmark on Llama-3.2-1B. See
[Reproducing the paper](user-guide.md#reproducing-the-paper).

### Does the website show calls that do not run yet?

Yes. Some Python calls on the website, such as `omx.embed` for the image, vision-language, physics and agent demos
and `release.push`, are planned. [ROADMAP.md](../ROADMAP.md) lists them. The demos ran with
`openmarkx run configs/demos/<name>.yaml`.

## Troubleshooting

### The run stops with "... is not installed: clone it and set OMX_GEOPT_DIR".

The `taming` and `geopt` adapters use the authors' code from your own checkouts. Set `OMX_WMAR_DIR` to a WMAR
checkout or `OMX_GEOPT_DIR` to a GeoPT checkout.

### A command fails and I want the traceback.

Set `OMX_DEBUG=1`.

### Detection fails with "this key runs plugin code".

The key was made with a plugin file, and OpenMarkX runs plugin code only when your process imported that exact
file, or when you trust it. Pass `--trust-plugin` (`trust_plugin=True`) for keys you or someone you trust made.

### Downloading `meta-llama/*` or MedGemma fails.

These models are gated on Hugging Face. Accept their terms on the model page and run `huggingface-cli login`.
