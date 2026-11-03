#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Chenchen Gu
# Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
#   - Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from transformers.trainer import unwrap_model
from torch.distributed import is_initialized, barrier
import functools
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union
import torch.nn as nn
import datasets
import evaluate
import torch
from datasets import load_dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision, ShardingStrategy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_utils import FSDPOption, get_last_checkpoint
from transformers.utils.versions import require_version
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention

from watermarks.aar.aar_watermark import AarWatermark
from watermarks.kgw.kgw_watermark import KGWWatermark
from watermarks.kth.kth_watermark import KTHWatermark
from watermarks.pro.pro_watermark import PROWatermark
from watermarks.watermark_types import WatermarkType


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np
import torch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    watermark_type: WatermarkType = field(
        default=None,
        metadata={
            "help": (
                "Type of watermark to use."
            ),
            "choices": [WatermarkType.AAR, WatermarkType.KGW, WatermarkType.KTH, WatermarkType.PRO],
        },
    )
    watermark_seed: int = field(
        default=42,
        metadata={
            "help": (
                "Seed for watermarking."
            )
        },
    )
    aar_watermark_k: int = field(
        default=1,
        metadata={
            "help": (
                "Number of previous tokens to hash for Aar watermark."
            )
        }
    )
    kth_watermark_key_len: int = field(
        default=256,
        metadata={
            "help": (
                "Key length for KTH watermark."
            )
        }
    )
    kth_watermark_num_shifts: int = field(
        default=1,
        metadata={
            "help": (
                "Number of possible shifts for KTH watermark."
            )
        }
    )
    kgw_watermark_gamma: float = field(
        default=0.5,
        metadata={
            "help": "Gamma for watermark, i.e. proportion of vocab that is greenlist."
        },
    )
    kgw_watermark_delta: float = field(
        default=2.0,
        metadata={
            "help": "Delta for watermark, i.e. value to add to logits in greenlist."
        },
    )
    kgw_watermark_seeding_scheme: str = field(
        default="simple_1",
        metadata={
            "help": "Seeding scheme to use for watermark. See kgw_watermarking for more details."
        },
    )
    pro_watermark_gamma: float = field(
        default=0.5,
        metadata={
            "help": "Gamma for PRO watermark, i.e. proportion of vocab that is greenlist."
        },
    )
    pro_watermark_delta: float = field(
        default=2.0,
        metadata={
            "help": "Delta for PRO watermark, i.e. value to add to logits in greenlist."
        },
    )
    pro_watermark_chunk_size: Optional[int] = field(
        default=10,
        metadata={
            "help": "Chunk size for PRO watermark processing."
        },
    )
    pro_embedding_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Embedding model to use for PRO watermark."
        },
    )
    pro_transform_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to transform model for PRO watermark."
        },
    )
    pro_mapping_file_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to mapping file for PRO watermark."
        },
    )
    transform_lr: float = field(default=1e-4)
    transform_weight_decay: float = field(default=0.01)
    transform_warmup_steps: int = field(default=100)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set. (Only if streaming is False.)"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def is_sagemaker_mp_enabled():
    try:
        import smdistributed.modelparallel.torch as smp
        return smp.state.model_parallel_enabled()
    except (ImportError, AttributeError):
        return False

@dataclass
class LogitsDistillTrainingArguments(TrainingArguments):
    """Training arguments for robust watermark distillation."""
    save_checkpoint_models: bool = field(
        default=True,
        metadata={"help": "Save model at every checkpoint, no deletion, no optimizer states."},
    )
    robustness_mode: str = field(
        default="vaccine",
        metadata={
            "help": "Robustness training mode.",
            "choices": ["vaccine", "vaccine_frozen", "random_frozen", "booster", "booster_anti"],
        },
    )
    train_transform: bool = field(
        default=True,
        metadata={"help": "Whether to train the transform model. Set False for frozen variants."},
    )
    rho: float = field(
        default=0.5,
        metadata={"help": "Perturbation scale for vaccine (gradient-based) or variance for random mode."},
    )
    alpha: float = field(default=0.1, metadata={"help": "Attack step size for booster mode."})
    lamb: float = field(default=1.0, metadata={"help": "Anti-unlearning regularizer weight for booster mode."})

    
def get_leaf_modules_with_grad(module):
    module_list = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, (LlamaAttention, LlamaMLP, OPTAttention)):
            module_list.append(sub_module)
    return module_list


class WatermarkLogitsDistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        watermarker: Union[AarWatermark, KTHWatermark, KGWWatermark, PROWatermark],
        argmax_watermark: bool = True,
        model_args=None,
        harmful_dataset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model = self._fsdp_teacher_model(self.teacher_model)
        self.teacher_model.eval()
        self.watermarker = watermarker
        self.argmax_watermark = argmax_watermark
        self.model_args = model_args
        self.mode = self.args.robustness_mode
        if self.argmax_watermark:
            self.loss_fct = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.vaccine_state = {"hooks": [], "gradient": {}}

        # Booster mode: harmful dataset for simulating attacks
        if harmful_dataset is not None:
            from torch.utils.data import DataLoader, RandomSampler
            sampler = RandomSampler(harmful_dataset)
            self.harmful_dataloader = DataLoader(
                harmful_dataset, sampler=sampler,
                batch_size=self.args.per_device_train_batch_size,
                num_workers=self.args.dataloader_num_workers, pin_memory=True,
            )
            self.harmful_data_iter = iter(self.harmful_dataloader)

        # Buffer for collecting transformed embeddings across steps
        self.transformed_buffer = []
        self.current_accumulation_step = 0


    ############################################################################
    # Phase 1: Capture gradients (pre_first_step & after_first_step)
    ############################################################################
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_full_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model: nn.Module):
        """
        Remove backward hooks and process the captured gradients:
        compute L2 norm, apply a safe clamp or early skip, then scale once by effective rho.
        """
        # Remove all hooks registered in pre_first_step
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"].clear()

        # Compute overall gradient L2 norm
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])
        hard_skip = 1e-6
        if grad_norm.item() < hard_skip:
            # Too small to perturb: zero out all
            for m in self.vaccine_state["gradient"]:
                self.vaccine_state["gradient"][m] = torch.zeros_like(self.vaccine_state["gradient"][m])
            return

        # Clamp to avoid extreme scaling
        soft_min = 1e-3
        safe_norm = max(grad_norm.item(), soft_min)

        # Compute effective rho once, accounting for accumulation steps
        effective_rho = self.args.rho / self.args.gradient_accumulation_steps
        scale = effective_rho / safe_norm

        # Apply perturbation and store it
        for module, grad in self.vaccine_state["gradient"].items():
            e_r = grad * scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
        # total_perturbation_norm = torch.norm(torch.stack([
        #     p.norm(p=2) for p in self.vaccine_state["gradient"].values()
        # ]), p=2).item()

        # print(f"[Total Perturbation Norm] {total_perturbation_norm:.6f}")

    ############################################################################
    # Phase 2: Apply perturbation using forward hooks
    ############################################################################
    @torch.no_grad()
    def pre_second_step(self, model: nn.Module):
        """
        Register forward hook on Attention/MLP layers to add the pre-computed perturbation.
        """
        leaf_modules = get_leaf_modules_with_grad(model)

        def forward_hook(module, input, output):
            pert = self.vaccine_state["gradient"].get(module, None)
            if pert is None or torch.all(pert == 0):
                return output
            if isinstance(output, tuple):
                # output is (hidden_states, attention_weights) or similar
                return (output[0] + pert, *output[1:])
            else:
                return output + pert


        for layer in leaf_modules:
            hook = layer.register_forward_hook(forward_hook)
            self.vaccine_state["hooks"].append(hook)


    @torch.no_grad()
    def after_second_step(self, model: nn.Module):
        """
        Remove forward hooks registered in pre_second_step.
        """
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"].clear()

    @torch.no_grad()
    def _grad_norm(self, gradients: Dict[nn.Module, torch.Tensor]) -> torch.Tensor:
        norms = [g.norm(p=2) for g in gradients.values()]
        return torch.norm(torch.stack(norms), p=2)

    def _pre_random_step(self, model: nn.Module):
        """Register forward hooks that add random Gaussian noise (for random_frozen mode)."""
        leaf_modules = get_leaf_modules_with_grad(model)
        variance = self.args.rho

        def forward_hook(module, input, output):
            perturb = (variance ** 0.5) * torch.randn_like(
                output[0] if isinstance(output, tuple) else output
            )
            if isinstance(output, tuple):
                return (output[0] + perturb, *output[1:])
            return output + perturb

        for layer in leaf_modules:
            hook = layer.register_forward_hook(forward_hook)
            self.vaccine_state["hooks"].append(hook)

    def _sample_from_harmful(self):
        try:
            return next(self.harmful_data_iter)
        except StopIteration:
            self.harmful_data_iter = iter(self.harmful_dataloader)
            return next(self.harmful_data_iter)

    def _compute_anti_watermark_loss(self, model, inputs):
        """Compute loss against anti-watermarked teacher (for booster_anti mode)."""
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        anti_logits = self.watermarker.watermark_logits(
            inputs["input_ids"], teacher_outputs.logits, anti=True
        )
        return torch.nn.KLDivLoss(reduction="batchmean", log_target=False)(
            torch.nn.functional.log_softmax(outputs.logits, dim=-1),
            torch.nn.functional.softmax(anti_logits, dim=-1),
        ) / outputs.logits.shape[1]

    def _save_optimizer_and_scheduler(self, output_dir):
        return

    def create_optimizer(self):
        """
        Override the default create_optimizer() to also include self.watermarker.transform_model parameters.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if hasattr(self.watermarker, "transform_model") and self.watermarker.transform_model is not None:
            class CombinedModel(torch.nn.Module):
                def __init__(self, model1, model2):
                    super().__init__()
                    self.model1 = model1
                    self.model2 = model2

                def named_parameters(self, prefix='', recurse=True):
                    for name, param in self.model1.named_parameters(prefix=prefix + "model.", recurse=recurse):
                        yield name, param
                    for name, param in self.model2.named_parameters(prefix=prefix + "transform_model.", recurse=recurse):
                        yield name, param

            opt_model = CombinedModel(opt_model, self.watermarker.transform_model)

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer



    def asymmetric_scale(self, v, gamma):
        scale = (1 - gamma) / gamma
        return torch.where(v > 0, v * scale, v)
    
    def row_col_mean_penalty(self, output, gamma=0.25):
        """Modified normalization loss using asymmetric scaling."""
        scaled_output = self.asymmetric_scale(output, gamma)
        row_penalty = torch.mean(scaled_output, dim=1).pow(2).sum()
        col_penalty = torch.mean(scaled_output, dim=0).pow(2).sum()
        return row_penalty + col_penalty


    def abs_value_penalty(self, output):
        """Calculate absolute value range penalties on the entire batch"""
        penalties = torch.relu(0.05 - torch.abs(output))
        mask = (penalties > 0).float()
        non_zero_count = torch.max(mask.sum(), torch.tensor(1.0, device=output.device))
        return (penalties * mask).sum() / non_zero_count


    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        train_transform = self.args.train_transform and hasattr(self.watermarker, 'transform_model')
        unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, 'accelerator') else model

        if self.mode in ("vaccine", "vaccine_frozen"):
            # Phase 1: Freeze transform, capture gradients
            if train_transform:
                for p in self.watermarker.transform_model.parameters():
                    p.requires_grad = False
            self.pre_first_step(model)
            student_loss = self.compute_loss(model, inputs, with_transform=False)
            self.accelerator.backward(student_loss)
            self.after_first_step(model)
            model.zero_grad()

            # Phase 2: Apply perturbation, optionally train transform
            if train_transform and self.mode == "vaccine":
                for p in self.watermarker.transform_model.parameters():
                    p.requires_grad = True
            self.pre_second_step(model)
            full_loss = self.compute_loss(model, inputs, with_transform=(self.mode == "vaccine" and train_transform))
            self.accelerator.backward(full_loss)
            self.after_second_step(model)
            return full_loss.detach()

        elif self.mode == "random_frozen":
            # Single phase: random perturbation, no transform training
            if train_transform:
                for p in self.watermarker.transform_model.parameters():
                    p.requires_grad = False
            self._pre_random_step(model)
            loss = self.compute_loss(model, inputs, with_transform=False)
            self.accelerator.backward(loss)
            self.after_second_step(model)
            return loss.detach()

        elif self.mode in ("booster", "booster_anti"):
            # Step 1: Compute harmful loss
            harmful_inputs = self._prepare_inputs(self._sample_from_harmful())
            if self.mode == "booster_anti":
                if train_transform:
                    for p in self.watermarker.transform_model.parameters():
                        p.requires_grad = False
                harmful_loss = self._compute_anti_watermark_loss(model, harmful_inputs)
            else:
                harmful_outputs = model(**harmful_inputs)
                harmful_loss = torch.nn.functional.cross_entropy(
                    harmful_outputs.logits.view(-1, harmful_outputs.logits.size(-1)),
                    harmful_inputs["labels"].view(-1),
                )
            self.accelerator.backward(harmful_loss)

            # Step 2: Store harmful gradients
            stored_grads = {
                name: param.grad.detach().clone()
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and param.grad is not None and param.data.shape == param.grad.shape
            }
            model.zero_grad()
            if not stored_grads:
                loss = self.compute_loss(model, inputs, with_transform=(self.mode == "booster" and train_transform))
                self.accelerator.backward(loss)
                return loss.detach() / self.args.gradient_accumulation_steps

            # Step 3: Perturb params in harmful direction
            grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in stored_grads.values()])) + 1e-7
            with torch.no_grad():
                for name, param in unwrapped.named_parameters():
                    if name in stored_grads:
                        param.add_(-self.args.alpha * stored_grads[name] / grad_norm)

            # Step 4: Compute perturbed harmful loss
            if self.mode == "booster_anti":
                perturbed_loss = self._compute_anti_watermark_loss(model, harmful_inputs)
            else:
                perturbed_outputs = model(**harmful_inputs)
                perturbed_loss = torch.nn.functional.cross_entropy(
                    perturbed_outputs.logits.view(-1, perturbed_outputs.logits.size(-1)),
                    harmful_inputs["labels"].view(-1),
                )
            self.accelerator.backward(perturbed_loss)

            perturb_grads = {
                name: param.grad.detach().clone()
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and param.grad is not None and param.data.shape == param.grad.shape
            }
            model.zero_grad()

            # Step 5: Restore params
            with torch.no_grad():
                for name, param in unwrapped.named_parameters():
                    if name in stored_grads:
                        param.add_(self.args.alpha * stored_grads[name] / grad_norm)

            # Step 6: Compute watermark loss + anti-unlearning regularizer
            with self.compute_loss_context_manager():
                wm_loss = self.compute_loss(model, inputs, with_transform=(self.mode == "booster" and train_transform))
            self.accelerator.backward(wm_loss)

            with torch.no_grad():
                for name, param in unwrapped.named_parameters():
                    if param.grad is not None and name in stored_grads and name in perturb_grads:
                        param.grad.add_(self.args.lamb * (stored_grads[name] - perturb_grads[name]))

            return wm_loss.detach() / self.args.gradient_accumulation_steps

        else:
            raise ValueError(f"Unknown robustness_mode: {self.mode}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        with_transform: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, any]]:
        """
        Compute student distillation loss, optionally add transform loss when with_transform=True.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # student loss
        if self.argmax_watermark:
            wm_tokens = self.watermarker.watermark_logits_argmax(inputs["input_ids"], teacher_outputs.logits)
            student_loss = self.loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                wm_tokens.view(-1)
            )
        else:
            watermarked = self.watermarker.watermark_logits(inputs["input_ids"], teacher_outputs.logits)
            student_loss = self.loss_fct(
                torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                torch.nn.functional.log_softmax(watermarked, dim=-1)
            ) / outputs.logits.size(1)

        # only compute transform penalty in Phase 2
        transform_loss = torch.tensor(0.0, device=outputs.logits.device, requires_grad=True)
        if with_transform and hasattr(self.watermarker, "transform_model") and self.watermarker.transform_model is not None:
            embeddings = self.watermarker.get_embeddings(inputs["input_ids"])
            transformed = self.watermarker.transform_model(embeddings)
            transform_loss = self.compute_transform_loss(transformed)

        total_loss = student_loss + transform_loss
        # log both losses
        self.log({
            "loss/student": student_loss.item(),
            "loss/transform": transform_loss.item(),
            "loss/total": total_loss.item(),
        })

        return (total_loss, outputs) if return_outputs else total_loss

    def compute_transform_loss(self, transformed_embeddings: torch.FloatTensor) -> torch.Tensor:
        self.transformed_buffer.append(transformed_embeddings)
        self.current_accumulation_step += 1

        if self.current_accumulation_step < self.args.gradient_accumulation_steps:
            return torch.tensor(0.0, device=transformed_embeddings.device, requires_grad=True)

        all_transformed = torch.cat(self.transformed_buffer, dim=0)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered = [torch.zeros_like(all_transformed) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, all_transformed)
            all_transformed = torch.cat(gathered, dim=0)

        scaled_output = self.asymmetric_scale(all_transformed, gamma=self.model_args.pro_watermark_gamma)
        row_penalty = torch.mean(scaled_output, dim=1).pow(2).sum()
        col_penalty = torch.mean(scaled_output, dim=0).pow(2).sum()
        mean_penalty = row_penalty + col_penalty

        range_penalty = self.abs_value_penalty(all_transformed)

        transform_loss = mean_penalty + range_penalty

        self.log({
            "loss/transform/row_penalty": row_penalty.item(),
            "loss/transform/col_penalty": col_penalty.item(),
            "loss/transform/mean_penalty": mean_penalty.item(),
            "loss/transform/range_penalty": range_penalty.item(),
            "loss/transform/total": transform_loss.item(),
        })

        self.transformed_buffer = []
        self.current_accumulation_step = 0

        return transform_loss

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial)
        if hasattr(self.watermarker, 'transform_model') and self.watermarker.transform_model is not None:
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            transform_path = os.path.join(output_dir, "transform_model.pt")
            torch.save(self.watermarker.transform_model.state_dict(), transform_path)
            print(f"[Info] transform_model saved to: {transform_path}")

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        super()._save(output_dir=output_dir, **kwargs)
        try:
            out_dir = output_dir if output_dir is not None else self.args.output_dir
            watermark_config = {}
            for k, v in vars(self.watermarker).items():
                if isinstance(v, (str, int, float, bool, list)):
                    watermark_config[k] = v
            config_path = os.path.join(out_dir, "watermark_config.json")
            with open(config_path, "w") as f:
                json.dump(watermark_config, f)
            print(f"[Info] watermark_config saved to: {config_path}")
        except Exception as e:
            print(f"Failed to save watermark_config: {e}")

  

    def _fsdp_teacher_model(self, model):
        if self.args.fsdp is not None and len(self.args.fsdp) > 0:
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullyShardedDataParallel as FSDP,
                CPUOffload,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            # Parse FSDP config
            cpu_offload = CPUOffload(offload_params=FSDPOption.OFFLOAD in self.args.fsdp)

            auto_wrap_policy = None
            if FSDPOption.AUTO_WRAP in self.args.fsdp:
                fsdp_config = self.args.fsdp_config or {}
                fsdp_min_num_params = fsdp_config.get("min_num_params", 0)
                fsdp_cls_to_wrap = fsdp_config.get("transformer_layer_cls_to_wrap", None)
                if fsdp_min_num_params > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy,
                        min_num_params=fsdp_min_num_params,
                    )
                elif fsdp_cls_to_wrap is not None:
                    transformer_cls_to_wrap = get_module_class_from_name(
                        model, fsdp_cls_to_wrap
                    )
                    if transformer_cls_to_wrap is None:
                        raise Exception("Could not find transformer layer class to wrap.")
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={transformer_cls_to_wrap},
                    )

            dtype = torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else None)
            mixed_precision_policy = (
                MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                if dtype else None
            )

            # Correctly map FSDP string -> ShardingStrategy
            strategy_map = {
                "no_shard": ShardingStrategy.NO_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "full_shard": ShardingStrategy.FULL_SHARD,
            }
            sharding = None
            for k in strategy_map:
                if k in self.args.fsdp:
                    sharding = strategy_map[k]
                    break

            if type(model) != FSDP:
                model = FSDP(
                    model,
                    sharding_strategy=sharding,
                    cpu_offload=cpu_offload,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                    device_id=self.args.device,
                )
        return model
 

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LogitsDistillTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
            
        )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        teacher_model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = data_args.max_train_samples
            if not data_args.streaming:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = data_args.max_eval_samples
            if not data_args.streaming:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
        
    # Whether to use argmax watermarking, and train using cross-entropy loss vs one-hot labels.
    argmax_watermark = None
        
    # Initialize watermarker
    if model_args.watermark_type == WatermarkType.AAR:
        watermarker = AarWatermark(
            vocab_size=len(tokenizer),
            k=model_args.aar_watermark_k,
            seed=model_args.watermark_seed,
            device=device,
        )
        argmax_watermark = True
        assert argmax_watermark, "Aar watermark only supports argmax watermarking"
    elif model_args.watermark_type == WatermarkType.KTH:
        watermarker = KTHWatermark(
            vocab_size=len(tokenizer),
            key_len=model_args.kth_watermark_key_len,
            seed=model_args.watermark_seed,
            device=device,
            num_shifts=model_args.kth_watermark_num_shifts,
        )
        argmax_watermark = True
        assert argmax_watermark, "KTH watermark only supports argmax watermarking"
    elif model_args.watermark_type == WatermarkType.KGW:
        watermarker = KGWWatermark(
            vocab=tokenizer.get_vocab().values(),
            gamma=model_args.kgw_watermark_gamma,
            delta=model_args.kgw_watermark_delta,
            seeding_scheme=model_args.kgw_watermark_seeding_scheme,
            processing_class=tokenizer,
            device=device,
        )
        argmax_watermark = False
        assert not argmax_watermark, "KGW watermark only supports non-argmax watermarking"
    elif model_args.watermark_type == WatermarkType.PRO:
        print(model_args.pro_mapping_file_path)
        watermarker = PROWatermark(
            device=device,
            tokenizer=tokenizer,
            gamma=model_args.pro_watermark_gamma,
            delta=model_args.pro_watermark_delta,
            chunk_size=model_args.pro_watermark_chunk_size,
            embedding_model=model_args.pro_embedding_model,
            transform_model_path=model_args.pro_transform_model_path,
            mapping_file_path=model_args.pro_mapping_file_path,
        )
        argmax_watermark = False
        assert not argmax_watermark, "PRO watermark only supports non-argmax watermarking"
    else:
        raise ValueError(f"Invalid watermark type: {model_args.watermark_type}")
    
    assert argmax_watermark is not None, "argmax_watermark must be set"

    # Load harmful dataset for booster modes
    harmful_dataset = None
    if training_args.robustness_mode in ("booster", "booster_anti"):
        from itertools import chain as iter_chain
        harmful_raw_dataset = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            split="train", streaming=False,
        )
        harmful_raw_dataset = harmful_raw_dataset.shuffle(seed=42).select(range(min(10000, len(harmful_raw_dataset))))
        harmful_tokenized = harmful_raw_dataset.map(
            tokenize_function, batched=True, remove_columns=harmful_raw_dataset.column_names,
        )
        harmful_dataset = harmful_tokenized.map(group_texts, batched=True)

    # Initialize our Trainer
    teacher_model = teacher_model.to(device)

    trainer = WatermarkLogitsDistillTrainer(
        teacher_model=teacher_model,
        watermarker=watermarker,
        argmax_watermark=argmax_watermark,
        model_args=model_args,
        harmful_dataset=harmful_dataset,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
    )
    
    trainer.save_optimizer = False


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if data_args.max_train_samples is not None:
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        if data_args.max_eval_samples is not None:
            metrics["eval_samples"] = data_args.max_eval_samples

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
