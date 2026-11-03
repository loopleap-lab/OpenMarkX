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

import datasets
import evaluate
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

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
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from watermarks.pro.pro_watermark import PROWatermark
from watermarks.watermark_types import WatermarkType


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."},
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
            "help": ("Type of watermark to use."),
            "choices": [WatermarkType.PRO],
        },
    )
    pro_watermark_gamma: float = field(
        default=0.5,
        metadata={"help": "Gamma for PRO watermark, i.e. proportion of vocab that is greenlist."},
    )
    pro_watermark_delta: float = field(
        default=2.0,
        metadata={"help": "Delta for PRO watermark, i.e. value to add to logits in greenlist."},
    )
    pro_watermark_chunk_size: Optional[int] = field(
        default=10,
        metadata={"help": "Chunk size for PRO watermark processing."},
    )
    pro_embedding_model: Optional[str] = field(
        default=None,
        metadata={"help": "Embedding model to use for PRO watermark."},
    )
    pro_transform_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to transform model for PRO watermark."},
    )
    pro_mapping_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to mapping file for PRO watermark."},
    )
    transform_lr: float = field(default=1e-4)
    transform_weight_decay: float = field(default=0.01)
    transform_warmup_steps: int = field(default=100)
    transform_loss_weight: float = field(
        default=1e-3,
        metadata={"help": "Weight for the transform model regularization loss."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )




@dataclass
class DataTrainingArguments:
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
        metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples.")},
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
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
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
    save_checkpoint_models: bool = field(
        default=True,
        metadata={"help": "Save model at every checkpoint, no deletion, no optimizer states."},
    )
    tanh_temp_init: float = field(
        default=10.0,
        metadata={"help": "Initial tanh temperature (low = smooth gradients for co-optimization)."},
    )
    tanh_temp_max: float = field(
        default=1000.0,
        metadata={"help": "Final tanh temperature (high = sharp ±1 for strong watermark)."},
    )
    tanh_anneal_steps: int = field(
        default=0,
        metadata={"help": "Steps to anneal tanh temperature from init to max. 0 = use max_steps."},
    )


class WatermarkLogitsDistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        watermarker: PROWatermark,
        model_args=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model = self._fsdp_teacher_model(self.teacher_model)
        self.teacher_model.eval()

        # Freeze teacher model parameters to save GPU memory
        # No need to compute gradients for frozen teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        logger.info("Teacher model parameters frozen (requires_grad=False)")

        self.watermarker = watermarker
        self.model_args = model_args
        self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        # Tanh temperature annealing for gradient flow
        self.tanh_temp_init = self.args.tanh_temp_init
        self.tanh_temp_max = self.args.tanh_temp_max
        self.tanh_anneal_steps = self.args.tanh_anneal_steps or self.args.max_steps
        if hasattr(self.watermarker, 'set_tanh_temperature'):
            self.watermarker.set_tanh_temperature(self.tanh_temp_init)
            logger.info(f"Tanh temperature annealing: {self.tanh_temp_init} -> {self.tanh_temp_max} over {self.tanh_anneal_steps} steps")

        # Track accumulation steps for periodic logging
        self.current_accumulation_step = 0
        self.transform_loss_weight = getattr(self.model_args, "transform_loss_weight", 1e-3)

    def _save_optimizer_and_scheduler(self, output_dir):
        """Save optimizer and scheduler states for proper checkpoint resuming."""
        if self.optimizer is not None:
            optimizer_path = os.path.join(output_dir, "optimizer.pt")
            try:
                # Save optimizer state dict
                # This includes states for BOTH LLM and transform model (via CombinedModel)
                # For FSDP, the Trainer's parent class handles the special state dict conversion
                optimizer_state = self.optimizer.state_dict()
                torch.save(optimizer_state, optimizer_path)

                # Log optimizer state size for verification
                file_size = os.path.getsize(optimizer_path) / (1024 ** 2)  # MB
                logger.info(f"Optimizer state saved to {optimizer_path} ({file_size:.1f} MB)")
            except Exception as e:
                logger.warning(f"Failed to save optimizer state: {e}")
                # Don't fail training if optimizer save fails

        if self.lr_scheduler is not None:
            scheduler_path = os.path.join(output_dir, "scheduler.pt")
            try:
                scheduler_state = self.lr_scheduler.state_dict()
                torch.save(scheduler_state, scheduler_path)
                logger.info(f"Scheduler state saved to {scheduler_path}")
            except Exception as e:
                logger.warning(f"Failed to save scheduler state: {e}")

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if hasattr(self.watermarker, "transform_model") and self.watermarker.transform_model is not None:

            class CombinedModel(torch.nn.Module):
                def __init__(self, model1, model2):
                    super().__init__()
                    self.model1 = model1
                    self.model2 = model2

                def named_parameters(self, prefix: str = "", recurse: bool = True):
                    for name, param in self.model1.named_parameters(prefix=prefix + "model.", recurse=recurse):
                        yield name, param
                    for name, param in self.model2.named_parameters(prefix=prefix + "transform_model.", recurse=recurse):
                        yield name, param

            opt_model = CombinedModel(opt_model, self.watermarker.transform_model)

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
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
        scaled_output = self.asymmetric_scale(output, gamma)
        row_means = torch.mean(scaled_output, dim=1)
        col_means = torch.mean(scaled_output, dim=0)
        row_penalty = row_means.pow(2).mean()
        col_penalty = col_means.pow(2).mean()
        return row_penalty + col_penalty

    def abs_value_penalty(self, output):
        penalties = torch.relu(0.05 - torch.abs(output))
        mask = (penalties > 0).float()
        non_zero_count = torch.max(mask.sum(), torch.tensor(1.0, device=output.device))
        return (penalties * mask).sum() / non_zero_count

    def compute_transform_loss(self, transformed_embeddings: torch.FloatTensor) -> torch.Tensor:
        self.current_accumulation_step += 1

        # Compute loss on EVERY step to ensure transform model gets same gradient signal as student
        # Use current micro-batch for per-batch regularization
        scaled_output = self.asymmetric_scale(transformed_embeddings, gamma=self.model_args.pro_watermark_gamma)
        row_means = torch.mean(scaled_output, dim=1)
        col_means = torch.mean(scaled_output, dim=0)
        row_penalty = row_means.pow(2).mean()
        col_penalty = col_means.pow(2).mean()
        mean_penalty = row_penalty + col_penalty
        range_penalty = self.abs_value_penalty(transformed_embeddings)
        transform_loss = mean_penalty + range_penalty

        # Log detailed metrics only on last accumulation step to avoid spam
        if self.current_accumulation_step >= self.args.gradient_accumulation_steps:
            self.log(
                {
                    "loss/transform/row_penalty": row_penalty.item(),
                    "loss/transform/col_penalty": col_penalty.item(),
                    "loss/transform/mean_penalty": mean_penalty.item(),
                    "loss/transform/range_penalty": range_penalty.item(),
                    "loss/transform/total_raw": transform_loss.item(),
                }
            )
            # Reset counter after logging
            self.current_accumulation_step = 0

        return transform_loss

    def _update_tanh_temperature(self):
        """Anneal tanh temperature based on current training step (linear schedule)."""
        if not hasattr(self.watermarker, 'set_tanh_temperature'):
            return
        step = self.state.global_step
        progress = min(step / max(self.tanh_anneal_steps, 1), 1.0)
        tau = self.tanh_temp_init + progress * (self.tanh_temp_max - self.tanh_temp_init)
        self.watermarker.set_tanh_temperature(tau)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("labels", None)

        # Anneal tanh temperature
        self._update_tanh_temperature()

        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        watermarked_logits = self.watermarker.watermark_logits(
            inputs["input_ids"], teacher_outputs.logits
        )
        log_p_student = F.log_softmax(outputs.logits, dim=-1)
        p_target = F.softmax(watermarked_logits, dim=-1)
        student_loss = self.loss_fct(log_p_student, p_target)

        transform_loss = torch.tensor(0.0, device=outputs.logits.device, requires_grad=True)
        if hasattr(self.watermarker, "transform_model"):
            # watermark_logits() already computed and cached transformed embeddings
            transformed = self.watermarker.last_transformed_embeddings
            transform_loss = self.compute_transform_loss(transformed)

        total_loss = student_loss + self.transform_loss_weight * transform_loss

        log_dict = {
            "loss/student": student_loss.item(),
            "loss/total": total_loss.item(),
            "tanh_temperature": self.watermarker.get_tanh_temperature(),
        }
        if transform_loss.requires_grad and transform_loss.item() != 0.0:
            log_dict["loss/transform_weighted"] = (self.transform_loss_weight * transform_loss).item()
        self.log(log_dict)

        return (total_loss, outputs) if return_outputs else total_loss

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save complete checkpoint including model, optimizer, scheduler, and transform model."""
        # Call parent to save student LLM, optimizer, and scheduler
        super()._save_checkpoint(model, trial)

        # Save transform model weights separately for easy loading
        if hasattr(self.watermarker, "transform_model") and self.watermarker.transform_model is not None:
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            transform_path = os.path.join(output_dir, "transform_model.pt")
            torch.save(self.watermarker.transform_model.state_dict(), transform_path)
            logger.info(f"Transform model saved to: {transform_path}")

        # Verify checkpoint completeness
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        expected_files = ["pytorch_model.bin", "optimizer.pt", "scheduler.pt", "transform_model.pt"]
        saved_files = [f for f in expected_files if os.path.exists(os.path.join(output_dir, f))]
        logger.info(f"Checkpoint saved with files: {saved_files}")

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
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

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
            mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype) if dtype else None

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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LogitsDistillTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
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
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
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

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
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

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

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
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    if model_args.watermark_type == WatermarkType.PRO:
        print(model_args.pro_mapping_file_path)

        # Determine transform model path: use checkpoint if resuming, otherwise use pre-trained
        transform_model_path = model_args.pro_transform_model_path
        checkpoint_to_resume = training_args.resume_from_checkpoint or last_checkpoint

        if checkpoint_to_resume is not None:
            checkpoint_transform_path = os.path.join(checkpoint_to_resume, "transform_model.pt")
            if os.path.exists(checkpoint_transform_path):
                transform_model_path = checkpoint_transform_path
                logger.info(f"Resuming transform model from checkpoint: {checkpoint_transform_path}")
            else:
                logger.warning(
                    f"Transform model checkpoint not found at {checkpoint_transform_path}. "
                    f"Loading from original path: {transform_model_path}"
                )

        watermarker = PROWatermark(
            device=device,
            tokenizer=tokenizer,
            gamma=model_args.pro_watermark_gamma,
            delta=model_args.pro_watermark_delta,
            chunk_size=model_args.pro_watermark_chunk_size,
            embedding_model=model_args.pro_embedding_model,
            transform_model_path=transform_model_path,
            mapping_file_path=model_args.pro_mapping_file_path,
        )
    else:
        raise ValueError(f"Invalid watermark type: {model_args.watermark_type}")

    teacher_model = teacher_model.to(device)

    trainer = WatermarkLogitsDistillTrainer(
        teacher_model=teacher_model,
        watermarker=watermarker,
        model_args=model_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    # Enable optimizer saving for proper checkpoint resuming
    # Optimizer includes states for both LLM and transform model
    trainer.save_optimizer = True

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # Verify checkpoint files if resuming
        if checkpoint is not None:
            logger.info(f"Resuming training from checkpoint: {checkpoint}")
            expected_files = {
                "pytorch_model.bin": "Student LLM weights",
                "optimizer.pt": "Optimizer state (LLM + Transform)",
                "scheduler.pt": "Learning rate scheduler state",
                "transform_model.pt": "Transform model weights"
            }
            for filename, description in expected_files.items():
                filepath = os.path.join(checkpoint, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath) / (1024 ** 2)  # MB
                    logger.info(f"  ✓ {description}: {filename} ({file_size:.1f} MB)")
                else:
                    logger.warning(f"  ✗ {description}: {filename} NOT FOUND")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        if data_args.max_train_samples is not None:
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

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
    main()


if __name__ == "__main__":
    main()
