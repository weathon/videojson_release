#!/usr/bin/env python3
"""Isolated ROCm pilot: validate an exported token dataset, smoke, then one epoch.

The smoke adapter is disposable. Train starts again from the frozen base and requires
a matching successful smoke receipt. Neither mode templates or truncates dataset rows.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


SEED = 20261030
MAX_LENGTH = 16384
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
SETTINGS = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": TARGET_MODULES, "learning_rate": 5e-5,
    "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1,
    "num_train_epochs": 1, "gradient_checkpointing": True,
    "attn_implementation": "sdpa", "dtype": "bfloat16", "seed": SEED,
    "loss_type": "chunked_nll", "lr_scheduler_type": "constant",
    "max_length": MAX_LENGTH, "packing": False,
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def sets_only(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=sets_only) + "\n")
    temporary.replace(path)


def validate_row(row, number, vocab_size):
    ids, labels = row.get("input_ids"), row.get("labels")
    if not isinstance(ids, list) or not isinstance(labels, list) or len(ids) != len(labels):
        raise ValueError(f"row {number}: input_ids and labels must be equal-length lists")
    if not 2 <= len(ids) <= MAX_LENGTH:
        raise ValueError(f"row {number}: length {len(ids)} outside 2..{MAX_LENGTH}; no truncation")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError(f"row {number}: invalid token ID")
    if any(type(label) is not int or label not in (-100, token) for token, label in zip(ids, labels)):
        raise ValueError(f"row {number}: labels must be -100 or the corresponding input ID")
    if labels[0] != -100 or not any(label != -100 for label in labels[1:]):
        raise ValueError(f"row {number}: first label must be masked and shifted targets must be nonempty")
    if "attention_mask" in row and row["attention_mask"] != [1] * len(ids):
        raise ValueError(f"row {number}: exported attention mask must contain only ones")
    if row.get("sequence_sha256") != digest_json(ids):
        raise ValueError(f"row {number}: sequence_sha256 does not match exported input IDs")
    return {"input_ids": ids, "labels": labels}


def load_rows(path, vocab_size):
    rows, identities = [], []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"row {number}: empty JSONL line")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"row {number}: expected an object")
            row = validate_row(raw, number, vocab_size)
            rows.append(row)
            identities.append({
                "line": number, "sequence_sha256": digest_json(row["input_ids"]),
                "masked_example_sha256": digest_json(row), "tokens": len(row["input_ids"]),
                "target_tokens": sum(label != -100 for label in row["labels"][1:]),
                **{key: raw[key] for key in ("row_id", "qid", "episode_id", "question_id") if key in raw},
            })
    if not rows:
        raise ValueError("empty training dataset")
    return rows, identities


def collate_lists(rows, pad_token_id):
    """Right-pad without shifting, retokenizing, or reconstructing target labels."""
    width = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": [row["input_ids"] + [pad_token_id] * (width - len(row["input_ids"])) for row in rows],
        "labels": [row["labels"] + [-100] * (width - len(row["labels"])) for row in rows],
        "attention_mask": [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in rows],
    }


def smoke_rows(rows, vocab_size):
    # Synthetic kernel probes are separate from the unmodified, complete real example.
    probes = []
    for size in (256, 2048):
        ids = [32 + (index % 64) for index in range(size)]
        if max(ids) >= vocab_size:
            raise ValueError("vocabulary too small for fixed synthetic smoke tokens")
        probes.append({"input_ids": ids, "labels": [-100] * (size - 32) + ids[-32:]})
    longest = max(rows, key=lambda row: len(row["input_ids"]))
    probes.append(longest)
    return probes


def verify_smoke_receipt(receipt, contract):
    if receipt.get("mode") != "smoke" or receipt.get("status") != "passed":
        raise ValueError("train requires a passed smoke receipt")
    if receipt.get("contract") != contract:
        raise ValueError("smoke contract differs: dataset, base weights, runtime, or training settings changed")
    stages = receipt.get("stages", [])
    if len(stages) != 3 or not all(stage.get("gradient_norm", 0) > 0 and stage.get("weight_delta_max", 0) > 0 for stage in stages):
        raise ValueError("smoke receipt lacks three successful gradient/update stages")
    if not receipt.get("reload", {}).get("passed"):
        raise ValueError("smoke receipt lacks adapter reload/NLL verification")


def verify_serving_receipt(serving, smoke, smoke_path):
    if serving.get("status") != "passed" or any(serving.get(name) is not True for name in (
        "adapter_loaded", "adapter_effect_verified", "base_consistency_verified",
    )):
        raise ValueError("formal training requires verified serving load, adapter effect, and base consistency")
    if serving.get("smoke_receipt_sha256") != digest_file(smoke_path):
        raise ValueError("serving receipt belongs to a different training smoke")
    if serving.get("source_adapter_files_sha256") != digest_json(smoke["adapter_files"]):
        raise ValueError("serving receipt belongs to different source adapter files")


def expected_target_names(layer_types):
    names = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{index}"
        if layer_type == "full_attention":
            names.extend(f"{prefix}.self_attn.{name}" for name in TARGET_MODULES[:4])
        elif layer_type == "linear_attention":
            names.extend(f"{prefix}.linear_attn.{name}" for name in TARGET_MODULES[4:9])
        else:
            raise ValueError(f"unexpected decoder layer type: {layer_type}")
        names.extend(f"{prefix}.mlp.shared_expert.{name}" for name in TARGET_MODULES[9:])
    return sorted(names)


def verify_data_manifest(manifest, train_file, model_dir):
    if manifest.get("train_jsonl_sha256") != digest_file(train_file):
        raise ValueError("training JSONL differs from the data manifest")
    if manifest.get("max_sequence_tokens") != MAX_LENGTH:
        raise ValueError("data manifest has a different sequence budget")
    assets = manifest.get("tokenizer", {}).get("file_sha256")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("data manifest lacks tokenizer asset hashes")
    for name, expected in assets.items():
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir.resolve()) or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"exporter/model tokenizer mismatch: {name}")


def memory_budget(fraction, free_bytes, total_bytes):
    if not 0 < fraction < 1:
        raise ValueError("--memory-fraction must be between zero and one")
    cap_bytes, reserve_bytes = int(total_bytes * fraction), 56 << 30
    if free_bytes < cap_bytes + reserve_bytes:
        raise RuntimeError(f"insufficient GPU memory: free={free_bytes}, cap={cap_bytes}, reserve={reserve_bytes}")
    return cap_bytes, reserve_bytes


def base_provenance(model_dir, status):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values())
    files.update(name for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json") if (model_dir / name).exists())
    entries = {}
    for name in sorted(files):
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir.resolve()) or not path.is_file():
            raise ValueError(f"unsafe or missing model shard: {name}")
        status("hashing_base", file=name)
        entries[name] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
    return {"path": str(model_dir.resolve()), "files": entries, "content_sha256": digest_json(entries)}


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def status(stage, **details):
        record = {"stage": stage, "mode": args.mode, "time_unix": time.time(), "elapsed_seconds": time.time() - started, **details}
        write_json(output / "status.json", record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    def append_metric(kind, **details):
        record = {"kind": kind, "time_unix": time.time(), **details}
        with (output / "step_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    try:
        status("validating")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("this pilot supports exactly one training process/GPU")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "triton")}
        for name in ("flash-linear-attention", "fla-core", "causal-conv1d", "flash-attn", "safetensors"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        if versions["trl"] != "1.14.1":
            raise ValueError("driver is scoped to inspected TRL 1.14.1 chunked_nll API")
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import SequentialSampler
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer

        if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly one visible ROCm GPU; verify physical ownership before launching")
        if not 0 < args.memory_fraction < 1:
            raise ValueError("--memory-fraction must be between zero and one")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        cap_bytes, reserve_bytes = memory_budget(args.memory_fraction, free_bytes, total_bytes)
        write_json(output / "memory_preflight.json", {
            "free_bytes": free_bytes, "total_bytes": total_bytes, "allocator_cap_bytes": cap_bytes,
            "minimum_extra_free_bytes": reserve_bytes,
            "limitation": "allocator limit does not reserve memory or bound external ROCm/kernel allocations",
        })
        device = torch.cuda.get_device_properties(0)
        runtime = {"packages": versions, "hip": torch.version.hip, "gpu_name": device.name,
                   "gpu_memory_bytes": device.total_memory, "python": sys.version,
                   "gcn_arch": getattr(device, "gcnArchName", None)}
        write_json(output / "runtime.json", runtime)
        model_dir = Path(args.model).resolve()
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if config.model_type != "qwen3_5_moe":
            raise ValueError("expected the pinned Qwen3.5-MoE checkpoint")
        data_manifest = json.loads(Path(args.data_manifest).read_text())
        verify_data_manifest(data_manifest, args.train_file, model_dir)
        rows, identities = load_rows(args.train_file, config.get_text_config().vocab_size)
        provenance = base_provenance(model_dir, status)
        write_json(output / "base_provenance.json", provenance)
        write_json(output / "dataset_rows.json", identities)
        contract = {"dataset_sha256": digest_file(args.train_file), "data_manifest_sha256": digest_file(args.data_manifest),
                    "base_sha256": provenance["content_sha256"],
                    "settings": {**SETTINGS, "memory_fraction": args.memory_fraction, "allocator_cap_bytes": cap_bytes},
                    "runtime": runtime, "driver_sha256": digest_file(__file__)}
        write_json(output / "contract.json", contract)
        if args.mode == "train":
            if not args.smoke_receipt:
                raise ValueError("--smoke-receipt is required for train mode")
            smoke_receipt = json.loads(Path(args.smoke_receipt).read_text())
            verify_smoke_receipt(smoke_receipt, contract)
            if not args.serving_smoke_receipt:
                raise ValueError("--serving-smoke-receipt is required for train mode")
            verify_serving_receipt(json.loads(Path(args.serving_smoke_receipt).read_text()), smoke_receipt, args.smoke_receipt)
        set_seed(SEED)
        status("loading_text_model")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model, loading = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
        )
        write_json(output / "loading_info.json", loading)
        if type(model).__name__ != "Qwen3_5MoeForCausalLM" or loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("text-only checkpoint loading was not clean; inspect loading_info.json")
        if any("visual" in name or "vision_tower" in name for name, _ in model.named_parameters()):
            raise RuntimeError("unexpected vision parameters in text-only model")
        model.config.use_cache = False
        matched = sorted(name for name, module in model.named_modules()
                         if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_MODULES)
        if matched != expected_target_names(model.config.get_text_config().layer_types):
            raise RuntimeError("actual LoRA modules differ from the inspected attention/shared-expert surface")
        model_source = Path(inspect.getfile(type(model)))
        peft_config = LoraConfig(**{name: SETTINGS[name] for name in ("r", "lora_alpha", "lora_dropout", "target_modules")},
                                 task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        write_json(output / "effective_peft_config.json", peft_config.to_dict())
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise RuntimeError("non-LoRA or empty trainable parameter set")
        write_json(output / "peft_modules.json", {
            "scope": "attention and shared-expert projections only; router, shared_expert_gate and routed experts frozen",
            "target_modules": matched, "hf_model_source": str(model_source), "hf_model_source_sha256": digest_file(model_source),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "parameters": [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)} for name, parameter in trainable],
        })
        probes = smoke_rows(rows, config.get_text_config().vocab_size)
        selected = [row for probe in probes for row in [probe] * 4] if args.mode == "smoke" else rows

        def collator(batch):
            return {key: torch.tensor(value, dtype=torch.long) for key, value in collate_lists(batch, tokenizer.pad_token_id).items()}

        class PilotTrainer(SFTTrainer):
            microsteps = 0

            def _get_train_sampler(self, train_dataset=None):
                if args.mode == "smoke":
                    return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
                return super()._get_train_sampler(train_dataset)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                loss, result = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                if result.logits is not None:
                    raise RuntimeError("chunked loss gate failed: full logits were materialized")
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite loss")
                return (loss, result) if return_outputs else loss

            def training_step(self, model, inputs, num_items_in_batch=None):
                torch.cuda.synchronize()
                before = time.monotonic()
                result = super().training_step(model, inputs, num_items_in_batch)
                torch.cuda.synchronize()
                self.microsteps += 1
                append_metric("microstep", microstep=self.microsteps, optimizer_step=self.state.global_step,
                              sequence_length=inputs["input_ids"].shape[-1],
                              target_tokens=int((inputs["labels"][..., 1:] != -100).sum().item()),
                              backward_loss=float(result.detach().float().item()), seconds=time.monotonic() - before,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
                return result

        stages = []

        class Metrics(TrainerCallback):
            def on_step_begin(self, training_args, state, control, **kwargs):
                self.started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                status("training", optimizer_step=state.global_step, total_steps=state.max_steps)

            def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
                active = [(name, parameter) for name, parameter in trainable if parameter.grad is not None]
                if not active:
                    raise RuntimeError("no adapter gradients")
                norms = torch.stack([parameter.grad.detach().float().norm() for _, parameter in active])
                self.norm = float(norms.norm().item())
                if not math.isfinite(self.norm) or self.norm <= 0:
                    raise RuntimeError("nonfinite or zero adapter gradient")
                self.name, self.parameter = active[int(norms.argmax().item())]
                self.before = self.parameter.detach().float().cpu().clone()

            def on_step_end(self, training_args, state, control, **kwargs):
                delta = float((self.parameter.detach().float().cpu() - self.before).abs().max().item())
                if not math.isfinite(delta) or delta <= 0:
                    raise RuntimeError("optimizer failed to change the selected nonzero-gradient adapter weight")
                record = {"step": state.global_step, "gradient_norm": self.norm, "weight_delta_max": delta,
                          "checked_parameter": self.name, "seconds": time.monotonic() - self.started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                if args.mode == "smoke":
                    index = state.global_step - 1
                    record.update(stage=["synthetic_256", "synthetic_2048", "real_longest"][index],
                                  tokens=len(probes[index]["input_ids"]), sequence_sha256=digest_json(probes[index]["input_ids"]))
                stages.append(record)
                append_metric("optimizer_step", **record)
                write_json(output / "optimizer_steps.json", stages)

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                append_metric("trainer_log", step=state.global_step, values=logs or {})

        training_args = SFTConfig(
            output_dir=str(output / "trainer"), num_train_epochs=1,
            max_steps=3 if args.mode == "smoke" else -1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=5e-5, lr_scheduler_type="constant", warmup_steps=0,
            bf16=True, fp16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=SEED, data_seed=SEED, report_to=[], save_strategy="no", eval_strategy="no",
            logging_steps=1, logging_nan_inf_filter=False, optim="adamw_torch",
            max_length=MAX_LENGTH, packing=False, padding_free=False, shuffle_dataset=False,
            dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
            assistant_only_loss=False, completion_only_loss=False,
            loss_type="chunked_nll", router_aux_loss_coef=None, use_liger_kernel=False,
            dataloader_num_workers=0, disable_tqdm=True,
        )
        trainer = PilotTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(selected),
                               processing_class=tokenizer, data_collator=collator, callbacks=[Metrics()])
        write_json(output / "effective_sft_config.json", trainer.args.to_dict())
        trainer.train(resume_from_checkpoint=False)
        if args.mode == "smoke" and len(stages) != 3:
            raise RuntimeError("smoke did not finish all three stages")
        status("checking_target_nll")
        real_row = probes[-1]

        def target_nll():
            trainer.model.eval()
            batch = trainer._prepare_inputs(collator([real_row]))
            with torch.no_grad(), trainer.compute_loss_context_manager():
                loss, result = trainer.compute_loss(trainer.model, batch, return_outputs=True)
            aux = result.aux_loss
            coefficient = trainer.model.config.get_text_config().router_aux_loss_coef
            aux_value = float(aux.item()) if aux is not None else 0.0
            nll = float(loss.item()) - coefficient * aux_value
            if not math.isfinite(nll):
                raise RuntimeError("nonfinite target NLL")
            if int(result.num_valid_tokens.item()) != sum(label != -100 for label in real_row["labels"][1:]):
                raise RuntimeError("chunked loss target count differs from exported labels")
            return {"nll": nll, "loss_with_aux": float(loss.item()), "aux_loss": aux_value,
                    "target_tokens": int(result.num_valid_tokens.item()), "sequence_sha256": digest_json(real_row["input_ids"]),
                    "masked_example_sha256": digest_json(real_row)}

        adapted = target_nll()
        with trainer.model.disable_adapter():
            base_nll = target_nll()
        status("saving_adapter")
        final = output / "adapter"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(final)
        trainer.state.save_to_json(str(output / "trainer_state.json"))
        saved = {str(path.relative_to(final)): {"bytes": path.stat().st_size, "sha256": digest_file(path)}
                 for path in sorted(final.rglob("*")) if path.is_file()}
        if "adapter_model.safetensors" not in saved or "adapter_config.json" not in saved:
            raise RuntimeError("adapter save is incomplete")
        status("reloading_adapter")
        # Discard the in-memory adapter before reloading its bytes onto the unchanged base.
        for name in list(trainer.model.peft_config):
            trainer.model.delete_adapter(name)
        trainer.model.load_adapter(str(final), adapter_name="default", is_trainable=False)
        trainer.model.set_adapter("default")
        reloaded = target_nll()
        reload_passed = math.isclose(adapted["nll"], reloaded["nll"], rel_tol=2e-4, abs_tol=2e-3)
        if not reload_passed:
            raise RuntimeError(f"saved adapter NLL mismatch: before={adapted['nll']} after={reloaded['nll']}")
        receipt = {"status": "passed", "mode": args.mode, "contract": contract, "stages": stages,
                   "base_target": base_nll, "adapted_target": adapted,
                   "reload": {"passed": reload_passed, "target": reloaded, "abs_tolerance": 2e-3, "relative_tolerance": 2e-4},
                   "adapter_files": saved, "optimizer_steps": trainer.state.global_step,
                   "effective_config_sha256": {name: digest_file(output / name) for name in
                                               ("effective_sft_config.json", "effective_peft_config.json")},
                   "training_rows": len(rows), "longest_real_tokens": len(real_row["input_ids"]),
                   "elapsed_seconds": time.time() - started,
                   "meaning": "training/reload checks only; target NLL is not held-out task evaluation"}
        write_json(output / "save_receipt.json", receipt)
        status("completed", receipt=str(output / "save_receipt.json"))
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        status("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--model", required=True, help="local pinned BF16 model directory")
    parser.add_argument("--train-file", required=True, help="exporter train.jsonl with input_ids and labels")
    parser.add_argument("--data-manifest", required=True, help="exporter manifest containing dataset and tokenizer hashes")
    parser.add_argument("--output-dir", required=True, help="must not exist")
    parser.add_argument("--smoke-receipt", help="successful matching smoke/save_receipt.json, required for train")
    parser.add_argument("--serving-smoke-receipt", help="matching verified serving receipt, required for train")
    parser.add_argument("--memory-fraction", type=float, default=0.70, help="PyTorch GPU allocator fraction, default 0.70")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
