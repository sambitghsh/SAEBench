import argparse
import json
import os
import random
import traceback

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.sparse_probing_sae_probes.main as sparse_probing_sae_probes
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils

MODEL_CONFIGS = {
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
    },
    "pythia-160m-deduped": {
        "batch_size": 256,
        "dtype": "float32",
        "layers": [6],
        "d_model": 768,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [5, 12, 19],
        "d_model": 2304,
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "sparse_probing_sae_probes": "eval_results/sparse_probing_sae_probes",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


TRAINER_LOADERS = {
    "MatryoshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    "BatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "TopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    "StandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    "StandardTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "PAnnealTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "JumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    "GatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
}

ALL_EVAL_TYPES = [
    "absorption",
    "core",
    "scr",
    "tpp",
    "sparse_probing",
    "sparse_probing_sae_probes",
    "autointerp",
    "unlearning",
    "ravel",
]

# sae-probes (used by sparse_probing_sae_probes) doesn't expose a dtype
# option in its eval config and internally runs the model / caches
# activations in float32. If the SAE itself is loaded in bfloat16 (e.g.
# for gemma-2-2b), the resulting matmul between float32 activations and
# bfloat16 SAE weights fails with a dtype mismatch. Force float32 just for
# this eval type's SAE loading; every other eval type is unaffected.
EVAL_TYPE_DTYPE_OVERRIDES = {
    "sparse_probing_sae_probes": "float32",
}


def get_all_hf_repo_autoencoders(
    repo_id: str, download_location: str = "downloaded_saes"
) -> list[str]:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))
    config_dir = snapshot_download(
        repo_id,
        allow_patterns=["*config.json"],
        local_dir=download_location,
        force_download=False,
    )

    config_locations = []

    for root, _, files in os.walk(config_dir):
        for file in sorted(files):
            if file == "config.json":
                config_locations.append(os.path.join(root, file))

    config_locations.sort()

    repo_locations = []

    for config in config_locations:
        repo_location = config.split(f"{download_location}/")[1].split("/config.json")[
            0
        ]
        repo_locations.append(repo_location)

    return repo_locations


def load_dictionary_learning_sae(
    repo_id: str,
    location: str,
    model_name,
    device: str,
    dtype: torch.dtype,
    layer: int | None = None,
    download_location: str = "downloaded_saes",
) -> base_sae.BaseSAE:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))

    config_file = f"{download_location}/{location}/config.json"

    with open(config_file) as f:
        config = json.load(f)

    trainer_class = config["trainer"]["trainer_class"]

    location = f"{location}/ae.pt"

    context_size = config.get("buffer", {}).get("ctx_len", 128)

    sae = TRAINER_LOADERS[trainer_class](
        repo_id=repo_id,
        filename=location,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )

    sae.cfg.context_size = context_size
    return sae


def verify_saes_load(
    repo_id: str,
    sae_locations: list[str],
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """Verify that all SAEs load correctly. Useful to check this before a big evaluation run."""
    for sae_location in sae_locations:
        sae = load_dictionary_learning_sae(
            repo_id=repo_id,
            location=sae_location,
            layer=None,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        del sae


def run_evals(
    repo_id: str,
    model_name: str,
    sae_locations: list[str],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    random_seed: int,
    api_key: str | None = None,
    force_rerun: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda selected_saes, is_final: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda selected_saes, is_final: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,  # type: ignore
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        # The core run_eval() interface isn't well suited for custom SAEs, so we have to do this instead.
        # It isn't ideal, but it works.
        # TODO: Don't hardcode magic numbers
        "core": (
            lambda selected_saes, is_final: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
                random_seed=random_seed,
            )
        ),
        "ravel": (
            lambda selected_saes, is_final: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size // 4,
                    llm_dtype=llm_dtype,
                    entity_attribute_selection={"city": ["Country", "Continent", "Language"]},
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
        "scr": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "tpp": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "sparse_probing": (
            lambda selected_saes, is_final: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "sparse_probing_sae_probes": (
            # NOTE: SparseProbingSaeProbesEvalConfig has no dtype/llm_dtype
            # field (confirmed via dataclasses.fields()), so we do NOT pass
            # llm_dtype here -- doing so raises a validation error. The
            # dtype mismatch this eval hits is instead fixed by loading the
            # SAE itself in float32, see EVAL_TYPE_DTYPE_OVERRIDES below.
            lambda selected_saes, is_final: sparse_probing_sae_probes.run_eval(
                sparse_probing_sae_probes.SparseProbingSaeProbesEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing_sae_probes",
                force_rerun,
            )
        ),
        "unlearning": (
            lambda selected_saes, is_final: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name="gemma-2-2b-it",
                    random_seed=random_seed,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size
                    // 8,  # 8x smaller batch size for unlearning due to longer sequences
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    verify_saes_load(
        repo_id,
        sae_locations,
        model_name,
        device,
        general_utils.str_to_dtype(llm_dtype),
    )

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        sae_load_dtype = general_utils.str_to_dtype(
            EVAL_TYPE_DTYPE_OVERRIDES.get(eval_type, llm_dtype)
        )

        try:
            for i, sae_location in enumerate(sae_locations):
                is_final = False
                if i == len(sae_locations) - 1:
                    is_final = True

                sae = load_dictionary_learning_sae(
                    repo_id=repo_id,
                    location=sae_location,
                    layer=None,
                    model_name=model_name,
                    device=device,
                    dtype=sae_load_dtype,
                )
                unique_sae_id = sae_location.replace("/", "_")
                unique_sae_id = f"{repo_id.split('/')[1]}_{unique_sae_id}"
                selected_saes = [(unique_sae_id, sae)]

                os.makedirs(output_folders[eval_type], exist_ok=True)
                eval_runners[eval_type](selected_saes, is_final)

                del sae

        except Exception as e:
            print(f"Error running {eval_type} evaluation: {e}")
            traceback.print_exc()
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAE Bench evaluations on dictionary_learning SAEs hosted on HuggingFace."
    )
    parser.add_argument(
        "--repo",
        dest="repos",
        action="append",
        nargs=2,
        metavar=("REPO_ID", "MODEL_NAME"),
        required=True,
        help=(
            "A HuggingFace repo_id followed by its model_name (must be a key in "
            "MODEL_CONFIGS, e.g. pythia-160m-deduped or gemma-2-2b). "
            "Repeat this flag to evaluate multiple repos, e.g. "
            "--repo org/repo1 pythia-160m-deduped --repo org/repo2 gemma-2-2b"
        ),
    )
    parser.add_argument(
        "--eval_types",
        nargs="+",
        choices=ALL_EVAL_TYPES,
        default=["core", "scr", "tpp", "sparse_probing", "sparse_probing_sae_probes", "ravel", "unlearning"],
        help="Which evaluations to run. Space-separated list.",
    )
    parser.add_argument(
        "--exclude_keywords",
        nargs="*",
        default=["checkpoints"],
        help="SAE folder names containing any of these keywords are excluded.",
    )
    parser.add_argument(
        "--include_keywords",
        nargs="*",
        default=[],
        help="If set, only SAE folder names containing at least one of these keywords are included.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-run evaluations even if cached results already exist.",
    )
    parser.add_argument(
        "--openai_api_key_file",
        default="openai_api_key.txt",
        help="Path to a file containing your OpenAI API key (only needed for autointerp).",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    """Seed every RNG used across the repo and enable stricter PyTorch determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    """
    This will run all evaluations on all selected dictionary_learning SAEs within the specified HuggingFace repos.
    Set repos and eval types via CLI args, e.g.:

        python run_eval_dict.py \
            --repo sam01ghsh/experiments_pythia-160m-deduped_matryoshka_batch_top_k_random_subset pythia-160m-deduped \
            --eval_types core scr tpp sparse_probing sparse_probing_sae_probes ravel

    You can pass multiple --repo flags to evaluate several repos in one run.
    NOTE: If your model (with associated model_name and batch sizes) is not in the MODEL_CONFIGS dictionary, you will need to add it.
    This relies on each SAE being located in a folder which contains an ae.pt file and a config.json file (which is the default save format in dictionary_learning).
    """
    args = parse_args()

    set_deterministic(args.random_seed)

    device = general_utils.setup_environment()

    eval_types = args.eval_types

    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    if "autointerp" in eval_types:
        try:
            with open(args.openai_api_key_file) as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception(
                f"Please create {args.openai_api_key_file} with your API key"
            )
    else:
        api_key = None

    if "unlearning" in eval_types:
        if not os.path.exists(
            "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
        ):
            raise Exception(
                "Please download bio-forget-corpus.jsonl for unlearning evaluation"
            )

    repos = [(repo_id, model_name) for repo_id, model_name in args.repos]

    exclude_keywords = args.exclude_keywords
    include_keywords = args.include_keywords

    for repo_id, model_name in repos:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. Add it to MODEL_CONFIGS first."
            )

        print(f"\n\n\nEvaluating {model_name} with {repo_id}\n\n\n")

        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]
        torch_dtype = general_utils.str_to_dtype(str_dtype)

        sae_locations = get_all_hf_repo_autoencoders(repo_id)

        sae_locations = general_utils.filter_keywords(
            sae_locations,
            exclude_keywords=exclude_keywords,
            include_keywords=include_keywords,
        )

        run_evals(
            repo_id=repo_id,
            model_name=model_name,
            sae_locations=sae_locations,
            llm_batch_size=llm_batch_size,
            llm_dtype=str_dtype,
            device=device,
            eval_types=eval_types,
            api_key=api_key,
            random_seed=args.random_seed,
            force_rerun=args.force_rerun,
        )


# Example usage:
#
# CUDA_VISIBLE_DEVICES=0 python run_eval_dict.py \
#   --repo sam01ghsh/experiments_gemma-2-2b_jump_relu_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_matryoshka_batch_top_k_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_batch_top_k_random_subset gemma-2-2b \
#   --eval_types absorption core scr tpp sparse_probing sparse_probing_sae_probes ravel unlearning


# CUDA_VISIBLE_DEVICES=1 python run_eval_dict.py \
#   --repo sam01ghsh/experiments_gemma-2-2b_jump_relu_baseline gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_jump_relu_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_matryoshka_batch_top_k_baseline gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_matryoshka_batch_top_k_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_batch_top_k_baseline gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_batch_top_k_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_standard_new_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_standard_new_baseline gemma-2-2b \
#   --eval_types autointerp

# CUDA_VISIBLE_DEVICES=2 python run_eval_dict.py \
#   --repo 'sam01ghsh/experiments_gemma-2-2b_jump_relu_baseline' gemma-2-2b \
#   --eval_types unlearning

# CUDA_VISIBLE_DEVICES=2 python run_eval_dict.py \
#   --repo sam01ghsh/experiments_gemma-2-2b_standard_new_random_subset gemma-2-2b \
#   --repo sam01ghsh/experiments_gemma-2-2b_standard_new_baseline gemma-2-2b \
#   --eval_types absorption core scr tpp sparse_probing sparse_probing_sae_probes ravel unlearning



# #
# python evaluate_saes.py \
#   --repo sam01ghsh/experiments_gemma-2-2b_matryoshka_batch_top_k_random_subset gemma-2-2b \
#   --eval_types absorption


# python run_eval_dict.py \
#   --repo sam01ghsh/experiments_gemma-2-2b_matryoshka_batch_top_k_random_subset gemma-2-2b \
#   --eval_types absorption core scr tpp ravel unlearning


# CUDA_VISIBLE_DEVICES=0 python run_eval_dict.py \
#   --repo madelynmathai/coreset-sweep-saes gemma-2-2b \
#   --include_keywords "gemma-2-2b" \
#   --eval_types ravel scr tpp



# CUDA_VISIBLE_DEVICES=0 python run_eval_dict.py \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_matryoshka_batch_top_k_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_jump_relu_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_batch_top_k_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_standard_new_baseline pythia-160m-deduped \
#   --eval_types absorption core scr tpp sparse_probing sparse_probing_sae_probes ravel unlearning autointerp

# CUDA_VISIBLE_DEVICES=2 python run_eval_dict.py \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_matryoshka_batch_top_k_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_jump_relu_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_batch_top_k_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_standard_new_baseline pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_matryoshka_batch_top_k_random_subset pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_jump_relu_random_subset pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_batch_top_k_random_subset pythia-160m-deduped \
#   --repo sam01ghsh/experiments_pythia-160m-deduped_standard_new_random_subset pythia-160m-deduped \
#   --eval_types absorption core scr tpp sparse_probing sparse_probing_sae_probes ravel unlearning autointerp
