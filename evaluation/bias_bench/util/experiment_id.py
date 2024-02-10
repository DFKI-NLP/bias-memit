"""
:authors: Nicholas Meade, Elinor Poole-Dayan, Siva Reddy (Repository: https://github.com/McGill-NLP/bias-bench/tree/main)
"""

def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    
    if "/" in experiment_id:
        experiment_id.replace("/", "-")

    return experiment_id
