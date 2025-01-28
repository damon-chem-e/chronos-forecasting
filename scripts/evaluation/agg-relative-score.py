import pandas as pd
import typer
from scipy.stats import gmean
from pathlib import Path

app = typer.Typer(pretty_exceptions_enable=False)
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def agg_relative_score(model_csv: Path, baseline_csv: Path):
    model_df = pd.read_csv(model_csv).set_index("dataset")
    baseline_df = pd.read_csv(baseline_csv).set_index("dataset")
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
        "model", axis="columns"
    )
    return relative_score.agg(gmean)


@app.command()
def main(
    model_name: str,
    baseline_name: str = "seasonal-naive",
    results_dir: Path = DEFAULT_RESULTS_DIR,
):
    """
    Compute the aggregated relative score as reported in the Chronos paper.
    Results will be saved to {results_dir}/{model_name}-agg-rel-scores.csv

    Parameters
    ----------
    model_name : str
        Name of the model used in the CSV files. The in-domain and zero-shot CSVs
        are expected to be named {model_name}-in-domain.csv and {model_name}-zero-shot.csv.
    results_dir : Path, optional, default = results/
        Directory where results CSVs generated by evaluate.py are stored
    """
    
    # in_domain_agg_score_df = agg_relative_score(
    #     results_dir / f"{model_name}-in-domain.csv",
    #     results_dir / f"{baseline_name}-in-domain.csv",
    # )
    # in_domain_agg_score_df.name = "value"
    # in_domain_agg_score_df.index.name = "metric"

    # zero_shot_agg_score_df = agg_relative_score(
    #     results_dir / f"{model_name}-zero-shot.csv",
    #     results_dir / f"{baseline_name}-zero-shot.csv",
    # )
    # zero_shot_agg_score_df.name = "value"
    # zero_shot_agg_score_df.index.name = "metric"

    # agg_score_df = pd.concat(
    #     {"in-domain": in_domain_agg_score_df, "zero-shot": zero_shot_agg_score_df},
    #     names=["benchmark"],
    # )
    # agg_score_df.to_csv(f"{results_dir}/{model_name}-agg-rel-scores.csv")

    # Check if the file exists before processing each benchmark
    in_domain_path = results_dir / f"{model_name}-in-domain.csv"
    zero_shot_path = results_dir / f"{model_name}-zero-shot.csv"

    print("Looking for in-domain benchmark at:", in_domain_path, "and zero-shot benchmark at:", zero_shot_path)

    # Initialize a dictionary to hold valid DataFrames
    valid_benchmarks = {}

    try:
        in_domain_agg_score_df = agg_relative_score(
            in_domain_path,
            results_dir / f"{baseline_name}-in-domain.csv",
        )
        in_domain_agg_score_df.name = "value"
        in_domain_agg_score_df.index.name = "metric"
        valid_benchmarks["in-domain"] = in_domain_agg_score_df
    except:
        print("No in-domain benchmark found.") 

    try:
        if os.path.exists(zero_shot_path):
            zero_shot_agg_score_df = agg_relative_score(
                zero_shot_path,
                results_dir / f"{baseline_name}-zero-shot.csv",
            )
            zero_shot_agg_score_df.name = "value"
            zero_shot_agg_score_df.index.name = "metric"
            valid_benchmarks["zero-shot"] = zero_shot_agg_score_df
    except:
        print("No zero-shot benchmark found.")

    # Concatenate only the valid benchmarks
    if valid_benchmarks:
        agg_score_df = pd.concat(valid_benchmarks, names=["benchmark"])
        agg_score_df.to_csv(f"{results_dir}/{model_name}-agg-rel-scores.csv")
    else:
        print("No valid benchmarks found. No output file generated.")

if __name__ == "__main__":
    app()
