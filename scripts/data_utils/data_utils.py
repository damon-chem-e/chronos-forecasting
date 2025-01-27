import pandas as pd
from gluonts.dataset.arrow import ArrowWriter

def huggingface_to_arrow(input_path: str, output_path: str):
    """
    Takes a huggingface dataset from https://huggingface.co/datasets/autogluon/chronos_datasets
    and converts it to a GluonTS-compatible arrow file.
    
    Arguments
    ---------------
    input_path: str
        The path to the huggingface dataset 
        (i.e. a parquet file for a single dataset from the huggingface set).
    output_path: str
        The path to save the arrow file to.
    """
    df = pd.read_parquet(input_path)
    # Get first target length to verify all are same length
    first_target_len = len(df.loc[0, 'target'])
    first_timestamp_len = len(df.loc[0, 'timestamp'])
    
    # Verify all targets and timestamps have same length
    assert all(len(target) == first_target_len for target in df['target']), "All time series must have same length"
    assert all(len(ts) == first_timestamp_len for ts in df['timestamp']), "All timestamp series must have same length"
    assert first_target_len == first_timestamp_len, "Timestamps and targets must have same length"
    
    # Convert targets and timestamps to lists
    targets = [df.loc[i, 'target'] for i in range(len(df))]
    timestamps = [df.loc[i, 'timestamp'] for i in range(len(df))]
    
    # Create dataset entries using actual timestamps
    dataset = [{"start": ts[0], "target": target} for ts, target in zip(timestamps, targets)]
    
    # Write to arrow file
    ArrowWriter(compression="lz4").write_to_file(dataset, path=output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Huggingface dataset to GluonTS arrow format')
    parser.add_argument('--input_path', type=str, help='Path to input parquet file')
    parser.add_argument('--output_path', type=str, help="Path to output arrow file. "
                        "If not specified, will use input path with .arrow extension")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = args.input_path.replace('.parquet', '.arrow')
        
    huggingface_to_arrow(args.input_path, args.output_path)

    


