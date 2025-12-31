import pandas as pd

def verify_ptbxl_snomed_parquet(file_path="data/Processed/ptbxl_with_snomed.parquet"):
    """
    Verifies the ptbxl_with_snomed.parquet file.

    Args:
        file_path (str): The path to the Parquet file.
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded {file_path}")

        # Filter for SNOMED columns
        snomed_cols = [col for col in df.columns if col.startswith("SNOMED_")]

        # Print the number of unique SNOMED columns
        num_snomed_cols = len(snomed_cols)
        print(f"Number of unique SNOMED columns: {num_snomed_cols}")

        # Show a sample of the raw SNOMED ID column names
        print("Sample of SNOMED column names:")
        for col in snomed_cols[:5]:
            print(f"- {col}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    verify_ptbxl_snomed_parquet()
