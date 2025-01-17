import pandas as pd
import numpy as np
import sys

def topsis(input_file, weights, impacts, result_file):
    try:
        # Check file extension and read input file
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            data = pd.read_excel(input_file)
        else:
            raise ValueError("Input file must be a .csv or .xlsx file")

        # Validate input data
        if data.shape[1] < 3:
            raise ValueError("Input file must contain at least three columns")
        
        # Extract and validate weights and impacts
        weights = [float(w) for w in weights.split(',')]
        impacts = impacts.split(',')

        if len(weights) != len(impacts) or len(weights) != (data.shape[1] - 1):
            raise ValueError("Number of weights and impacts must match the number of criteria columns")

        if not all(impact in ['+', '-'] for impact in impacts):
            raise ValueError("Impacts must be either '+' or '-' only")

        # Normalize the decision matrix
        matrix = data.iloc[:, 1:].values
        norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))

        # Calculate weighted normalized decision matrix
        weighted_matrix = norm_matrix * weights

        # Determine ideal best and worst values
        ideal_best = np.where(np.array(impacts) == '+', weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
        ideal_worst = np.where(np.array(impacts) == '+', weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

        # Calculate distances from ideal best and worst
        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

        # Calculate TOPSIS score and rank
        scores = dist_worst / (dist_best + dist_worst)
        data['Topsis Score'] = scores
        data['Rank'] = scores.argsort()[::-1] + 1

        # Write results to output file
        if result_file.endswith('.csv'):
            data.to_csv(result_file, index=False)
        elif result_file.endswith('.xlsx'):
            data.to_excel(result_file, index=False)
        else:
            raise ValueError("Result file must be a .csv or .xlsx file")

    except FileNotFoundError:
        print("Error: Input file not found")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        _, input_file, weights, impacts, result_file = sys.argv
        topsis(input_file, weights, impacts, result_file)
