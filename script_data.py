import pandas as pd
import numpy as np
import sys


def process_data(csv_path):
    """
    Splits data into prefix, suffix, and label from the dataset.

    Parameters:
    csv_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    tuple: A tuple containing the prefix, suffix, and label.
    """
    with open(csv_path, 'r') as file:
        file_contents = file.read()

    lines = file_contents.split("\n")

    if len(lines) >= 1:
        random_line = np.random.randint(1, len(lines))
    else:
        return "", "", file_contents

    label = lines[random_line - 1]
    prefix = "\n".join(lines[:random_line - 1])
    suffix = "\n".join(lines[random_line:])
    return prefix, suffix, label


def create_data(csv_path, cursor_in_the_middle):
    """
    Creates a dataset for code completion from personal project files.

    Parameters:
    path (str): Path to the CSV file containing the dataset.
    cursor_in_the_middle (bool): True if we want to generate code completion for partial line completion,
                                  and False for whole line completion.
    
    Returns:
    None
    """
    files = ["files/image_NN.py", "files/models.py", "files/filtering.py", "files/minimaxAI.py"]

    if cursor_in_the_middle:
        data = pd.DataFrame(columns=["prefix", "suffix", "label", "label_prefix"])
        for file in files:
            for _ in range(8):
                processed = process_data(file, cursor_in_the_middle)
                if len(processed[2]) < 3:
                    random_char = 0
                else:
                    random_char = np.random.randint(0, int(len(processed[2])*0.8))
                
                label_prefix = processed[2][:random_char]
                label = processed[2][random_char:]
                data.loc[len(data)] = {"prefix": processed[0] + "\n" + label_prefix, "suffix": processed[1], "label": label, "label_prefix": label_prefix}

    else:
        data = pd.DataFrame(columns=["prefix", "suffix", "label"])
        for file in files:
            for _ in range(8):
                processed = process_data(file, cursor_in_the_middle)
                data.loc[len(data)] = {"prefix": processed[0], "suffix": processed[1], "label": processed[2]}
    data.to_csv(csv_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] in ["True", "False"]:
        create_data(sys.argv[1], True if sys.argv[2] == "True" else False)
    else:
        print("Error: csv path not given")

