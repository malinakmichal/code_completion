from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sacrebleu import corpus_chrf
from rouge_score import rouge_scorer
import sys
import numpy as np


def fill_in_code(row):
    """
    Generates code to complete a specific code segment based on prefix and suffix.

    Parameters:
    row (pd.Series): A row from the input DataFrame containing 'prefix' and 'suffix' columns.
    
    Returns:
    string: Generated code to complete the segment.
    """
    checkpoint = "bigcode/tiny_starcoder_py"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    input = f"<fim_prefix>{row['prefix']}\n<fim_suffix>{row['suffix']}\n<fim_middle>"

    input_tokenized = tokenizer(input, return_tensors="pt", return_attention_mask=True)
    outputs = model.generate(input_tokenized['input_ids'], max_new_tokens=20, attention_mask=input_tokenized["attention_mask"], pad_token_id=tokenizer.eos_token_id)
    generated_code = tokenizer.decode(outputs[0]).split("<fim_middle>")[1].split("\n")[0]
    return generated_code


def check_syntax(code):
    """
    Checks if the input code is syntactically correct and can be compiled.

    Parameters:
    code (str): The code string to check for syntax validity.
    
    Returns:
    bool: True if the code is syntactically correct, False otherwise.
    """
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return False
    return True


def evaluate(row):
    """
    Evaluates data for whole lines code completion using various metrics: CHRF, exact match, correct syntax, and ROUGE score.

    Parameters:
    row (pd.Series): Row containing the actual and generated code to be evaluated.
    
    Returns:
    pd.Series: A Series containing calculated metrics: 'chrf', 'exact_match', 'correct_syntax', and 'rouge'.
    """
    if row['result'].lower() == "nan" and row['label'].lower() == "nan":
        return pd.Series({"CHRF": 0, "exact_match": "nan", "correct_syntax": "nan", "rougeL": 0})
    chrf_score = corpus_chrf([row['result']], [[row['label']]]).score

    exact_match = str(row['result'] == row['label'])

    correct_syntax = str(check_syntax(row['result'].lstrip("\ ")))

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(row['label'], row['result'])

    if isinstance(rouge_score, dict):
        rouge_score = rouge_score['rougeL'].fmeasure
    else:
        rouge_score = 0

    return pd.Series({"CHRF": chrf_score, "exact_match": exact_match, "correct_syntax": correct_syntax, "rougeL": rouge_score})


def evaluate_in_the_middle(row):
    """
    Evaluates data for partial line (cursor placed in the middle of a line) code completion using various metrics: CHRF, exact match, correct syntax, and ROUGE score.

    Parameters:
    row (pd.Series): Row containing the actual and generated code to be evaluated.
    
    Returns:
    pd.Series: A Series containing calculated metrics: 'chrf', 'exact_match', 'correct_syntax', and 'rouge'.
    """
    if row['result'].lower() == "nan" and row['label'].lower() == "nan":
        return pd.Series({"CHRF": 0, "exact_match": "nan", "correct_syntax": "nan", "rougeL": 0})
    chrf_score = corpus_chrf([row['result']], [[row['label_prefix'] + row['label']]]).score

    exact_match = str(row['result'] == row['label_prefix'] + row['label'] or row['label_prefix'] + row['result'] == row['label_prefix'] + row['label'])

    correct_syntax = str(check_syntax(row['result'].lstrip("\ ")) or check_syntax(row['label_prefix'] + (row['result'].lstrip("\ "))))

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(row['label_prefix'] + row['label'], row['result'])

    if isinstance(rouge_score, dict):
        rouge_score = rouge_score['rougeL'].fmeasure
    else:
        rouge_score = 0

    return pd.Series({"CHRF": chrf_score, "exact_match": exact_match, "correct_syntax": correct_syntax, "rougeL": rouge_score})


def print_results(data):
    """
    Prints various metrics of the input dataset: CHRF, exact match, correct syntax, and ROUGE score.

    Parameters:
    data (pd.DatFrame): Dataset containing various metrics: CHRF, exact match, correct syntax, and ROUGE score.
    
    Returns:
    None
    """
    data['correct_syntax'] = data['correct_syntax'].astype(str)
    data['exact_match'] = data['exact_match'].astype(str)

    average_CHRF = data['CHRF'].mean()
    print(f"CHRF:\n{round(average_CHRF, 2)} %\n")

    syntax = data['correct_syntax'].value_counts()
    print(f"Correct syntax:\n{syntax.to_string(index=True, header=False)}")
    print(f"{round(100*((syntax.get('True', 0) + 1)/(syntax.get('True', 0) + syntax.get('False', 0) + 1)), 2)} %\n")

    match = data['exact_match'].value_counts()
    print(f"Exact match:\n{match.to_string(index=True, header=False)}")
    print(f"{round(100*((match.get('True', 0) + 1)/(match.get('True', 0) + match.get('False', 0) + 1)), 2)} %\n")

    average_rougeL = data['rougeL'].mean()
    print(f"rougeL:\n{round(100*average_rougeL, 2)} %\n")


def proccess(csv_path, cursor_in_the_middle):
    """
    Generates the code completions, evaluates the dataset, and outputs the metrics.

    Parameters:
    csv_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    None
    """
    data = pd.read_csv(csv_path)
    data = data.replace(np.nan, 'NaN')
    data['result'] = data.apply(fill_in_code, axis=1)

    data['label'] = data['label'].astype(str)
    data['result'] = data['result'].astype(str)

    data['label'] = data['label'].fillna("NaN")
    data['result'] = data['result'].fillna("NaN")
    data = data.replace('', 'NaN')
    data = data.replace('nan', 'NaN')

    if cursor_in_the_middle:
        data[['CHRF', 'exact_match', 'correct_syntax', 'rougeL']] = data.apply(evaluate_in_the_middle, axis=1)
    else:
        data[['CHRF', 'exact_match', 'correct_syntax', 'rougeL']] = data.apply(evaluate, axis=1)

    data.to_csv(csv_path, index=False)

    print_results(data)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] in ["True", "False"]:
        proccess(sys.argv[1], True if sys.argv[2] == "True" else False)
    else:
        print("Error: csv path not given")