import csv
import sys
import copy
import json
import os.path
import logging
import argparse

from pathlib import Path
from datetime import datetime
from inference import GenTools

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_output_file_path(output_path):
    # Check if the provided path is a directory
    if os.path.isdir(output_path):
        # If it's a directory, generate a filename
        file_path = os.path.join(output_path, f"llm_ann-{timestamp}.csv")
    else:
        # Ensure the parent directory exists and is writable
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            raise ValueError(f"Directory '{output_dir}' does not exist.")
        if not os.path.isdir(output_dir):
            raise ValueError(f"'{output_dir}' is not a directory.")
        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Directory '{output_dir}' is not writable.")

        # Ensure the file has a .csv extension
        if not output_path.lower().endswith('.csv'):
            raise ValueError("The output file must have a '.csv' extension.")
        file_path = output_path

    return file_path


if __name__ == "__main__":
    gen_tools = GenTools()

    parser = argparse.ArgumentParser(description="Annotating text using LLMs")
    parser.add_argument("--input-path", type=str, help="Path to the input CSV file. Example: './data/input.csv'",
                        required=True)
    parser.add_argument("--output-path", type=str,
                        help="Full path or directory for the output CSV file. "
                             "If a directory is provided, a filename will be automatically generated.",
                        required=True)
    parser.add_argument("--text-columns", nargs='+',
                        help="List of text columns to replace placeholders in the prompt template, in the specified order.",
                        required=True)
    parser.add_argument("--prompt-template", type=str,
                        help="Path to the prompt template file. Example: './templates/prompt.txt'", required=True)
    parser.add_argument('--annotator-models', nargs='+',
                        help='List of LLMs for annotation. supported models: {}'.format(
                            str(gen_tools.endpoint_manager.list_models())), required=True)
    parser.add_argument('--log-steps', type=int, help='logging steps', default=100)
    args = parser.parse_args()

    # -------------------
    # logging the command
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = os.path.join(logs_dir, f'llm_ann_{timestamp}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    full_command = ' '.join(sys.argv[:1])
    for key, value in vars(args).items():
        full_command += f" --{key} {value}"
    logging.info(f"Command: {full_command}")
    # -------------------

    if not os.path.exists(args.input_path):
        raise Exception("input file path does not exist")
    if not os.path.exists(args.prompt_template):
        raise Exception("prompt template file does not exist")

    # check if all user models are in the endpoint manager's list
    all_models_valid = all(model in gen_tools.endpoint_manager.list_models() for model in args.annotator_models)
    if not all_models_valid:
        raise Exception("error: one or more provided models are invalid and not supported.")

    # read prompt template
    with open(args.prompt_template, 'r') as file:
        prompt_template = file.read()
        logging.info(f"Prompt template: {prompt_template}")

    output_file_path = get_output_file_path(args.output_path)

    if args.input_path == output_file_path:
        raise Exception("Annotations cannot be saved into input file")

    with open(args.input_path, 'r') as input_path:
        csv_reader = csv.DictReader(input_path)

        # check if the text column exists
        if not all(col in csv_reader.fieldnames for col in args.text_columns):
            raise Exception("some text_columns don't exist in input file.")

        header = list(copy.deepcopy(csv_reader.fieldnames))
        annotator_models = {model: f'pred_{model}' for model in args.annotator_models}
        header.extend(annotator_models.values())

        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=header)
            writer.writeheader()
            for row_index, row in enumerate(csv_reader, start=1):
                for model_name, model_pred_col in annotator_models.items():
                    try:
                        # replacing prompt placeholders with row values
                        arguments = [row[col].strip() for col in args.text_columns]
                        prompt = gen_tools.build_prompt(prompt_template, arguments)
                        row[model_pred_col] = gen_tools.execute_prompt(prompt, model_name=model_name)
                    except Exception as e:
                        log_dict = {'prompt': prompt}
                        json_string = json.dumps(log_dict, ensure_ascii=False)
                        logging.error(f"Exception: {json_string}")
                        print('error in generating llm response for row index {}.\n\ndetail: {}'.format(row_index,
                                                                                                        str(e)))
                    try:
                        writer.writerow(row)
                        output_file.flush()
                    except Exception as e:
                        print(f"Failed to write row: {row}. Error: {e}")
                if row_index == 1 or row_index % args.log_steps == 0:
                    print('annotated {} records.'.format(row_index))
