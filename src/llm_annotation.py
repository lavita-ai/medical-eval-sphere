import csv
import sys
import copy
import json
import os.path
import logging
import argparse

from utils import LangTools
from datetime import datetime
from inference import GenTools

if __name__ == "__main__":
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    gen_tools = GenTools()

    parser = argparse.ArgumentParser(description="annotating text using LLMs")
    parser.add_argument("--input-file", type=str, help="path to raw input csv file")
    parser.add_argument("--output-dir", type=str, help="output csv file directory")
    parser.add_argument("--text-columns", nargs='+',
                        help="list of text columns to replace placeholders in prompt template in that order")
    parser.add_argument("--prompt-template", type=str, help="path to prompt template")
    parser.add_argument('--annotator-models', nargs='+',
                        help='list of LLMs for annotation. supported models: {}'.format(
                            str(gen_tools.endpoint_manager.list_models())))
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

    if not os.path.exists(args.input_file):
        raise Exception("input file path does not exist")
    if not os.path.exists(args.output_dir):
        raise Exception("output directory path does not exist")
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

    output_path = '{}/llm_ann-{}.csv'.format(args.output_dir, timestamp)

    with open(args.input_file, 'r') as input_file:
        csv_reader = csv.DictReader(input_file)

        # check if the text column exists
        if not all(col in csv_reader.fieldnames for col in args.text_columns):
            raise Exception("some text_columns don't exist in input file.")

        header = list(copy.deepcopy(csv_reader.fieldnames))
        annotator_models = {model: f'pred_{model}' for model in args.annotator_models}
        header.extend(annotator_models.values())

        with open(output_path, 'a', newline='') as output_file:
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
                    writer.writerow(row)
                    output_file.flush()
                if row_index == 1 or row_index % args.log_steps == 0:
                    print('annotated {} records.'.format(row_index))
