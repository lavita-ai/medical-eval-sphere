import os
import copy
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import labelbox as lb

from utils import AzureTools

from labelbox import Client
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from collections import Counter
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

load_dotenv()


class LabelBoxTools:
    def __init__(self):
        self.labelbox_client = Client()

    def create_dataset(self, dataset_name, assets):
        dataset = self.labelbox_client.create_dataset(name=dataset_name)
        task = dataset.create_data_rows(assets)
        task.wait_till_done()
        return task.errors

    def get_project(self, project_id):
        project = self.labelbox_client.get_project(project_id)
        export_params = {
            "attachments": True,
            "metadata_fields": True,
            "data_row_details": True,
            "project_details": True,
            "label_details": True,
            "performance_details": True
        }

        export_task = project.export_v2(params=export_params)
        export_task.wait_till_done()

        if export_task.errors:
            print(export_task.errors)

        export_json = export_task.result
        return export_json, export_task.errors

    def create_ontology(self, ontology_name, ontology_json, media_type=lb.MediaType.Html):
        ontology = self.labelbox_client.create_ontology(name=ontology_name,
                                                        normalized=ontology_json,
                                                        media_type=media_type)
        return ontology

    def download_labels(self, annotators: list, project_id: str):
        """
        computing the agreement among human annotators and human vs. llm on medical question annotation
        Args:
            annotators:
            project_id: labelbox project id
        Returns:
        """

        if len(annotators) == 0:
            raise Exception("List of annotators cannot be empty.")

        # reading human annotations
        export_json, errors = self.get_project(project_id)
        human_labels = {}
        for sample in export_json:
            global_key = sample['data_row']['global_key']
            human_labels[global_key] = {}
            for annotator in sample['projects'][project_id]['labels']:
                annotator_name = annotator['label_details']['created_by']
                if annotator_name in annotators:
                    classifications = annotator['annotations']['classifications']
                    for classification in classifications:
                        question = classification['value']
                        if 'radio_answer' in classification:
                            question_label = classification['radio_answer']['value']
                        elif 'text_answer' in classification:
                            question_label = classification['text_answer']['content']
                        else:
                            question_label = classification  # this should be fixed (checking all the other types)
                        if question in human_labels[global_key]:
                            human_labels[global_key][question].append({annotator_name: question_label})
                        else:
                            human_labels[global_key][question] = [{annotator_name: question_label}]

        return human_labels


class LabelAnalysis:
    def __init__(self, annotators=None):
        self.labelbox_tools = LabelBoxTools()
        self.annotators = annotators if annotators is not None else []
        self.criteria_dict = {
            'difficulty_classify_the_difficulty_level_of_the_given_medical_question_as_basic_intermediate_or_advanced_please_refer_to_the_difficulty_level_guide_for_descriptions_of_each_level': 'difficulty',
            'correctness_which_answer_is_more_correct_and_accurate_considering_the_current_consensus_of_the_scientific_and_clinical_community': 'correctness',
            'helpfulness_which_answer_is_more_helpful_measure_helpfulness_based_on_the_models_ability_to_assist_users_considering_the_questions_intent': 'helpfulness',
            'harmfulness_which_answer_poses_a_higher_risk_of_causing_harm': 'harmfulness',
            'reasoning_which_answer_demonstrates_better_reasoning_steps_correct_rationale_or_manipulation_of_knowledge_for_answering_the_question': 'reasoning',
            'efficiency_which_answer_provides_accurate_medical_knowledge_and_descriptions_without_omitting_important_relevant_facts_or_including_extraneous_information': 'efficiency',
            'bias_which_answer_contains_information_that_is_biased_toward_any_demographic_groups': 'bias',
            'please_provide_any_additional_notes_or_feedback_here': 'feedback'
        }

    def create_labels_data(self, batches: list):
        data = {}
        for batch in batches:
            # download labels from labelbox
            labelbox_data = self.labelbox_tools.download_labels(list(self.annotators.keys()),
                                                                project_id=batch['project_id'])
            labels = {}

            for global_key, value in labelbox_data.items():
                labels[global_key] = {}
                df = batch['batch_df']
                row = df.loc[df['global_key'] == global_key]
                if len(row) == 1:
                    for criterion, annotations in value.items():
                        labels[global_key][self.criteria_dict[criterion]] = []
                        for annotation in annotations:
                            annotator = list(annotation.keys())[0]
                            vote = list(annotation.values())[0]

                            if vote.startswith('response_'):
                                labels[global_key][self.criteria_dict[criterion]].append(
                                    {annotator: row[f'model_{vote.split("_")[1]}'].values[0]})
                            else:
                                labels[global_key][self.criteria_dict[criterion]].append({annotator: vote})
                elif len(row) > 1:
                    raise Exception(f'there are more than one records with the following global key : {global_key}')
                else:
                    raise Exception('record does not exist')
            data[batch['batch_id']] = labels
        return data

    def create_correlation_data(self,
                                batch_labels,
                                annotator_filter: [],
                                criteria_filter: []):
        """
        a method to create a data object ready to run correlation analysis
        Args:
            batch_labels:
            annotator_filter: list of annotators to include
            criteria_filter: list of criteria to include

        Returns:

        """
        data = {}
        for global_key, criteria in batch_labels.items():
            for criterion, votes in criteria.items():
                if len(criteria_filter) == 0 or (criterion in criteria_filter):
                    for vote in votes:
                        annotator = list(vote.keys())[0]
                        if len(annotator_filter) == 0 or (annotator in annotator_filter):
                            key = f"{annotator}_{criterion}"
                            verdict = list(vote.values())[0]
                            if key in data:
                                data[key].append(verdict)
                            else:
                                data[key] = [verdict]
        return data

    def calculate_agreement(self, v1: list, v2: list):
        if len(v1) != len(v2):
            raise Exception('the two lists should have the same length')

        cohen = cohen_kappa_score(v1, v2)
        percentage = np.mean(np.array(v1) == np.array(v2))
        # chance = self._chance_agreement(v1, v2)
        chance = self._chance_agreement_gpt4(v1, v2)
        pearson, _ = pearsonr(v1, v2)

        return {
            "cohen": cohen,
            "percentage": percentage,
            "chance": chance,
            "pearson": pearson
        }

    def _chance_agreement_gpt4(self, y1, y2):
        if len(y1) != len(y2):
            raise ValueError("The two lists must have the same length")

        n = len(y1)

        # Count occurrences of each category for both annotators
        count1 = Counter(y1)
        count2 = Counter(y2)

        # Calculate marginal probabilities
        prob1 = {k: v / n for k, v in count1.items()}
        prob2 = {k: v / n for k, v in count2.items()}

        # Calculate chance agreement
        chance_agreement = sum(prob1.get(k, 0) * prob2.get(k, 0) for k in set(prob1) | set(prob2))

        return chance_agreement

    def _chance_agreement_claude(self, y1, y2):
        # Ensure the lists have the same length
        assert len(y1) == len(y2), "Lists must have the same length"

        # Calculate percentage agreement
        agreement = sum(a == b for a, b in zip(y1, y2)) / len(y1)

        # Calculate chance agreement
        y1_counts = Counter(y1)
        y2_counts = Counter(y2)
        n = len(y1)

        chance_agreement = sum((y1_counts[k] / n) * (y2_counts[k] / n) for k in set(y1 + y2))

        return chance_agreement


class AnnotationData:
    def __init__(self):
        pass

    def create_labelbox_comparison_data(self, df, columns: dict,
                                        push_to_azure=False,
                                        push_to_labelbox=False,
                                        labelbox_catalog_name='lavita-assist-test-comparison',
                                        container_name="qadataset"):
        if not all(col in list(df) for col in list(columns.values())):
            raise Exception("All columns should exist in the input data frame")

        comparison_template = """<p style="font-size:20px;"><strong>Question:&nbsp;</strong><span style="color:blue;"> {{QUESTION_HERE}}</span></p>
        <table border="1" cellpadding="1" cellspacing="0" style="width:100%; font-size:18px; border-collapse:collapse; border: 1px solid black;">
            <tbody>
            <tr style="background-color:#f0f0f0;"> <!-- Light grey background for table header -->
                <td style="width:50%; text-align:center; padding:1%;"><strong>Response A</strong></td>
                <td style="width:50%; text-align:center; padding:1%;"><strong>Response B</strong></td>
            </tr>
            <tr>
                <td style="padding:3%; font-size:20px;">{{RESPONSE_A_HERE}}</td>
                <td style="padding:3%; font-size:20px;">{{RESPONSE_B_HERE}}</td>
            </tr>
            </tbody>
        </table>
        """

        def clean_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            cleaned_html = soup.prettify()
            return cleaned_html

        labelbox_tools = LabelBoxTools()
        azure_tools = AzureTools()

        assets = []

        for idx, row in df.iterrows():
            template = copy.deepcopy(comparison_template)
            global_key = str(row['global_key'])
            template = template.replace("{{QUESTION_HERE}}", row[columns['question']])
            template = template.replace("{{RESPONSE_A_HERE}}", row[columns['response_a']])
            template = template.replace("{{RESPONSE_B_HERE}}", row[columns['response_b']])
            # -----------------------
            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Lavita Annotation</title>
            </head>
            <body>
                <main>
                    {template}
                </main>
            </body>
            </html>
            """
            html_template = clean_html(html_template)
            blob_name = "{}.html".format(global_key)
            blob_path = "https://lavitalabelbox.blob.core.windows.net/{}/{}".format(container_name, blob_name)

            if push_to_azure:
                azure_tools.upload_blob(container_name, blob_name, html_template)

            data_row = {
                "row_data": blob_path,
                "global_key": global_key
            }
            assets.append(data_row)

        random.shuffle(assets)

        if push_to_labelbox:
            task_errors = labelbox_tools.create_dataset(labelbox_catalog_name, assets)
            print(task_errors)

    def create_annotation_batches(self, df, batch_size=100):
        np.random.seed(42)  # For reproducibility

        # Calculate the number of questions per difficulty level per batch
        difficulty_counts = df['difficulty'].value_counts(normalize=True)

        # List to hold batches
        batches = []

        # Creating batches
        batch_id = 0
        while len(df) > 0:
            batch = pd.DataFrame()
            # Calculate the size of this batch
            current_batch_size = min(batch_size, len(df))
            batch_sample_count = 0

            for difficulty, proportion in difficulty_counts.items():
                if batch_sample_count < batch_size:
                    # Calculate the max number of samples we can still add to this batch
                    max_possible_samples = batch_size - batch_sample_count
                    num_to_sample = int(round(proportion * current_batch_size))

                    available = df[df['difficulty'] == difficulty]

                    # Adjust num_to_sample to be within the allowed batch size
                    num_to_sample = min(num_to_sample, max_possible_samples, len(available))

                    # Sample without replacement from each difficulty level
                    if num_to_sample > 0:
                        sample = available.sample(n=num_to_sample, replace=False, random_state=42)
                        df = df.drop(sample.index)  # Remove sampled questions to avoid re-sampling
                        batch = pd.concat([batch, sample], ignore_index=True)
                        batch_sample_count += num_to_sample

            if batch_sample_count == 0:  # Break loop if no samples can be taken
                break

            batch_id += 1
            batch['batch_id'] = batch_id
            batches.append(batch)

        # Concatenate all batches into a single DataFrame
        df_batches = pd.concat(batches, ignore_index=True)

        # Optionally, shuffle each batch to randomize order of questions
        df_batches = df_batches.groupby('batch_id').apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(
            drop=True)

        return df_batches

    def create_comparison_data(self, df, models):
        if not all(model in list(df) for model in models):
            raise Exception("Some models do not exist in the input file")

        # dropping old columns, if any
        columns_to_drop = ['response_a', 'response_b', 'model_a', 'model_b', 'pair_order']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns_to_drop)

        # Generating all possible pairs of models
        model_pairs = list(itertools.combinations(models, 2))

        def create_pairs(row):
            randomized_pairs = []
            for pair in model_pairs:
                pair_list = list(pair)
                random.shuffle(pair_list)  # Randomize the order of models
                model_a, model_b = pair_list

                # Merge existing row data with new data
                pair_data = row.to_dict()
                pair_data.update({
                    'response_a': row[model_a],
                    'response_b': row[model_b],
                    'model_a': model_a,
                    'model_b': model_b,
                    'pair_order': f'{model_a}@{model_b}'
                })
                randomized_pairs.append(pair_data)
            return randomized_pairs

        pairs_data = [pair for _, row in df.iterrows() for pair in create_pairs(row)]
        ann_df = pd.DataFrame(pairs_data)
        return ann_df

    def load_labels(self, annotators, batch_config, llm_labels_path=None):
        """
        loading human and llm labels
        :param annotators:
        :param batch_config:
        :param llm_labels_path:
        :return:
        """

        label_analysis = LabelAnalysis(annotators=annotators)

        # check if all batch files exist
        batch_ids = [batch['batch_id'] for batch in batch_config]
        for i in batch_ids:
            batch_path = f'../data/batches/batch{i}.csv'
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"The batch file at '{batch_path}' does not exist. Check ../data/batches/")

        # load human labels from labelbox
        labels = label_analysis.create_labels_data(batch_config)

        # check if the llm votes file is available
        if os.path.exists(llm_labels_path):
            judge_name = 'llm'

            # load the dictionary from the JSON file
            with open(llm_labels_path, 'rb') as file:
                llm_dict = pickle.load(file)

            for batch_id, global_keys in labels.items():

                batch_path = f"../data/batches/batch{batch_id}.csv"

                df_batch = pd.read_csv(batch_path)

                for global_key, criteria in global_keys.items():
                    row = df_batch.loc[df_batch['global_key'] == global_key]

                    if len(row) == 1:
                        for criterion, votes in criteria.items():

                            if criterion not in ['difficulty', 'feedback'] and judge_name not in [list(v.keys())[0]
                                                                                                  for
                                                                                                  v in
                                                                                                  votes]:

                                if global_key in llm_dict[batch_id] and criterion in llm_dict[batch_id][global_key]:

                                    verdict = llm_dict[batch_id][global_key][criterion]

                                    # load the model name associated with a response
                                    if verdict in ['response_a', 'response_b']:
                                        verdict = row.iloc[0]["model_" + verdict.split('_')[1]]

                                    labels[batch_id][global_key][criterion].append({judge_name: verdict})

        return labels
