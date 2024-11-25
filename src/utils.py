import re
import csv
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContentSettings
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

parent_path = Path(os.path.abspath(__file__)).parents[0]
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(root_path))

azure_storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')


class AzureTools:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)

    def get_container_client(self, container_name):
        try:
            container_client = self.blob_service_client.get_container_client(container=container_name)
            if not container_client.exists():
                container_client.create_container()
            return container_client
        except Exception as e:
            print("Error in getting Azure container client: {}".format(repr(e)))

    def upload_blob(self, container_name, blob_name, blob_content):
        """
        uploading a blob to an existing container
        :param container_client:
        :param local_file_path:
        :return:
        """
        try:
            container_client = self.get_container_client(container_name=container_name)
            blob_client = container_client.get_blob_client(blob_name)
            content_type = "text/html"
            blob_client.upload_blob(blob_content, blob_type="BlockBlob", overwrite=True,
                                    content_settings=ContentSettings(content_type=content_type))

        except Exception as e:
            print(" {}".format(repr(e)))

    def upload_file(self, container_name: str, local_file_path: str):
        try:
            blob_name = os.path.basename(local_file_path)
            container_client = self.blob_service_client.get_container_client(container=container_name)
            file_size = os.stat(local_file_path).st_size
            with tqdm.wrapattr(open(file=local_file_path, mode="rb"), "read", total=file_size) as data:
                container_client.upload_blob(name=blob_name, data=data, content_type="text/html", overwrite=True)
        except Exception as e:
            print(" {}".format(repr(e)))

    def delete_blobs(self, container_name):
        # Get a reference to the container
        container_client = self.blob_service_client.get_container_client(container_name)
        # List all blobs in the container
        blob_list = container_client.walk_blobs()
        # Delete each blob in the container
        for blob in blob_list:
            container_client.delete_blob(blob)

    def list_container_blobs(self, container_name):
        """
        listing all blobs in a specific container
        :param container_name:
        :return: list of blob names
        """
        try:
            container_client = self.blob_service_client.get_container_client(container=container_name)
            blobs = container_client.list_blobs()
            blob_list = [blob.name for blob in blobs]
            return blob_list
        except Exception as e:
            print("Error in listing blobs in container: {}".format(repr(e)))


class LangTools:
    def __init__(self):
        from lingua import LanguageDetectorBuilder
        self.detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def detect_lang(self, doc):
        language = self.detector.detect_language_of(doc)
        if language is not None:
            return language.iso_code_639_1.name.lower()
        return "not detected"


class ClusteringTools:
    def __init__(self):
        self.clustering_methods = ["dbscan", "hierarchical"]

    @staticmethod
    def is_valid_json_string(x):
        """
        Check if x is a valid JSON string that can be parsed into a Python object.
        """
        if isinstance(x, str):
            try:
                json.loads(x)
                return True
            except json.JSONDecodeError:
                return False
        return False

    def dataset_similarity(self, embeddings_a, embeddings_b, method="mean"):
        """Calculate the general similarity between two datasets using cosine similarity."""
        if isinstance(embeddings_a, (list, pd.Series)) or self.is_valid_json_string(embeddings_a):
            embeddings_a = np.array(embeddings_a)
        if isinstance(embeddings_b, (list, pd.Series)) or self.is_valid_json_string(embeddings_b):
            embeddings_b = np.array(embeddings_b)

        if not isinstance(embeddings_a, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                f"Expected embeddings_a to be numpy.ndarray or pandas.DataFrame, but got {type(embeddings_a)}")
        if not isinstance(embeddings_b, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                f"Expected embeddings_b to be numpy.ndarray or pandas.DataFrame, but got {type(embeddings_b)}")

        similarities = []
        for vec_a in embeddings_a:
            vec_a_similarities = []
            for vec_b in embeddings_b:
                similarity = cosine_similarity(vec_a, vec_b)
                vec_a_similarities.append(similarity)
            if method == "mean":
                similarities.append(np.mean(vec_a_similarities))
            elif method == "max":
                similarities.append(max(vec_a_similarities))

        # Average the similarities to get a general measure of similarity between the datasets
        average_similarity = np.mean(similarities)
        return average_similarity, similarities

    @staticmethod
    def visualize_clusters(df):
        if not all(col in list(df) for col in ['embedding', 'cluster']):
            raise Exception('Input df should have the following columns: embedding, cluster')

        embeddings = np.vstack(df['embedding'])

        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plotting the 2D embeddings
        plt.figure(figsize=(10, 8))

        for cluster in np.unique(df['cluster']):
            idx = df['cluster'] == cluster
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Cluster {cluster}')

        plt.title('2D visualization of embeddings by cluster')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    def find_clusters(self, df, clustering_method='dbscan', similarity_threshold=0.8):
        if clustering_method not in self.clustering_methods:
            raise Exception("Clustering method is not supported.")
        if 'embedding' not in list(df):
            raise Exception("Input df should include an \"embedding\" column")
        else:
            try:
                if isinstance(df.iloc[0]['embedding'], str):
                    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)).reshape(1, -1))
            except:
                raise Exception(
                    "The embedding column in df is stored as string. There is an error when converting the string.")

        embeddings = np.vstack(df['embedding']).copy()
        threshold_distance = 1 - similarity_threshold

        if clustering_method == "hierarchical":
            distances = squareform(pdist(embeddings, metric='cosine'))
            # hierarchical clustering
            z = linkage(distances, 'ward')
            clusters = fcluster(z, threshold_distance, criterion='distance')
        elif clustering_method == "dbscan":
            distance_matrix = cosine_distances(embeddings)
            dbscan = DBSCAN(eps=threshold_distance, min_samples=1, metric='precomputed')
            clusters = dbscan.fit_predict(distance_matrix)

        df['cluster'] = clusters

        return df

    @staticmethod
    def print_clusters(df, text_column='input_text'):
        if 'cluster' not in list(df):
            raise Exception("Input df should include an \"cluster\" column")

        unique_clusters = df['cluster'].unique()

        for cluster in unique_clusters:
            if cluster != -1:
                cluster_df = df[df['cluster'] == cluster]
                for text in cluster_df[text_column]:
                    print(text + '\n\n')
            print("================================")

    @staticmethod
    def select_cluster_representatives(df, method='max'):
        if 'cluster' not in list(df):
            raise Exception("Input df should include an \"cluster\" column")
        if 'embedding' not in list(df):
            raise Exception("Input df should include an \"embedding\" column")

        df_selected = pd.DataFrame()
        if method == 'max':
            df_selected = df.groupby('cluster', group_keys=False).apply(
                lambda g: g.loc[g['input_text'].str.len() == g['input_text'].str.len().max()]
            )
            df_selected = df_selected.drop_duplicates('cluster').reset_index(drop=True)
        elif method == 'medoid':
            for cluster in df['cluster'].unique():
                cluster_df = df[df['cluster'] == cluster].copy()

                # Calculate the pairwise distance matrix for the embeddings in the cluster
                distance_matrix = squareform(pdist(np.vstack(cluster_df['embedding']), metric='euclidean'))

                # Find the index of the medoid (the point with the minimum total distance to others)
                medoid_index = np.argmin(distance_matrix.sum(axis=0))
                representative = cluster_df.iloc[[medoid_index]]
                df_selected = pd.concat([df_selected, representative], axis=0, ignore_index=True)

            df_selected = df_selected.reset_index(drop=True)

        return df_selected


def get_faqs():
    faqs = [
        "Should I Go Gluten-Free?",
        "Is a Daily Glass of Wine Healthy?",
        "Are Short Workouts Worth It?",
        "Is Tap Water Safe to Drink?",
        "Sugar or High Fructose Corn Syrup?",
        "Does Cholesterol in Food Count?",
        "Do Vaccines Cause Autism Spectrum Disorder?",
        "Is Microwaved Food Unsafe?",
        "Do Cell Phones Cause Brain Cancer?",
        "Can I Be Fat and Healthy?",
        "Can I drink alcohol if I'm taking painkillers?",
        "Why must some medicines be taken on an empty stomach?",
        "Can I take my medicine abroad?",
        "How long is a prescription valid for?",
        "Does grapefruit affect my medicine?",
        "Can clothes and towels spread germs?",
        "How long is someone contagious after a viral infection?",
        "How long will I be infectious after starting antibiotics?",
        "How should I collect and store a stool sample?",
        "How should I collect and store a urine sample?",
        "Is it safe to use hair dye when I'm pregnant or breastfeeding?",
        "Is it safe to use fake tan during pregnancy?",
        "What if I'm pregnant and I haven't had chickenpox?",
        "Can HIV be passed to an unborn baby during pregnancy or through breastfeeding?",
        "When will my periods start again after pregnancy?",
        "What Causes High Blood Pressure?",
        "What Are Systolic and Diastolic Blood Pressure?",
        "What Is a Normal Blood Pressure?",
        "What Health Problems Are Associated With High Blood Pressure?",
        "How Do I Know If I Have High Blood Pressure?",
        "What Is the Treatment for High Blood Pressure?",
        "What Are the Side Effects of High Blood Pressure Drugs?",
        "What Type of Diet Should I Follow if I Have High Blood Pressure?",
        "When Should I Call My Doctor About High Blood Pressure?",
        "Are There Any Drugs that Cause High Blood Pressure?"
    ]

    faqs = [faq.lower().strip() for faq in faqs]

    return faqs


def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def clean_json_string(json_string):
    """Cleans escape characters and formats JSON string properly."""
    # Replace escaped newlines and tabs with actual newlines/tabs
    json_string = json_string.replace('\\n', '\n').replace('\\t', '\t')
    # Remove unnecessary escape characters before quotes
    json_string = re.sub(r'\\(?=["\'])', '', json_string)
    return json_string


def extract_json(text):
    """Fallback method to extract JSON using \n\n pattern."""
    # Locate the start of the 'text' field
    start = text.find("text='") + len("text='")
    if start == -1:
        return None

    # Extract everything starting from "text='"
    remaining_text = text[start:]

    # Split at the first occurrence of '\n\n'
    json_part = remaining_text.split("\\n\\n", 1)[0]  # First part before explanations

    # Remove surrounding single quotes (if any)
    if json_part.endswith("'"):
        json_part = json_part[:-1]

    try:
        # Clean and parse the JSON
        json_part = clean_json_string(json_part)
        return json.loads(json_part)
    except json.JSONDecodeError:
        return None


def jsonify(text):
    """Extract and convert JSON string from LLM response."""
    # Attempt to match using regex
    match = re.search(r"text='({.*})'", text)

    if match:
        json_string = match.group(1).strip()
        # Clean up the JSON string
        json_string = json_string.replace('\\n', '\n').replace('\\t', '').replace('\t', '')
        json_string = re.sub(r'\\(?=["\'])', '', json_string)

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON with regex: {e}")

    # Fallback to extract_json method
    json_data = extract_json(text)
    if json_data:
        return json_data

    # Raise exception if both methods fail
    raise ValueError("No valid JSON found in the input text")


def normalize_json_response(s):
    if not is_valid_json(s):
        # Remove any leading/trailing backticks and the optional 'json' keyword
        s = re.sub(r'^```json|```$', '', s.strip()).strip('```')

        # fixing missing ','
        s = re.sub(r'\"\s*\"', '", "', s)
        if not is_valid_json(s):
            def replace_quotes(match):
                # escape double quotes inside the string, excluding the ones at the boundaries
                return re.sub(r'(?<!\\)"', r'\"', match.group(0))

            pattern = r'(?<=": ")(.*?)(?="[,}])'
            s = re.sub(pattern, replace_quotes, s)
            last_brace_pos = s.rfind('}')
            if last_brace_pos != -1:
                # this is to get rid of the additional explanation some llms add after the json object
                # (usually not the case for gpt-4)
                s = s[:last_brace_pos + 1]

            if not is_valid_json(s):
                return ""
            else:
                return s
        else:
            return s
    else:
        return s


def count_lines_in_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            # Create a CSV reader object
            file_reader = csv.reader(file)
            # Count the lines using the CSV reader
            line_count = sum(1 for row in file_reader)
            return line_count
    except FileNotFoundError:
        # File doesn't exist; return 0
        return 0


def get_sample_size(n_population, p=0.5, e=0.05):
    """
    calculating sample size
    Args:
        n_population: population size
        p: population proportion (maximizes sample size)
        e: margin of error
    Returns:
    """
    # Constants
    z = norm.ppf(0.975)  # Z-score for 95% confidence level

    # Cochran's formula
    n_0 = (z ** 2 * p * (1 - p)) / e ** 2

    # Adjusting for finite population
    n = n_0 / (1 + (n_0 - 1) / n_population)

    return n, n_0


def r_pearson(x, y):
    if not (isinstance(x, list) and isinstance(y, list)):
        raise Exception('x and y should be lists')

    # Step 1: Calculate the means of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    # Step 2: Calculate the deviations from the mean for x and y
    dev_x = [i - mean_x for i in x]
    dev_y = [i - mean_y for i in y]

    # Step 3: Calculate the numerator of the Pearson correlation formula
    numerator = sum(dx * dy for dx, dy in zip(dev_x, dev_y))

    # Step 4: Calculate the denominator of the Pearson correlation formula
    sum_sq_dev_x = sum(dx ** 2 for dx in dev_x)
    sum_sq_dev_y = sum(dy ** 2 for dy in dev_y)
    denominator = (sum_sq_dev_x * sum_sq_dev_y) ** 0.5

    # Step 5: Calculate the Pearson correlation coefficient
    pearson_correlation = numerator / denominator

    return pearson_correlation


def create_public_dataset(labels, annotators):
    difficulty_levels = {"1": "basic",
                         "2": "intermediate",
                         "3": "advanced"}

    def anonymize_name(name):
        name = name.replace("hf_", "").replace("-lavita", "")
        return name

    def get_annotations(ann):
        ann_json = {}
        for criterion, votes in ann.items():
            ann_json[criterion] = {}
            for vote in votes:
                annotator = list(vote.keys())[0]
                verdict = anonymize_name(list(vote.values())[0])
                ann_json[criterion][annotators[annotator]] = verdict
        return ann_json

    batch_files = []

    for i in labels.keys():
        batch_files.append(f'../data/batches/batch{i}.csv')

    batch_df_list = [pd.read_csv(batch_file) for batch_file in batch_files]
    batch_df = pd.concat(batch_df_list, ignore_index=True)

    dataset_json = {}

    for batch_id, questions in labels.items():
        for global_key, annotations in questions.items():
            dataset_json[global_key] = {}
            dataset_json[global_key]['batch_id'] = batch_id
            row = batch_df.loc[batch_df['global_key'] == global_key]

            if len(row) == 1:
                dataset_json[global_key]['medical_question'] = str(row.iloc[0]['corrected_input_text'])
                dataset_json[global_key]['model_a'] = anonymize_name(str(row.iloc[0]['model_a']))
                dataset_json[global_key]['response_a'] = str(row.iloc[0]['response_a'])
                dataset_json[global_key]['model_b'] = anonymize_name(str(row.iloc[0]['model_b']))
                dataset_json[global_key]['response_b'] = str(row.iloc[0]['response_b'])
                dataset_json[global_key]['pair_order'] = anonymize_name(str(row.iloc[0]['pair_order']))
                annotations = get_annotations(annotations)
                annotations['difficulty']['llm'] = difficulty_levels[str(row.iloc[0]['difficulty'])]
                dataset_json[global_key]['annotations'] = annotations

    return dataset_json
