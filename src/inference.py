import os
import re
import json
import requests
import tiktoken
import anthropic
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
from together import Together
from huggingface_hub import InferenceClient

from utils import normalize_json_response

load_dotenv()

huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')


class GenTools:
    def __init__(self):
        self.clients = {
            'openai': OpenAI(),
            'anthropic': anthropic.Anthropic(),
            'together': Together()
        }
        self.endpoint_manager = EndpointManager()
        self.hf_endpoints = HuggingFaceEndpoints()

    def get_client(self, model_name):
        if model_name.startswith('openai'):
            return self.clients['openai']
        elif model_name.startswith('anthropic'):
            return self.clients['anthropic']
        elif model_name.startswith('together'):
            return self.clients['together']
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def execute_prompt(self, user_prompt, model_name, system_prompt="You are a helpful assistant.", output_json=True):
        if output_json:
            system_prompt = "You are a helpful assistant designed to output JSON."
        model_url = self.endpoint_manager.endpoint_urls[model_name]
        if model_name.startswith('hf_'):
            return self.hf_endpoints.get_llm_response(user_prompt, model_url, use_openai=False, temperature=0,
                                                      max_tokens=1024)
        elif model_name.startswith(('openai', 'together')):
            client = self.get_client(model_name)

            params = {
                "model": model_url,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            if not model_name.startswith('together'):
                params['seed'] = 42
                params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**params)
            response_json = self._process_response(response, output_json)
            response_json['prompt_tokens'] = response.usage.prompt_tokens
            response_json['completion_tokens'] = response.usage.completion_tokens
            response_json['total_tokens'] = response.usage.total_tokens
            return json.dumps(response_json)
        elif model_name.startswith('anthropic'):
            response = self.get_client(model_name).messages.create(
                model=model_url,
                max_tokens=2048,
                system=system_prompt,
                temperature=0,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response
        else:
            raise Exception("model is not found on the list of endpoints.")

    @staticmethod
    def _process_response(response, output_json):
        content = response.choices[0].message.content.strip()

        if output_json:
            normalized_content = normalize_json_response(content.replace('\n', ''))
            return {'response': json.loads(normalized_content)}
        else:
            return {'response': content}

    @staticmethod
    def build_prompt(prompt_template, arguments):
        if not isinstance(arguments, list):
            raise TypeError("Arguments must be in a list.")
        if not all(isinstance(x, str) for x in arguments):
            raise TypeError("All arguments must be strings.")

        placeholder_pattern = r'\{\{[A-Z_]+\}\}'  # e.g., {{INSERT_QUESTION_HERE}}
        placeholder_matches = re.findall(placeholder_pattern, prompt_template)

        if len(placeholder_matches) != len(arguments):
            raise Exception(
                """the number of parameters doesn't match the number of placeholders in the prompt template.
                # {} parameters: {}
                # {} prompt placeholders: {}""".format(len(arguments),
                                                       ', '.join(arguments),
                                                       len(placeholder_matches),
                                                       ','.join(placeholder_matches)))
        else:
            for placeholder, replacement in zip(placeholder_matches, arguments):
                prompt_template = prompt_template.replace(placeholder, replacement)
            return prompt_template

    def get_openai_embedding(self, text,
                             model="text-embedding-3-small",
                             embedding_size=512):
        # some values to use based on model (reference: https://supabase.com/blog/matryoshka-embeddings)
        # text-embedding-3-small: 512 and 1536
        # text-embedding-3-large: 256, 1024, and 3072
        return self.clients['openai'].embeddings.create(input=text,
                                                        model=model,
                                                        dimensions=embedding_size).data[0].embedding

    def get_embeddings(self, df,
                       text_field='corrected_query',
                       max_tokens=8000,
                       embedding_model="text-embedding-3-large",
                       embedding_size=1024,
                       embedding_encoding="cl100k_base"):
        """
        This method loads or computes embeddings for the given DataFrame.
        If embeddings already exist, it loads them. Otherwise, it computes new embeddings.
        """
        # Step 1: Check if embeddings already exist
        if 'embedding' in df.columns:
            # Convert stored string embeddings to numpy arrays if necessary
            if isinstance(df.iloc[0]['embedding'], str):
                df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)).reshape(1, -1))
            return df

        # Step 2: Compute embeddings if they don't exist
        encoding = tiktoken.get_encoding(embedding_encoding)
        df["n_tokens"] = df[text_field].apply(lambda x: len(encoding.encode(x)))
        df = df[df.n_tokens <= max_tokens]
        df["embedding"] = df[text_field].apply(lambda x: self.get_openai_embedding(
            x, model=embedding_model, embedding_size=embedding_size))

        return df
    
    def save_embeddings(self, df, save_path):
        if 'embedding' not in df.columns:
            raise Exception("Embedding column does not exist")

        def convert_to_json(x):
            # Check if the embedding is already a valid JSON string
            if isinstance(x, str):
                try:
                    # Try to load it; if successful, it's already a valid JSON string
                    json.loads(x)
                    return x
                except json.JSONDecodeError:
                    pass

            # If it's a numpy array, convert to list first
            if isinstance(x, np.ndarray):
                x = x.tolist()

            # If it's a nested list, flatten it to a single list
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                x = [item for sublist in x for item in sublist]

            # Convert to JSON string
            return json.dumps(x)

        # Apply the conversion function to each row
        df['embedding'] = df['embedding'].apply(convert_to_json)

        # Save the DataFrame to CSV
        df.to_csv(save_path, index=False)


class EndpointManager:
    def __init__(self):
        self.endpoint_urls = {
            "hf_biomistral-7b-dare": "your_huggingface_endpoint_url",
            "hf_meditron3-70b": "your_huggingface_endpoint_url",
            "hf_Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "hf_alpacare-llama2-13b": "your_huggingface_endpoint_url",
            "openai_gpt-4-0125-preview": "gpt-4-0125-preview",
            "openai_gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
            "openai_gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
            "anthropic_claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
            "together_llama-3.1-405b-instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        }

    def get_endpoint_url(self, model_name):
        return self.endpoint_urls.get(model_name)

    def set_endpoint_url(self, model_name, url):
        self.endpoint_urls[model_name] = url

    def list_models(self):
        return list(self.endpoint_urls.keys())


class HuggingFaceEndpoints:
    def __init__(self):
        pass

    def get_llm_response(self, prompt, model_url,
                         use_openai=False,
                         do_stream=False,
                         temperature=0.001,
                         max_tokens=1024):
        if use_openai:
            client = OpenAI(
                base_url="{}/v1/".format(model_url),
                api_key=huggingface_api_key
            )

            chat_completion = client.chat.completions.create(
                model="tgi",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=do_stream,
                seed=42,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
        else:
            # check if it's a dedicated endpoint
            if "huggingface.cloud" in model_url:
                headers = {
                    "Accept": "application/json",
                    "Authorization": "Bearer {}".format(huggingface_api_key),
                    "Content-Type": "application/json"
                }

                def query(payload):
                    query_response = requests.post(model_url, headers=headers, json=payload)
                    return query_response.json()

                response = query({
                    "inputs": prompt,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens
                    }
                })
            else:
                # it means the model is on the Inference API (serverless)
                client = InferenceClient(
                    model_url,
                    token=huggingface_api_key,
                )

                chat_completion = client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    stream=do_stream,
                    seed=42,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response = chat_completion.choices[0].message.content

        return response
