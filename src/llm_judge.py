import ast
import pickle
import random
import pandas as pd
from pprint import pprint
from collections import Counter
from collections import OrderedDict

from utils import jsonify
from annotation_tools import LabelAnalysis

label_analysis = LabelAnalysis()

label_to_int = {
    'response_a': 1,
    'response_b': 2,
    'tie': 3,
    'neither': 4
}

llms = ["openai_gpt-4o-2024-05-13", "anthropic_claude-3-5-sonnet-20240620"]


def get_labels(runs, num_batches=4):
    labels = {}

    for batch_id in range(1, num_batches + 1):

        labels[batch_id] = {}

        file_path = f"../data/llm_judge/batch{batch_id}-llm.csv"

        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            global_key = row['global_key']

            labels[batch_id][global_key] = {}

            for llm in llms:
                labels[batch_id][global_key][llm] = {}

                for run_id in runs:
                    vote = row[f"pred_{llm}_{run_id}"]

                    json_object = {}
                    # jsonify the model's response (by default, LLM's output is saved as string)
                    if llm.startswith('openai'):
                        parsed_list = ast.literal_eval(vote)
                        json_object = parsed_list['response']
                    elif llm.startswith('anthropic'):
                        json_object = jsonify(vote)

                    for criterion, verdict in json_object.items():
                        labels[batch_id][global_key][llm].setdefault(criterion, []).append(verdict['verdict'])

    return labels


def get_majority_vote(llm_votes):
    vote_counts = Counter(llm_votes)
    most_common, count = vote_counts.most_common(1)[0]

    # Return the most common vote if it has a majority
    if count > 1:
        return most_common

    # Handle cases with three distinct votes
    if len(vote_counts) == 3:
        # Randomly choose between 'tie' or 'neither' if both are present
        if 'tie' in vote_counts and 'neither' in vote_counts:
            return random.choice(['tie', 'neither'])
        # Return 'tie' or 'neither' if either is present
        return 'tie' if 'tie' in vote_counts else 'neither' if 'neither' in vote_counts else None

    return None


def find_majority(labels, num_batches=4):
    disagreement = {}
    no_majority = 0

    for batch_id in range(1, num_batches + 1):
        for global_key, llms_votes in labels[batch_id].items():
            for llm, criteria in llms_votes.items():
                for criterion, votes in criteria.items():
                    unique_votes = set(votes)

                    # Case 1: All votes are the same
                    if len(unique_votes) == 1:
                        labels[batch_id][global_key][llm][criterion] = votes[0]
                    else:
                        # Case 2: Determine the majority vote
                        majority_vote = get_majority_vote(votes)
                        labels[batch_id][global_key][llm][criterion] = majority_vote

                        # Track disagreement counts
                        disagreement[llm] = disagreement.get(llm, 0) + 1

                        # Count cases where there's no clear majority
                        if len(unique_votes) == 3:
                            no_majority += 1

    return labels, (disagreement, no_majority)


# Main execution block
def main(num_batches=4):
    # reading all labels from csv files
    labels_ab = get_labels(['ab1', 'ab2', 'ab3'])
    labels_ba = get_labels(['ba1', 'ba2', 'ba3'])

    # for each model, find the majority vote across ab and ba runs
    # at the end of this step, we will have one vote for each criterion of each question for each model per run
    labels_ab, stat_ab = find_majority(labels_ab)
    labels_ba, stat_ba = find_majority(labels_ba)

    print("Vote inconsistency in [ab] runs by models")
    print("-----------------------------------------")
    pprint(stat_ab)
    print("\nVote inconsistency in [ba] runs by models")
    print("-----------------------------------------")
    pprint(stat_ba)

    models_votes = {}
    disagreements = {}

    # now, we need to compare ab and ba run votes for each model and finalize model votes across runs
    for batch_id in range(1, num_batches + 1):
        models_votes[batch_id] = {}
        for question_id, models in labels_ab[batch_id].items():
            models_votes[batch_id][question_id] = {}
            for model, criteria in models.items():
                for criterion, vote_ab in criteria.items():
                    vote_ba = labels_ba[batch_id][question_id][model][criterion]
                    # note that order of responses is reversed in ba runs.
                    # so response b in ba run is the same as response a in ab run
                    if (vote_ab == 'response_a' and vote_ba == 'response_b') or (
                            vote_ab == 'response_b' and vote_ba == 'response_a') or (
                            vote_ab == 'tie' and vote_ba == 'tie') or (vote_ab == 'neither' and vote_ba == 'neither'):
                        # agreement between the ab and ba runs of the same model
                        models_votes[batch_id][question_id].setdefault(criterion, []).append(vote_ab)
                    else:
                        disagreements.setdefault(model, {}).setdefault(criterion, {'count': 0})
                        disagreements[model][criterion]['count'] += 1

                        # following "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," in the disagreement case,
                        # we take a conservative approach and will take "tie" and "neither" as votes
                        if criterion in ['harmfulness', 'bias']:
                            models_votes[batch_id][question_id].setdefault(criterion, []).append('neither')
                        else:
                            models_votes[batch_id][question_id].setdefault(criterion, []).append('tie')

    # Sort merged_dict based on count in descending order
    sorted_disagreements = {}

    for model, criteria_dict in disagreements.items():
        # Sort criteria_dict based on the count in descending order
        sorted_criteria_dict = OrderedDict(
            sorted(criteria_dict.items(), key=lambda item: item[1]['count'], reverse=True))
        sorted_disagreements[model] = sorted_criteria_dict

    print("\nDisagreements between [ab] and [ba] runs by model")
    print("-------------------------------------------------")
    for k, v in sorted_disagreements.items():
        print(f'* model: {k}')
        pprint(v)

    final_votes = {}
    llm_1_votes = []
    llm_2_votes = []

    # now that we have one final vote per criterion, per question, per model, we compare models' votes
    for batch_id in range(1, num_batches + 1):

        final_votes[batch_id] = {}

        for question_id, criteria in models_votes[batch_id].items():

            final_votes[batch_id][question_id] = {}

            for criterion, votes in criteria.items():

                if len(votes) == 2:
                    # It means for this criterion and this question, both models were consistent in their judgment,
                    # and that's why we have two votes from two models

                    llm_1_votes.append(label_to_int[votes[0]])
                    llm_2_votes.append(label_to_int[votes[1]])

                    if len(list(set(votes))) == 1:
                        # if yes, both votes are the same, and there's agreement between the two models
                        final_votes[batch_id][question_id][criterion] = votes[0]
                    else:
                        # following "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," in the disagreement case,
                        # we take a conservative approach and will take "tie" and "neither" as votes
                        if criterion in ['harmfulness', 'bias']:
                            final_votes[batch_id][question_id][criterion] = 'neither'
                        else:
                            final_votes[batch_id][question_id][criterion] = 'tie'
                else:
                    print("There's a missing vote")

    print("\nAgreement between the two llm judges")
    print("------------------------------------")
    agreements = label_analysis.calculate_agreement(llm_1_votes, llm_2_votes)
    pprint(agreements)

    with open('../data/llm_judge/llm_judgements.pkl', 'wb') as file:
        pickle.dump(final_votes, file)


if __name__ == "__main__":
    main()
