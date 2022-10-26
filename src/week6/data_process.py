from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import AutoConfig
from functools import partial
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from collections import defaultdict, OrderedDict
from collections import Counter
import string
import re
import argparse
import json
import sys

MODEL_NAME = 'xlm-roberta-base'


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_squad(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def compute_squad(predictions, references):
    pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
    dataset = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                            "id": ref["id"],
                        }
                        for ref in references
                    ]
                }
            ]
        }
    ]
    score = evaluate_squad(dataset=dataset, predictions=pred_dict)
    return score


def getLanguageDataSet(data, language):
    def printAndL(x):
        return x["language"] == language

    return data.filter(printAndL)


def get_train_features(tk, samples):
    '''
    Tokenizes all of the text in the given samples, splittling inputs that are too long for our model
    across multiple features. Finds the token offsets of the answers, which serve as the labels for
    our inputs.
    '''
    batch = tk.batch_encode_plus(
        [[q, c] for q, c in zip(samples['question_text'], samples['document_plaintext'])],
        padding='max_length',
        truncation='only_second',
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    # Get a list which maps the input features index to their original index in the
    # samples list (for split inputs). E.g. if our batch size is 4 and the second sample
    # is split into 3 inputs because it is very large, sample_mapping would look like
    # [0, 1, 1, 1, 2, 3]
    sample_mapping = batch.pop('overflow_to_sample_mapping')
    # Get all of the character offsets for each token
    offset_mapping = batch.pop('offset_mapping')

    # Store the start and end tokens
    batch['start_tokens'] = []
    batch['end_tokens'] = []

    # Iterate through all of the offsets
    for i, offsets in enumerate(offset_mapping):
        # Get the right sample by mapping it to its original index
        sample_idx = sample_mapping[i]
        # Get the sequence IDs to know where context starts so we can ignore question tokens
        sequence_ids = batch.sequence_ids(i)

        # Get the start and end character positions of the answer
        ans = samples['annotations'][sample_idx]
        start_char = ans['answer_start'][0]
        end_char = start_char + len(ans['answer_text'][0])
        # while end_char > 0 and (end_char >= len(samples['context'][sample_idx]) or samples['context'][sample_idx][end_char] == ' '):
        #   end_char -= 1

        # Start from the first token in the context, which can be found by going to the
        # first token where sequence_ids is 1
        start_token = 0
        while sequence_ids[start_token] != 1:
            start_token += 1

        end_token = len(offsets) - 1
        while sequence_ids[end_token] != 1:
            end_token -= 1

        # By default set it to the CLS token if the answer isn't in this input
        if start_char < offsets[start_token][0] or end_char > offsets[end_token][1]:
            start_token = 0
            end_token = 0
        # Otherwise find the correct token indices
        else:
            # Advance the start token index until we have passed the start character index
            while start_token < len(offsets) and offsets[start_token][0] <= start_char:
                start_token += 1
            start_token -= 1

            # Decrease the end token index until we have passed the end character index
            while end_token >= 0 and offsets[end_token][1] >= end_char:
                end_token -= 1
            end_token += 1

        batch['start_tokens'].append(start_token)
        batch['end_tokens'].append(end_token)

    # batch['start_tokens'] = np.array(batch['start_tokens'])
    # batch['end_tokens'] = np.array(batch['end_tokens'])

    return batch


def collate_fn(inputs):
    '''
    Defines how to combine different samples in a batch
    '''
    input_ids = torch.tensor([i['input_ids'] for i in inputs])
    attention_mask = torch.tensor([i['attention_mask'] for i in inputs])
    start_tokens = torch.tensor([i['start_tokens'] for i in inputs])
    end_tokens = torch.tensor([i['end_tokens'] for i in inputs])

    # Truncate to max length
    max_len = max(attention_mask.sum(-1))
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'start_tokens': start_tokens,
            'end_tokens': end_tokens}


def predict(model: nn.Module, valid_dl: DataLoader):
    """
    Evaluates the model on the given dataset
    :param model: The model under evaluation
    :param valid_dl: A `DataLoader` reading validation data
    :return: The accuracy of the model on the dataset
    """
    # VERY IMPORTANT: Put your model in "eval" mode -- this disables things like
    # layer normalization and dropout
    model.eval()
    start_logits_all = []
    end_logits_all = []

    # ALSO IMPORTANT: Don't accumulate gradients during this process
    with torch.no_grad():
        for batch in tqdm(valid_dl, desc='Evaluation'):
            batch = {b: batch[b].to('cuda') for b in batch}

            # Pass the inputs through the model, get the current loss and logits
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # Store the "start" class logits and "end" class logits for every token in the input
            start_logits_all.extend(list(outputs['start_logits'].detach().cpu().numpy()))
            end_logits_all.extend(list(outputs['end_logits'].detach().cpu().numpy()))

        return start_logits_all, end_logits_all


def post_process_predictions(examples, dataset, logits, num_possible_answers=20, max_answer_length=30):
    all_start_logits, all_end_logits = logits
    # Build a map from example to its corresponding features. This will allow us to index from
    # sample ID to all of the features for that sample (in case they were split up due to long input)
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(dataset):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # Create somewhere to store our predictions
    predictions = OrderedDict()

    # Iterate through each sample in the dataset
    for j, sample in enumerate(tqdm(examples)):

        # Get the feature indices (all of the features split across the batch)
        feature_indices = features_per_example[j]
        # Get the original context which predumably has the answer text
        context = sample['context']

        preds = []
        # Iterate through all of the features
        for ft_idx in feature_indices:

            # Get the start and end answer logits for this input
            start_logits = all_start_logits[ft_idx]
            end_logits = all_end_logits[ft_idx]

            # Get the offsets to map token indices to character indices
            offset_mapping = dataset[ft_idx]['offset_mapping']

            # Sort the logits and take the top N
            start_indices = np.argsort(start_logits)[::-1][:num_possible_answers]
            end_indices = np.argsort(end_logits)[::-1][:num_possible_answers]
            # Iterate through start and end indices
            for start_index in start_indices:
                for end_index in end_indices:

                    # Ignore this combination if either the indices are not in the context
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or offset_mapping[
                        start_index] is None or offset_mapping[end_index] is None:
                        continue

                    # Also ignore if the start index is greater than the end index of the number of tokens
                    # is greater than some specified threshold
                    if start_index > end_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    ans_text = context[offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                    preds.append({
                        'score': start_logits[start_index] + end_logits[end_index],
                        'text': ans_text
                    })

        if len(preds) > 0:
            # Sort by score to get the top answer
            answer = sorted(preds, key=lambda x: x['score'], reverse=True)[0]
        else:
            answer = {'score': 0.0, 'text': ""}

        predictions[sample['id']] = answer['text']
    return predictions


def val_collate_fn(inputs):
    input_ids = torch.tensor([i['input_ids'] for i in inputs])
    attention_mask = torch.tensor([i['attention_mask'] for i in inputs])

    # Truncate to max length
    max_len = max(attention_mask.sum(-1))
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def get_validation_features(tk, samples):
    # First, tokenize the text. We get the offsets and return overflowing sequences in
    # order to break up long sequences into multiple inputs. The offsets will help us
    # determine the original answer text
    batch = tk.batch_encode_plus(
        [[q, c] for q, c in zip(samples['question'], samples['context'])],
        padding='max_length',
        truncation='only_second',
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    # We'll store the ID of the samples to calculate squad score
    batch['example_id'] = []
    # The overflow sample map tells us which input each sample corresponds to
    sample_map = batch.pop('overflow_to_sample_mapping')

    for i in range(len(batch['input_ids'])):
        # The sample index tells us which of the values in "samples" these features belong to
        sample_idx = sample_map[i]
        sequence_ids = batch.sequence_ids(i)

        # Add the ID to map these features back to the correct sample
        batch['example_id'].append(samples['id'][sample_idx])

        # Set offsets for non-context words to be None for ease of processing
        batch['offset_mapping'][i] = [o if sequence_ids[k] == 1 else None for k, o in
                                      enumerate(batch['offset_mapping'][i])]

    return batch


def process_train_data(language):
    dataset = load_dataset("copenlu/answerable_tydiqa")
    dataset = getLanguageDataSet(dataset, language)['train']
    dataset = dataset.remove_columns("language")
    dataset = dataset.remove_columns("document_url")

    tk = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = dataset.map(partial(get_train_features, tk), batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset


def process_validation_data(language):
    dataset = load_dataset("copenlu/answerable_tydiqa")
    dataset = getLanguageDataSet(dataset, language)['validation']
    dataset = dataset.remove_columns("language")
    dataset = dataset.remove_columns("document_url")

    tk = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = dataset.map(partial(get_train_features, tk), batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset
