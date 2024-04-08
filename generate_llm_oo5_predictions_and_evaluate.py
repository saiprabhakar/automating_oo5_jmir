#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from collections import defaultdict

import pandas as pd
import openai
from vertexai.language_models import TextGenerationModel

import constants
import utils


# # open ai credentials

# In[2]:


with open('cred.json', 'r') as f:
    cred_json = json.load(f)

openai.api_key = cred_json['api_key']
openai.organization_id = cred_json['organization_id']
# list_models = openai.Model.list()
# [x.id for x in list_models['data'] if 'gpt' in x.id]


# ## get example conversations

# In[3]:


# extract conversations and gt
example_sentences, example_sentences_idx, gt_idx = [], [], []
with open('example_conv_sents.txt', 'r') as f:
    file_lines = f.read().splitlines()
    for x in file_lines:
        x = x.strip()

        if not x:
            continue
        if x.startswith("gt"):
            idx = int(x[3:].split(":")[0])
            sent = ":".join(x[3:].strip().split(":")[1:]).strip()
            gt_idx.append(idx)
        else:
            idx = int(x.split(":")[0])
            sent = ":".join(x.split(":")[1:]).strip()

        example_sentences.append(sent)
        example_sentences_idx.append(idx)

example_sentences[:3], example_sentences_idx[:3], gt_idx


# ## Get predictions for OO5 item 1 from OpenAI and Google models

# In[5]:


fileid = "example"
response_all = defaultdict(list)

# (model, use_few_shot, use_explanation)
eval_options = [
    ('gpt-3.5-turbo-0301', True, True),
    ('gpt-3.5-turbo-0301', False, True),
    ('gpt-3.5-turbo-0301', False, False),
    ('gpt-3.5-turbo-0301', True, False),
    # ('text-bison@001', True, True),
 ]
tag = f'_1'

for model, use_few_shot, use_explanation in eval_options:
    print('running', model)

    # batch the examples into batches of BATCH_SIZE
    idx = list(range(0, len(example_sentences), constants.BATCH_SIZE))
    example_sentences_idx = list(range(len(example_sentences)))

    # predict for each batch
    for batchid, (st, end) in enumerate(zip(idx[:], idx[1:] + [len(example_sentences)])):
        utterances = example_sentences[st:end]
        utterances_ids = example_sentences_idx[st:end]
        
        # Create a list of message objects - input to the LLM model
        start_offset = list(utterances_ids)[0] if constants.USE_ZERO_ALL_BATCHES else 0
        messages = utils.get_messages(utterances, [x-start_offset for x in utterances_ids],
                                use_few_shot=use_few_shot,
                                example_inputs=constants.example_inputs,
                                example_outputs=constants.example_outputs, use_explanation=use_explanation)
        
        # get predictions for OpenAI and Google models
        # process google model's input differently as per their documentation
        if 'bison' in model:
            google_prompt = get_google_prompt(messages)
            parameters = {
                "max_output_tokens": 256,
                "temperature": 0.0,
                "top_p": 1,
                "top_k": 40
            }
            google_model = TextGenerationModel.from_pretrained("text-bison@001")
            response = google_model.predict(
                google_prompt,
                **parameters
            )

            response_all[f"{fileid}_{tag}_{model}_fewshot_{use_few_shot}_useexp_{use_explanation}"].append({
                'vendor': 'google', 'utterances_ids': utterances_ids, 'response': response,
                "messages": messages, "google_prompt": google_prompt, "model": model, 'start_idx': start_offset})
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1,
            )
        # store the LLM predictions
        response_all[f"{fileid}_{tag}_{model}_fewshot_{use_few_shot}_useexp_{use_explanation}"].append({
            'utterances_ids': utterances_ids, 'response': response, "messages": messages, 'start_offset': start_offset})


# ## process predictions and evaluate the models

# In[7]:


display_metric = []
for model, use_few_shot, use_explanation in eval_options:
    print("model:", model, "use_few_shot:", use_few_shot, "use_explanation:", use_explanation)
    metrics_all = defaultdict(list)

    llm_out = response_all[f"{fileid}_{tag}_{model}_fewshot_{use_few_shot}_useexp_{use_explanation}"]
    
    # parse and get the valid sentence ids from the LLM predictions
    pred_sentence_ids_list_all = []
    for llm_response in llm_out:
        # get the predictions for the LLM models
        if 'bison' in model:
            out = llm_response['response'].text.strip()
        else:
            out = llm_response['response'].choices[0].message["content"].strip()

        # parse the predictions to get the sentence ids
        pred_sentence_ids_list = utils.parse_and_get_utt_ids_v3(out)

        # get the start offset if it's not zero
        start_offset = llm_response.get('start_offset', 0)
        pred_sentence_ids_list = [int(start_offset + x) for x in pred_sentence_ids_list]
        
        # check if the sentence ids are valid - if it is in the rane of the input sentences
        input_utterances_ids = [int(x) for x in set(llm_response['utterances_ids'])]
        not_found = []
        for sentence_id in pred_sentence_ids_list:
            if sentence_id not in input_utterances_ids:
                print("Sentence ID {} not found in utterances".format(sentence_id))
                not_found.append(sentence_id)
        for id in not_found:
            pred_sentence_ids_list.remove(id)
        pred_sentence_ids_list_all.extend(pred_sentence_ids_list)

    # get the TP, FP, FN clusters for the predictions
    pred_array = [1 if x in pred_sentence_ids_list_all else 0 for x in example_sentences_idx]
    pred_sentence_ids_list_all = list(set(pred_sentence_ids_list_all))
    metrics, clusters = utils.get_metrics(gt_idx, pred_sentence_ids_list_all)

    for k, v in metrics.items():
        metrics_all[k].append(v)

    # calculate precision, recall and f1
    precisiondenominator = max(1, sum(metrics_all['true_positives']) + sum(metrics_all['false_positives']))
    recalldenominator = max(1, sum(metrics_all['true_positives']) + sum(metrics_all['false_negatives']))
    precision = sum(metrics_all['true_positives'])/precisiondenominator
    recall = sum(metrics_all['true_positives'])/recalldenominator
    f1 = 2*precision*recall/max(precision+recall, 1)

    display_metric.append({
        'tag': tag,
        'model':model, 'use_few_shot': use_few_shot,
        'precision': precision, 'recall': recall, 'f1': f1,
        'num_file': 1, 'use_explanation': use_explanation
    })

    print('pred_sentence_ids_list_all', pred_sentence_ids_list_all, 'gt_idx', gt_idx)
    print('clusters', clusters)
    print('metrics', metrics)
    print()


# # Results on the example conversation

# In[ ]:


from IPython.display import display


# In[12]:


metrics_df = pd.DataFrame(display_metric)
display(metrics_df)

