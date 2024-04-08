# Option talk analysis - code release for JMIR paper
- We run OpenAI and Google PaLM models on the example conversation to predict transcript lines that are positive for item 1 Observer Option 5 (OO5)

## Contents:
This repo contains code and algo for
- Formatting sentences in the example conversation file along with instruction/prompt and few-shot examples into input for OpenAI and PaLM models
- Parsing the predictions from the LLMs to find sentences that are predicted positive for OO5 item 1
- Evaluating the predictions
    - Clustering of ground truth and predictions based on proximity and a proximity threshold of 1
    - Comparing the clusters of ground truth and predictions allowing for an offset of 1 to find the ones that are TP, FP, and FN
    - Computing Precision, Recall, and F1 with TP, FP, and FN clusters

## Data
- Example conversation is stored in `example_conv_sents.txt`, the transcript lines that are ground truth for item 1 Observer Option 5 are marked as `gt`.

## Installation instructions
```
conda create -n oo5 python=3.10
conda activate oo5
pip install -r requirements.txt
```

## Running the code
- Download OpenAI credentials and save it in `cred.json` it will be of the format `{"api_key": "...", "organization_id": "..."}`.
- Authenticate GCloud if using `text-bison` model, but it is disabled by default.
- Run `generate_llm_oo5_predictions_and_evaluate.ipynb` or `generate_llm_oo5_predictions_and_evaluate.py`

### Run via Notebook `generate_llm_oo5_predictions_and_evaluate.ipynb` 
- Use run all

### Run via script `generate_llm_oo5_predictions_and_evaluate.py`
- the script is equvalent to the notebook above, it was generate using `jupyter nbconvert --to script generate_llm_oo5_predictions_and_evaluate.ipynb`
- run it with `python generate_llm_oo5_predictions_and_evaluate.py`
- Output:
```
running gpt-3.5-turbo-0301
running gpt-3.5-turbo-0301
running gpt-3.5-turbo-0301
running gpt-3.5-turbo-0301
model: gpt-3.5-turbo-0301 use_few_shot: True use_explanation: True
pred_sentence_ids_list_all [10, 11, 12, 14, 16, 17, 18, 21, 22, 24, 25, 26] gt_idx [12, 13, 14, 17, 18, 30]
clusters {'tp_clusters_in_gt_expanded': [[11, 19]], 'tp_clusters_in_pred': [[10, 18]], 'fn_clusters_in_pred': [[21, 26]], 'fn_clusters_in_gt_expanded': [[29, 31]], 'fp_clusters_in_pred': [[21, 26]]}
metrics {'precision': 0.5, 'recall': 0.5, 'true_positives': 1, 'false_positives': 1, 'false_negatives': 1}

model: gpt-3.5-turbo-0301 use_few_shot: False use_explanation: True
pred_sentence_ids_list_all [10, 14, 15, 17, 18, 21, 22, 24, 25, 26] gt_idx [12, 13, 14, 17, 18, 30]
clusters {'tp_clusters_in_gt_expanded': [[11, 19]], 'tp_clusters_in_pred': [[14, 18]], 'fn_clusters_in_pred': [[21, 26]], 'fn_clusters_in_gt_expanded': [[29, 31]], 'fp_clusters_in_pred': [[10, 10], [21, 26]]}
metrics {'precision': 0.3333333333333333, 'recall': 0.5, 'true_positives': 1, 'false_positives': 2, 'false_negatives': 1}

model: gpt-3.5-turbo-0301 use_few_shot: False use_explanation: False
pred_sentence_ids_list_all [10, 14, 15, 17, 18, 21, 22, 24, 25, 26] gt_idx [12, 13, 14, 17, 18, 30]
clusters {'tp_clusters_in_gt_expanded': [[11, 19]], 'tp_clusters_in_pred': [[14, 18]], 'fn_clusters_in_pred': [[21, 26]], 'fn_clusters_in_gt_expanded': [[29, 31]], 'fp_clusters_in_pred': [[10, 10], [21, 26]]}
metrics {'precision': 0.3333333333333333, 'recall': 0.5, 'true_positives': 1, 'false_positives': 2, 'false_negatives': 1}

model: gpt-3.5-turbo-0301 use_few_shot: True use_explanation: False
pred_sentence_ids_list_all [10, 11, 12, 14, 16, 17, 18, 21, 22, 24, 25, 26] gt_idx [12, 13, 14, 17, 18, 30]
clusters {'tp_clusters_in_gt_expanded': [[11, 19]], 'tp_clusters_in_pred': [[10, 18]], 'fn_clusters_in_pred': [[21, 26]], 'fn_clusters_in_gt_expanded': [[29, 31]], 'fp_clusters_in_pred': [[21, 26]]}
metrics {'precision': 0.5, 'recall': 0.5, 'true_positives': 1, 'false_positives': 1, 'false_negatives': 1}

  tag               model  use_few_shot  precision  recall        f1  num_file  use_explanation
0  _1  gpt-3.5-turbo-0301          True   0.500000     0.5  0.500000         1             True
1  _1  gpt-3.5-turbo-0301         False   0.333333     0.5  0.333333         1             True
2  _1  gpt-3.5-turbo-0301         False   0.333333     0.5  0.333333         1            False
3  _1  gpt-3.5-turbo-0301          True   0.500000     0.5  0.500000         1            False
```