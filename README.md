# Option talk analysis - code release for JMIR paper
- We run OpenAI and Google PaLM models on the example conversation to predict transcript lines that are positive for item 1 Observer Option 5 (OO5)

## Installation instructions
```
conda create -n oo5 python=3.10
conda activate oo5
pip install -r requirements.txt
```

## Running
- Download OpenAI credentials and save it in `cred.json` it will be of the format `{"api_key": "...", "organization_id": "..."}`.
- Authenticate GCloud if using `text-bison` model, but it is disabled by default.
- Example conversation is stored in `example_conv_sents.txt`, the transcript lines that are ground truth for item 1 Observer Option 5 are marked as `gt`.
- Run `generate_llm_oo5_predictions_and_evaluate.ipynb`.

## Notebook `generate_llm_oo5_predictions_and_evaluate.ipynb` contains code and algorithm for
- Formatting sentences in the example conversation file along with instruction/prompt and few-shot examples into input for OpenAI and PaLM models
- Parsing the predictions from the LLMs to find sentences that are predicted positive for OO5 item 1
- Evaluating the predictions
    - Clustering of ground truth and predictions based on proximity and a proximity threshold of 1
    - Comparing the clusters of ground truth and predictions allowing for an offset of 1 to find the ones that are TP, FP, and FN
    - Computing Precision, Recall, and F1 with TP, FP, and FN clusters


