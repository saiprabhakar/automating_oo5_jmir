import re
import constants

def cluster_numbers(numbers, threshold):
    """
    Clusters numbers that are within `threshold` of each other
    """
    clusters = []
    current_cluster = []
    # Sort the numbers in ascending order
    numbers.sort()
    # cluster number within threshold
    for i in range(len(numbers)):
        if not current_cluster:
            current_cluster.append(numbers[i])
        else:
            if numbers[i] - current_cluster[-1] <= threshold:
                current_cluster.append(numbers[i])
            else:
                clusters.append(current_cluster.copy())
                current_cluster = [numbers[i]]
    # consider the last cluster that was not appended
    if current_cluster:
        clusters.append(current_cluster)
    return clusters

def isoverlap(range1, range2):
    """
    Check if two ranges overlap
    """
    if range1[0] <= range2[1] and range1[1] >= range2[0]:
        return True
    return False

def calculate_precision_recall(gt_ranges_expanded, pred_ranges):
    """
    Calculate precision and recall for the given gt and pred ranges
    """
    true_positives, false_positives, false_negatives = 0, 0, 0
    tp_clusters_in_gt_expanded, tp_clusters_in_pred = [], []
    fn_clusters_in_gt_expanded, fn_clusters_in_pred = [], []
    # compare gt and pred clusters to find true positives and false negatives
    for gt_range in gt_ranges_expanded:
        overlap_found = False
        for pred_range in pred_ranges:
            if isoverlap(pred_range, gt_range):
                overlap_found = True
                break
        if overlap_found:
            true_positives += 1
            tp_clusters_in_gt_expanded.append(gt_range)
            tp_clusters_in_pred.append(pred_range)
        else:
            false_negatives += 1
            fn_clusters_in_gt_expanded.append(gt_range)
            fn_clusters_in_pred.append(pred_range)

    # compare gt and pred clusters to find false positives
    fp_clusters_in_pred = []    
    for pred_range in pred_ranges:
        overlap_found = False
        for gt_range in gt_ranges_expanded:
            if isoverlap(pred_range, gt_range):
                overlap_found = True
                break
        if not overlap_found:
            false_positives += 1
            fp_clusters_in_pred.append(pred_range)
    
    # calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    cluster_types = {
        'tp_clusters_in_gt_expanded': tp_clusters_in_gt_expanded,
        'tp_clusters_in_pred': tp_clusters_in_pred,
        'fn_clusters_in_pred': fn_clusters_in_pred,
        'fn_clusters_in_gt_expanded': fn_clusters_in_gt_expanded,
        'fp_clusters_in_pred': fp_clusters_in_pred}
    agg_info = {
        'precision': precision, 'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }
    return agg_info, cluster_types

def get_metrics(y_idx, pred_idx):
    """
    Get precision and recall for the given gt and pred indices
    """
    # cluster the indices
    y_clusters = cluster_numbers(y_idx, constants.PROIMITY_THRESHOLD)
    pred_clusters = cluster_numbers(pred_idx, constants.PROIMITY_THRESHOLD)
    # expand the gt by offset
    y_idx_expanded = sorted(set([
        x 
        for y_idx in y_clusters
        for y in y_idx
        for x in range(y-constants.OFFSET, y+constants.OFFSET+1)
    ]))
    y_clusters_expanded = cluster_numbers(y_idx_expanded, 1)

    # convert the clusters to ranges
    y_range_expanded = [[x[0], x[-1]] for x in y_clusters_expanded]
    pred_ranges = [[x[0], x[-1]] for x in pred_clusters]
    
    # find precision and recall by finding overlapping ranges
    metrics, clusters = calculate_precision_recall(y_range_expanded, pred_ranges)
    return metrics, clusters

def parse_and_get_utt_ids_v3(input_string):
    """
    Parse the input string preductions from LLM and get the sentence ids
    """
    input_string = input_string.lower()
    matches = re.findall(r'<(.*?)>', input_string)
    matches = [
        x.replace('sentence ids', '').replace('sentence id', '').replace('sentence', '')
        for x in matches]
    matches = [
        " ".join(x.replace(',', ' ').replace(':', ' ').replace('-', ' ').replace('/', ' ').strip().split())
        for x in matches]

    # print(matches)
    sentence_ids = []

    for match in matches:
        assert bool(re.search(r'[^0-9. ]', match)) == False, f"Invalid output format {match}"

        ranges = re.findall(r'\d+\.\d+|\d+', match)
        if not ranges: # this can happen if the sentence ids are empty
            continue
        
        # Check if it's a range or a single ID
        if len(ranges) == 2:
            start, end = int(float(ranges[0])), int(float(ranges[1]))
            for i in range(start, end + 1):
                sentence_ids.append(i)
        else:
            sentence_ids.append(int(float(ranges[0])))
    return sentence_ids

def get_google_prompt(messages):
    prompt = []
    for m in messages:
        if m['role'] == 'system':
            content = [m['content'], '']
        elif m['role'] == 'user':
            content = ['Input:', m['content']]
        else:
            content = ['Output:', m['content'], '']
        prompt.extend(content)
        
    prompt = "\n".join(prompt)
    return prompt

def format_conv(utterances, utterances_ids):
    """
    Format the conversation for LLM input
    0 - hi
    1 - hello
    ...
    """
    formatted_conv = []
    for utterance, utterance_id in zip(utterances, utterances_ids):
        formatted_conv.append(f'{utterance_id} - {utterance}')
    return "\n".join(formatted_conv)

def get_messages(utterances, utterances_ids, use_few_shot=False,
                 example_inputs=None, example_outputs=None,
                 use_explanation=True):
    """
    Get the messages that will be used as input for LLM
    """
    formatted_conv = format_conv(utterances, utterances_ids)
    messages = [{"role": "system", "content": constants.system_prompt}]

    if use_explanation:
        option_prompt = constants.option_prompt_original
    else:
        option_prompt = constants.option_prompt_no_explanation
        
    if use_few_shot:
        for _, (example_input, example_output)  in enumerate(zip(example_inputs, example_outputs)):
            messages.append(
                {"role": "user", "content": f'Conversation:\n{example_input}\nInstruction:{option_prompt}'},
            )
            messages.append(
                {"role": "assistant", "content": example_output}
            )
    messages.append(
        {"role": "user", "content": f'Conversation:\n{formatted_conv}\nInstruction:{option_prompt}'}
    )
    return messages
