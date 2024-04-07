import json, collections
from argparse import ArgumentParser
from ast import literal_eval


# 计算span
def compute_multi_spans(gold_spans, pred_spans):
    """
    Compute the F1 score for multiple spans.
    :param gold_spans: A list of ground truth spans, each represented as a (start, end) tuple
    :param pred_spans: A list of predicted spans, each represented as a (start, end) tuple
    :return: The F1 score
    """
    gold_sets = [set(range(int(start), int(end))) for start, end in gold_spans]
    pred_sets = [set(range(int(start), int(end))) for start, end in pred_spans]

    gold_union = set.union(*gold_sets)
    pred_union = set.union(*pred_sets)

    tp = len(gold_union.intersection(pred_union))
    pred_len = len(pred_union)
    gold_len = len(gold_union)

    return tp, pred_len, gold_len


def find_start_end_index(str, sub_str, strict=False):
    """
    Get the (start, end) index for span in the response string
    :param str: LLM response string
    :param sub_str: predicted hallucination spans
    :return: (start, end), where str[start:end] is sub_str
    """
    str = str.lower()
    sub_str = sub_str.lower().strip()
    start = str.find(sub_str)
    if start == -1:
        if len(sub_str) >= 10 and not strict:
            # Allow matching 10 characters at the beginning and end, with a total length ratio between [0.7, 1.3].
            match_length = 10
            prefix_match = str.find(sub_str[:match_length])
            suffix_match = str.find(sub_str[-match_length:])
            new_start, new_end = prefix_match, suffix_match + match_length
            
            if prefix_match != -1 and suffix_match != -1 and 0.7 < (new_end - new_start)/len(sub_str) < 1.3:
                return new_start, new_end
            else:
                sub_str = ' '.join(sub_str.split())
                start = str.find(sub_str)
                if start == -1:
                    return -1, -1
    end = start + len(sub_str)
    return (start, end)

def cal_metrics(response, labels, output):
    """
    Get the span level and case level results
    :param sample: LLM response string
    :param sub_str: predicted hallucination spans
    :return: match_size, prediction, groud_truth, pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt
    """
    try:
        output_list = literal_eval(output)
    except:
        # if fail to parse, we set it as empty
        output_list = []
            
    predict_span = []
    
    for output in output_list:
        tup = find_start_end_index(response, output, strict=False)
        if tup != (-1, -1):
            predict_span.append(tup)
    
    ground_truth_span = []
    for label in labels:
        tup = (label['start'], label['end'])
        ground_truth_span.append(tup)

    if len(predict_span) == 0:
        predict_span.append((0,0))
    if len(ground_truth_span) == 0:
        ground_truth_span.append((0,0))
        
    match_size, prediction, groud_truth = compute_multi_spans(ground_truth_span, predict_span)
    
    # case_level
    pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt = 0, 0, 0, 0

    if output_list!=[] and labels!=[]:
        pos_true_cnt += 1
    elif output_list==[] and labels!=[]:
        neg_true_cnt += 1
    elif output_list==[] and labels==[]:
        neg_false_cnt += 1
    elif output_list!=[] and labels==[]:
        pos_false_cnt += 1
    else:
        raise Exception('Error! output: {} labels: {}'.format(output, labels))
    return match_size, prediction, groud_truth, pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt

def print_metrics(task_cur):
    """
    print span level and case level metrics
    :param task_cur: task type
    :return: case level results and char level results
    """
    print("\ntask_type: {}".format(task_cur))
    
    record['recall'+task_cur] = record['match_size_'+task_cur] / record['groud_truth_'+task_cur] if record['groud_truth_'+task_cur] else 0
    record['precision'+task_cur] = record['match_size_'+task_cur] / record['prediction_'+task_cur] if record['prediction_'+task_cur] else 0
    record['f1'+task_cur] = 2 * record['recall'+task_cur] * record['precision'+task_cur] / (record['recall'+task_cur] + record['precision'+task_cur]) if (record['recall'+task_cur] + record['precision'+task_cur]) > 0 else 0  
    char_res = "span_level:\nprecision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(record['precision'+task_cur], record['recall'+task_cur], record['f1'+task_cur])
    
    record['case_recall_'+task_cur] = record['pos_true_cnt_'+task_cur] / (record['pos_true_cnt_'+task_cur] + record['neg_true_cnt_'+task_cur]) if (record['pos_true_cnt_'+task_cur] + record['neg_true_cnt_'+task_cur]) > 0 else 0
    record['case_precision_'+task_cur] = record['pos_true_cnt_'+task_cur] / (record['pos_true_cnt_'+task_cur] + record['pos_false_cnt_'+task_cur]) if (record['pos_true_cnt_'+task_cur] + record['pos_false_cnt_'+task_cur]) > 0 else 0
    record['f1_'+task_cur] = 2 * record['case_recall_'+task_cur] * record['case_precision_'+task_cur] / (record['case_recall_'+task_cur] + record['case_precision_'+task_cur]) if (record['case_recall_'+task_cur] + record['case_precision_'+task_cur]) > 0 else 0
    case_res = "case_level:\nprecision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(record['case_precision_'+task_cur], record['case_recall_'+task_cur], record['f1_'+task_cur])
    
    print(case_res)
    print(char_res)
    return case_res, char_res

def main(args, 
         chosen_task=['QA', 'Data2txt', 'Summary'], 
         ):
    chosen_task.append('All')
    
    for i, sample in enumerate(data):
        response, labels, output, task =\
        sample['response'], sample['labels'], sample['output'], sample['task_type']
        
        res = cal_metrics(response, labels, output)
        temp_tp, temp_pred_len, temp_gold_len, pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt = res

        # calculate for difference tasks
        for task_cur in chosen_task:
            if task_cur == task or task_cur == 'All':
                record['match_size_'+task_cur]+= temp_tp
                record['prediction_'+task_cur]+= temp_pred_len
                record['groud_truth_'+task_cur]+= temp_gold_len
                
                record['pos_true_cnt_'+task_cur] += pos_true_cnt
                record['neg_true_cnt_'+task_cur] += neg_true_cnt
                record['neg_false_cnt_'+task_cur] += neg_false_cnt
                record['pos_false_cnt_'+task_cur] += pos_false_cnt
        
    for i, task_cur in enumerate(chosen_task):
        print_metrics(task_cur)
        
    print('chosen task: {}'.format(chosen_task))
    print('filepath: {}'.format(args.filepath))
    print('test_data_size: {}'.format(len(data)))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--chosen_task', type=str, default='QA,Data2txt,Summary')
    
    args = parser.parse_args()
    args.chosen_task = args.chosen_task.split(',')
    
    print('loading data from: {}'.format(args.filepath.split('/')[-1]))
    data = []
    with open(args.filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    record = collections.defaultdict(int)
    
    main(args, 
        chosen_task=args.chosen_task, 
        )