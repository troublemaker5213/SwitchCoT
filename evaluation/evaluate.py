import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def compute_group_stats(samples, group_key):
    scores = {}
    token_counts = {}
    for sample in samples:
        key = sample.get(group_key)
        if key is None:
            continue
        if key not in scores:
            scores[key] = []
            token_counts[key] = []
        scores[key].append(sample['score'])
        for i in sample['token_counts']:
            token_counts[key].append(i)
    # 处理均值和排序
    scores = {k: np.round(np.mean(v) * 100, 1) for k, v in scores.items()}
    token_counts = {k: np.round(np.mean(v), 1) for k, v in token_counts.items()}
    scores = dict(sorted(scores.items()))
    token_counts = dict(sorted(token_counts.items()))
    return scores, token_counts



def evaluate(data_name, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))
    # print("="*100)
    # print(score_mat)
    # print(mean_score)
    # print("="*100)

    sum_token_counts=0
    correct_num=0
    for s in samples:
        for i in range(len(s['token_counts'])):
            if s['score'][i]==True:
                correct_num+=1
                sum_token_counts+=int(s['token_counts'][i])


    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc(pass@1)": mean_score[0],
        "acc": sum(i for i in mean_score) / len(mean_score),
        "token_counts(right)": sum_token_counts / correct_num if correct_num != 0 else 0,
        "token_counts(all)": sum(sum(num for num in s['token_counts']) for s in samples) / len(scores),
    }



    for key in ['subject', 'category', 'difficulty','level','type']:
        if key in samples[0]:
            acc, tokens = compute_group_stats(samples, key)
            result_json[f"{key}_acc"] = acc
            result_json[f"{key}_token_counts"] = tokens


    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
