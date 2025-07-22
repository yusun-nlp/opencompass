import ast
import json
import os
import re

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BiodataDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        final_path = os.path.join(path, f'{name}.jsonl')
        with open(final_path, 'r') as f:
            data = [json.loads(line) for line in f]
        if '-dict' in name:
            new_data = []
            for ins in data:
                new_ins = ins.copy()
                new_ins['prompt'] = (
                    ins['prompt'] +
                    'Please put your final answer with \\boxed{}' +
                    ' in json format, such as {')
                gold_keys = list(ins['ground_truth'].keys())
                for key in gold_keys:
                    new_ins['prompt'] += f"\"{key}\": xx, "
                new_ins['prompt'] = new_ins['prompt'][:-2] + '}'
                new_data.append(new_ins)
        else:
            new_data = data
        dataset = Dataset.from_list(new_data)
        return dataset


def extract_boxed_text(text):
    # 提取boxed中的内容 - 修正正则表达式以正确匹配嵌套结构
    pattern = re.compile(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None

    # 取第一个匹配的内容
    boxed_content = matches[-1].strip()

    # 只有当存在完整的\text{...}时才去掉包装，否则保持原样
    # 使用更严格的正则表达式，确保\text{...}是完整的
    clean_content = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_content)

    # 去掉LaTeX转义符
    # 处理常见的LaTeX转义字符
    clean_content = re.sub(r'\\(.)', r'\1', clean_content)

    return clean_content


@ICL_EVALUATORS.register_module()
class BiodataClsEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        ans_dict = {
            'positive': 'yes',
            'negative': 'no',
        }

        correct = []
        details = []
        for pred, ans in zip(predictions, references):
            pred = extract_boxed_text(pred)
            if not pred:
                pred = None
            else:
                pred = pred.lower()
            if ans in ans_dict:
                ans = ans_dict[ans]
            if pred in ans_dict:
                pred = ans_dict[pred]
            detail = {'pred': pred, 'answer': ans}
            detail['score'] = 100 if ans in pred else 0
            details.append(detail)
            correct.append(detail['score'])

        score = sum(correct) / len(correct) if correct else 0.0

        return {'score': score, 'details': details}


def extract_number(text):
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    if not matches:
        return None
    return [float(match) for match in matches][-1]


@ICL_EVALUATORS.register_module()
class BiodataRMSEEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        avg_score = 0
        details = []
        for prediction, reference in zip(predictions, references):
            pred = extract_number(prediction)
            ans = reference
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                pred = 0
            rmse_score = np.sqrt(np.mean((np.array(pred) - np.array(ans))**2))
            detail['score'] = rmse_score
            avg_score += rmse_score
            details.append(detail)

        score = avg_score / len(predictions)

        return {'score': score, 'details': details}


def extract_dict_text(text):
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return [match for match in matches][-1]


@ICL_EVALUATORS.register_module()
class BiodataDictEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        correct = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_dict_text(pred)
            if pred:
                try:
                    pred = json.loads(pred)
                except Exception:
                    try:
                        pred = ast.literal_eval(pred)
                    except Exception:
                        pred = None
            detail = {'pred': pred, 'answer': ans}
            if not pred or not isinstance(pred,
                                          dict) or pred.keys() != ans.keys():
                detail['score'] = 10
                details.append(detail)
                correct.append(detail['score'])
                continue
            cur_score = []
            for key in pred.keys():
                try:
                    pred_num = float(pred[key])
                except Exception:
                    pred_num = 0
                ans_num = float(ans[key])
                rmse_score = np.sqrt(
                    np.mean((np.array(pred_num) - np.array(ans_num))**2))
                cur_score.append(rmse_score)
            detail['score'] = sum(cur_score) / len(cur_score)
            details.append(detail)
            correct.append(detail['score'])

        score = sum(correct) / len(correct) if correct else 0.0

        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataStringEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        all_f1 = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                detail['f1'] = 0
                all_f1.append(0)
                continue
            pred_set = set(p.lower().strip() for p in pred.split(',')
                           if p.strip())
            ans_set = set(a.lower().strip() for a in ans.split(',')
                          if a.strip())

            # 计算交集、并集
            intersection = pred_set & ans_set
            # 计算精确率、召回率
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(ans_set) if ans_set else 0
            # 计算F1 score
            f1 = 2 * precision * recall / (precision + recall) if (
                precision + recall) > 0 else 0

            detail['f1'] = f1 * 100
            details.append(detail)
            all_f1.append(detail['f1'])

        final_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        return {'f1': final_f1, 'details': details}


def dedup_ec_codes(ec_numer_list):
    """
    删除易被更泛化 EC 号覆盖的、更具体的 EC 号。

    规则示例：
        EC3.5.4.9  与 EC3.5.4.- 同时出现 → 去掉 EC3.5.4.9
        EC3.5.4.- 与 EC3.5.-.- 同时出现 → 去掉 EC3.5.4.-

    参数
    ----
    codes : List[str]
        原始 EC 号列表，元素格式须满足 ECa.b.c.d，其中 a–d 可以是数字或 '-'

    返回
    ----
    List[str]
        去重后的 EC 号列表，保持原有顺序
    """
    EC_PATTERN = re.compile(r'^ec(\d+|-)\.(\d+|-)\.(\d+|-)\.(\d+|-)$')
    # 先做一次规范化，保留顺序
    normalized = [c.strip() for c in ec_numer_list]
    remaining = set(normalized)  # 用集合便于快速查询

    for code in normalized:
        if code not in remaining:  # 可能在之前的循环里被删掉
            continue

        m = EC_PATTERN.match(code)
        if not m:
            # 不是合法 EC 格式，保留原状
            continue

        parts = list(
            m.groups())  # ['3', '5', '4', '9']  或 ['3', '5', '4', '-']
        # 依次生成更泛化的版本：EC3.5.4.-, EC3.5.-.-, EC3.-.-.-（不含自身）
        for i in range(3, 0, -1):
            generalized = parts[:i] + ['-'] * (4 - i)
            gen_code = 'ec' + '.'.join(generalized)
            if gen_code in remaining and gen_code != code:
                # 如果集合里已存在更泛化的版本，删除当前更具体的
                remaining.discard(code)
                break

    # 按原顺序返回仍保留的条目
    return [c for c in normalized if c in remaining]


@ICL_EVALUATORS.register_module()
class BiodataECNumberEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        all_f1 = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                detail['f1'] = 0
                all_f1.append(0)
                continue
            pred_set = set(p.lower().strip() for p in pred.split(',')
                           if p.strip())
            ans_set = set(a.lower().strip() for a in ans.split(',')
                          if a.strip())
            # 计算并集
            intersection = pred_set | ans_set
            intersect_list = dedup_ec_codes(list(intersection))
            score = 1
            for intersect in intersect_list:
                if intersect not in ans_set:
                    score = 0
                    break
            detail['score'] = score
            details.append(detail)
            all_f1.append(detail['score'])

        final_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        return {'score': final_f1 * 100, 'details': details}
