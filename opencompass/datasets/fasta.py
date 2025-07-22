import json
import os
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class FASTADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        final_path = os.path.join(path, f'{name}.jsonl')
        with open(final_path, 'r') as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)
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
class FASTAEvaluator(BaseEvaluator):
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
        all_precision = []
        all_recall = []
        details = []
        for pred, ans in zip(predictions, references):
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                detail['f1'] = 0
                detail['precision'] = 0
                detail['recall'] = 0
                all_f1.append(0)
                all_precision.append(0)
                all_recall.append(0)
                continue
            pred_set = set(p.lower().strip() for p in pred.split(';')
                           if p.strip())
            ans_set = set(a.lower().strip() for a in ans.split(';')
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
            detail['precision'] = precision * 100
            detail['recall'] = recall * 100
            details.append(detail)
            all_f1.append(detail['f1'])
            all_precision.append(detail['precision'])
            all_recall.append(detail['recall'])

        final_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        final_precision = sum(all_precision) / len(
            all_precision) if all_precision else 0.0
        final_recall = sum(all_recall) / len(all_recall) if all_recall else 0.0

        return {
            'f1': final_f1,
            'precision': final_precision,
            'recall': final_recall,
            'details': details
        }
