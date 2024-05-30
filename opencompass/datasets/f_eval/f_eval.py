import json
import os.path as osp
import re
from typing import List

import requests
from datasets import Dataset, DatasetDict

from opencompass.datasets import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import general_postprocess

# The url of the backend.
_url = 'http://172.31.13.71:5001'


def first_number_postprocess_plus(text: str) -> float:
    """Return the first number in a string."""

    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    if match:
        score = float(match.group(1))
        if score > 100 or score < 0:
            return 0
        else:
            return float(match.group(1))
    else:
        return 0


def first_option_postprocess_with_gold(text: str,
                                       options: str,
                                       gold_options: dict = None) -> str:
    """Find first valid option for text.

    The gold options can be specified.
    """

    patterns = [
        f'[Tt]he answer is [{options}]',
        f'[Tt]he correct answer is [{options}]',
        f'答案是(.*?)[{options}]',
        f'答案为(.*?)[{options}]',
        f'固选(.*?)[{options}]',
        f'答案应该是(.*?)[{options}]',
        f'(\s|^)[{options}][\s。，,\.$]',  # noqa
        f'[{options}]',
    ]
    if gold_options is None:
        gold_options = {i: i for i in options}
    assert len(gold_options) == len(gold_options)

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return gold_options[i]
    return ''


def llm_eval_json_postprocess(text: str) -> float:
    pattern = re.compile(r'\{.*?"综合得分":.*?\}', re.DOTALL)
    json_res = pattern.findall(text)
    if len(json_res) == 1:
        json_res = json_res[0].replace(' ', '').replace('\n', '')
        try:
            res = eval(json_res)
            print('json', res['综合得分'])
            return res['综合得分']
        except:  # noqa: E722
            print('json error')
            return 0
    if '综合得分：' in text:
        scores = text.split('综合得分：')
        for idx in range(1, len(scores), 2):
            score = scores[idx].strip()
            if len(score) > 0 and score[0].isdigit():
                print('text', score[0])
                return int(score[0])
    if '综合得分:' in text:
        scores = text.split('综合得分:')
        for idx in range(1, len(scores), 2):
            score = scores[idx].strip()
            if len(score) > 0 and score[0].isdigit():
                print('text', score[0])
                return int(score[0])
    return 0


@ICL_EVALUATORS.register_module()
class LanguageExpressionEvaluator(BaseEvaluator):
    """Connecting backend for language expression evaluation."""

    def __init__(self, url=None) -> None:
        super().__init__()
        if url is None:
            url = _url
        self.url = url
        import os
        if 'OPENCOMPASS_LQ_BACKEND' in os.environ:
            self.url = os.environ['OPENCOMPASS_LQ_BACKEND']

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if len(references) > 0:
            headers = {'Accept': 'application/json'}
            r = requests.post(self.url,
                              headers=headers,
                              json={
                                  'predictions': predictions,
                                  'references': references
                              })
            ret = r.json()
            return {'score': ret['score'], 'details': ret['details']}
        else:
            return {'score': None}


def icl_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    text = text.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
    return ret1


class COTEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if str(j) in i:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


@LOAD_DATASET.register_module()
class ICLDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            filename = osp.join(path, f'icl-{split}-{name}.jsonl')
            with open(filename, 'r', encoding='utf-8') as f:
                raw_data = []
                for line in f:
                    row = json.loads(line)
                    assert 'question' in row and 'answer' in row
                    question = row['question']
                    answers = row['answer']
                    if split == 'dev':
                        answers = answers[0]
                    raw_data.append({'question': question, 'answer': answers})
                dataset[split] = Dataset.from_list(raw_data)

        return dataset


@ICL_EVALUATORS.register_module()
class AnomalyEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            if 'A' in prediction and 'B' in prediction:
                prediction = prediction.split('\n\n')[-1]
            if 'A' in prediction:
                processed_predictions.append(0)
            elif 'B' in prediction:
                processed_predictions.append(1)
            else:
                processed_predictions.append(-1)

        details = []
        cnt = 0
        for pred, cand_ans in zip(processed_predictions, references):
            detail = {'pred': pred, 'answer': cand_ans, 'correct': False}
            cnt += int(pred == cand_ans)
            if int(pred == cand_ans):
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100

        return {'accuracy': score, 'details': details}


@ICL_EVALUATORS.register_module()
class ICLEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split('\n')[0].lower()
            if 'answer is' in prediction:
                prediction = prediction.split('answer is')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        details = []
        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            detail = {'pred': pred, 'answer': cand_ans, 'correct': False}
            cnt += int(any([cand == pred for cand in cand_ans]))
            if int(any([cand == pred for cand in cand_ans])):
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class ICL_ZHEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split('\n')[0].lower()
            if '答案是' in prediction:
                prediction = prediction.split('答案是')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        details = []
        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            detail = {'pred': pred, 'answer': cand_ans, 'correct': False}
            cnt += int(any([cand == pred for cand in cand_ans]))
            if int(any([cand == pred for cand in cand_ans])):
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}


def fallacy_attack_postprocess(text: str) -> float:
    import json
    start = text.find('{')
    end = text.find('}') + 1

    if start == -1 or end == 0:
        return 0
    else:
        dict_str = text[start:end].replace("'", '"')
        try:
            dict_obj = json.loads(dict_str)
        except json.JSONDecodeError:
            return 0
        score = dict_obj.get('事实正确性',
                             "Key '事实正确性' not found in the dictionary.")
        if isinstance(score, str):
            score = 0
        return score


def coreference_postprocess(text: str) -> str:
    process_text = text.split('\n')[0]
    if ')' in process_text or '）' in process_text:
        process_text = re.split('[)）]', text)[0]
        return process_text
    else:
        return text


class CREvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            count += 1
            detail = {'pred': i, 'answer': j, 'correct': False}
            if len(i) > 10:
                prefix = '指的是'
                pred = i.strip()
                gold = j.strip()
                if len(pred) < len(gold):
                    continue
                if pred[-1] == '。':
                    pred = pred[:-1]
                if pred[-len(gold):] == gold:
                    correct += 1
                    detail['correct'] = True
                elif pred[:len(gold)] == gold:
                    correct += 1
                    detail['correct'] = True
                elif prefix in pred:
                    pred = pred[pred.rfind(prefix) + len(prefix):]
                    if pred.startswith(gold) or pred.startswith('“' + gold):
                        correct += 1
                        detail['correct'] = True
                else:
                    if gold in pred[max(len(pred) - len(gold) - 2, 0):]:
                        correct += 1
                        detail['correct'] = True
                detail['pred'] = pred
            else:
                if i == j:
                    correct += 1
                    detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


class ContradictionNLIEvaluator(BaseEvaluator):
    """Connecting backend for contradiction evaluation."""

    def __init__(self, url=None):
        super().__init__()
        if url is None:
            url = _url
        self.url = url
        import os
        if 'OPENCOMPASS_LQ_BACKEND' in os.environ:
            self.url = os.environ['OPENCOMPASS_LQ_BACKEND']

    def score(self, predictions: List, references: List) -> dict:

        def detect_main_language(text):
            chinese_characters = len(
                [c for c in text if '\u4e00' <= c <= '\u9fff'])
            english_characters = len(
                [c for c in text if 'a' <= c.lower() <= 'z'])

            if chinese_characters > english_characters:
                return 'zh'
            else:
                return 'en'

        data = {
            'predictions': predictions,
            'references': references,
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.url + '/subject/contradiction',
                                 data=json.dumps(data),
                                 headers=headers)
        result = eval(response.text)
        return result


@ICL_EVALUATORS.register_module()
class EmotionClsEvaluator(BaseEvaluator):
    """Connecting backend for emotion evaluation."""

    def __init__(self, url=None):
        super().__init__()
        if url is None:
            url = _url
        self.url = url
        import os
        if 'OPENCOMPASS_LQ_BACKEND' in os.environ:
            self.url = os.environ['OPENCOMPASS_LQ_BACKEND']

    def score(self, predictions: List, references: List) -> dict:
        data = {'predictions': predictions, 'references': references}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.url + '/subject/emotion',
                                 data=json.dumps(data),
                                 headers=headers)
        result = eval(response.text)
        return result


@LOAD_DATASET.register_module()
class ContradictionDataset(BaseDataset):

    @staticmethod
    def load(path):
        data = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        def preprocess(example):
            example['reference'] = example['prompt']
            return example

        dataset = Dataset.from_list(data)
        dataset = dataset.map(preprocess)
        return dataset
