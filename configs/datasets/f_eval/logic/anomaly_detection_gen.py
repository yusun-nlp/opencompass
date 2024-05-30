from opencompass.openicl import GenInferencer
from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets.f_eval.f_eval import AnomalyEvaluator

anomaly_detection_zh_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train")

anomaly_detection_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}’{pronoun}‘指A.{opt1},B.{opt2},请回答A或者B：",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=10))

anomaly_detection_en_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train",
    test_range="[:100]")

anomaly_detection_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt} 'What does '{pronoun}' refer to? A. {opt1}, B. {opt2}. Please answer A or B:",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=10))

anomaly_detection_eval_cfg = dict(evaluator=dict(type=AnomalyEvaluator), )

anomaly_detection_gen_datasets = [
    dict(
        abbr='anomaly_detection-zh',
        type=JsonlDataset,
        path='./data/f_eval/logic/anomaly_detection-zh.jsonl',
        reader_cfg=anomaly_detection_zh_reader_cfg,
        infer_cfg=anomaly_detection_zh_infer_cfg,
        eval_cfg=anomaly_detection_eval_cfg),
    dict(
        abbr='anomaly_detection-en',
        type=JsonlDataset,
        path='./data/f_eval/logic/anomaly_detection-en.jsonl',
        reader_cfg=anomaly_detection_en_reader_cfg,
        infer_cfg=anomaly_detection_en_infer_cfg,
        eval_cfg=anomaly_detection_eval_cfg)
]
