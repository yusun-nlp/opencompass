from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

anomaly_detection_zh_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train")

anomaly_detection_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
                "{prompt}’{pronoun}‘指{opt1}。",  # noqa
            1:
                "{prompt}’{pronoun}‘指{opt2}。",  # noqa
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

anomaly_detection_en_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train",
    test_range="[:100]"
)

anomaly_detection_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
                "{prompt}'{pronoun}'refer to {opt1}.",  # noqa
            1:
                "{prompt}'{pronoun}'refer to {opt2}.",  # noqa
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

anomaly_detection_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

anomaly_detection_ppl_datasets = [
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
