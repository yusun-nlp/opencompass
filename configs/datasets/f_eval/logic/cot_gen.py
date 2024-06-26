from opencompass.datasets import JsonlDataset
from opencompass.datasets.f_eval.f_eval import COTEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

cot_reader_cfg = dict(
    input_columns=["prompt", 'cot'], output_column="answer"
)

cot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}\n{cot}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=100),
)

cot_eval_cfg = dict(evaluator=dict(type=COTEvaluator))

cot_datasets = [
    dict(
        abbr="cot-en",
        type=JsonlDataset,
        path='./data/f_eval/logic/cot-en.jsonl',
        reader_cfg=cot_reader_cfg,
        infer_cfg=cot_infer_cfg,
        eval_cfg=cot_eval_cfg,
    ),
    dict(
        abbr="cot-zh",
        type=JsonlDataset,
        path='./data/f_eval/logic/cot-zh.jsonl',
        reader_cfg=cot_reader_cfg,
        infer_cfg=cot_infer_cfg,
        eval_cfg=cot_eval_cfg,
    )
]
