from opencompass.datasets.f_eval.f_eval import ContradictionNLIEvaluator, ContradictionDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

contradiction_reader_cfg = dict(
    input_columns=["prompt"], output_column="reference"
)

contradiction_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=100),
)

contradiction_eval_cfg = dict(evaluator=dict(type=ContradictionNLIEvaluator))

contradiction_datasets = [
    dict(
        abbr="contradiction-en",
        type=ContradictionDataset,
        path='./data/f_eval/logic/contradiction-en.jsonl',
        reader_cfg=contradiction_reader_cfg,
        infer_cfg=contradiction_infer_cfg,
        eval_cfg=contradiction_eval_cfg,
    ),
    dict(
        abbr="contradiction-zh",
        type=ContradictionDataset,
        path='./data/f_eval/logic/contradiction-zh.jsonl',
        reader_cfg=contradiction_reader_cfg,
        infer_cfg=contradiction_infer_cfg,
        eval_cfg=contradiction_eval_cfg,
    )
]
