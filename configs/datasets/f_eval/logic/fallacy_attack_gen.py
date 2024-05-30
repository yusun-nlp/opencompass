from opencompass.datasets import JsonlDataset

from opencompass.datasets.f_eval.f_eval import fallacy_attack_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator

prompt_zh = """
我们会给出一个模型的输出文本，以及一个符合常识的参考。
请你以公正的评判者的身份，评估模型输出的文本是否符合常识。需要从下面的几个维度对> 文本进行评估:
1. 事实正确性\n你需要按照遵守以下的流程：
1. 从不同维度对模型输出进行评价，并给每一个维度一个1～10的分数。
3.你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。


当模型回答与给出的参考答案完全相悖时，总> 分必须是1到2分；
当模型回答与给出的参考答案无关并且不太符合常识时，总分必须是3到6分；
当模型回答与给出的参考答案基本相同以及与生活常识基> 本相同时，总分必须是7到10分。
之后，在你回答的末尾，按照以下json格式（包括括号）返回你所有的打分结果: {'维度一': 打分, '维度二': 打分}
例如：{'事实正确性': 9}。
记住，你的重点是看模型输出的最终意思，如果先错后对以最后的表态为准。
（假设模型的输出有多种含义，可能会前后矛盾，请选择你最后的含义进行> 评估打分，只要最终含义符合逻辑、前面错了也不减分）
下面我会给一个dict格式的json，包含参考答案和模型输出：
"""

fallacy_attack_reader_cfg = dict(
    input_columns=["prompt", "annotation", "answer"], output_column="label",
)

fallacy_attack_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=25),
)

fallacy_attack_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        fallacy_role="HUMAN",
                        prompt=prompt_zh
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="{'参考': '{answer}','模型输出': '{annotation}{prediction}'}")])),
        postprocessor=dict(
            type=fallacy_attack_postprocess
        )
    ),
    pred_role="BOT",
)

fallacy_attack_datasets = [
    dict(
        abbr="fallacy_attack-zh",
        type=JsonlDataset,
        path='./data/f_eval/logic/fallacy_attack-zh.jsonl',
        reader_cfg=fallacy_attack_reader_cfg,
        infer_cfg=fallacy_attack_infer_cfg,
        eval_cfg=fallacy_attack_eval_cfg,
    ),
    dict(
        abbr="fallacy_attack-en",
        type=JsonlDataset,
        path='./data/f_eval/logic/fallacy_attack-en.jsonl',
        reader_cfg=fallacy_attack_reader_cfg,
        infer_cfg=fallacy_attack_infer_cfg,
        eval_cfg=fallacy_attack_eval_cfg,
    )
]
