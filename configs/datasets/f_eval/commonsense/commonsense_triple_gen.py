from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import RandomRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.datasets.f_eval.f_eval import llm_eval_json_postprocess

commonsense_triple_reader_cfg = dict(
    input_columns=['start', 'relation', 'references'],
    output_column='ends')

commonsense_triple_infer_cfg_zh = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "实体：{start}，关系：{relation}",
                ),
                dict(
                    role="BOT",
                    prompt="和实体构成对应关系的词：{ends}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

commonsense_triple_infer_cfg_en = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "Entity: {start}. Relation: {relation}.",
                ),
                dict(
                    role="BOT",
                    prompt="Words that can form a corresponding relation with the entity: {ends}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

eval_prompt = "请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[事实与解释性回答]，因此你需要从下面的几个维度对回答进行评估: [事实正确性, 满足用户需求, 清晰度, 完备性]\n" + \
              "我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程： \n" + \
              "1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。\n" + \
              "2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。\n" + \
              "3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。\n" + \
              "4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。 其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。\n" + \
              "当模型回答的实体中存在与实体和关系有严重事实错误，或生成了有害内容时，总分必须是1到2分；\n" + \
              "当模型回答没有严重错误而且基本无害，但是存在低质量的实体时，总分为3到5分；\n" + \
              "当模型回答质量与参考答案相近，在所有维度上表现良好，总分得6到8分；\n" + \
              "只有当模型回答包含了全部的参考答案的实体，并且还有预测了更多符合事实的实体，才能得9到10分。\n" + \
              "作为示例，参考答案可以得到8分。" + \
              "请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：\n" + \
              "{“维度一”: 打分, “维度二”: 打分, ..., \"综合得分\": 打分}，例如：{\"事实正确性\": 9, \"满足用户需求\": 6, ..., \"综合得分\": 7}。"


commonsense_triple_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        prompt=eval_prompt
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="用户的提问：与实体{start}构成关系{relation}的实体有哪些？\n参考答案：{references}\n模型答案：{prediction}")])),
        postprocessor=dict(
            type=llm_eval_json_postprocess
        )
    ),
    pred_role="BOT",
)

commonsense_triple_datasets = [
    dict(
        type=JsonlDataset,
        abbr='commonsense_triple-zh',
        path='data/f_eval/commonsense/commonsense_triple-zh.jsonl',
        reader_cfg=commonsense_triple_reader_cfg,
        infer_cfg=commonsense_triple_infer_cfg_zh,
        eval_cfg=commonsense_triple_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='commonsense_triple-en',
        path='data/f_eval/commonsense/commonsense_triple-en.jsonl',
        reader_cfg=commonsense_triple_reader_cfg,
        infer_cfg=commonsense_triple_infer_cfg_en,
        eval_cfg=commonsense_triple_eval_cfg),
]
