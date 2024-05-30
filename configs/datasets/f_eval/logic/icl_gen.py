from opencompass.datasets.f_eval.f_eval import ICLEvaluator, ICLDataset, ICL_ZHEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

icl_zh_datasets = []
for k in [0, 4]:
    icl_zh_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='dev')

    if k == 0:
        icl_zh_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN',
                             prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答：'),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        icl_zh_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN',
                             prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答: 答案是 {answer}。\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(role='HUMAN',
                             prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答：'),
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

    icl_zh_eval_cfg = dict(evaluator=dict(type=ICL_ZHEvaluator), pred_role="BOT")

    icl_zh_datasets.append(
        dict(
            type=ICLDataset,
            abbr='icl-zh' if k == 0 else f'icl-zh_{k}shot',
            path='./data/f_eval/logic/',
            name="zh",
            reader_cfg=icl_zh_reader_cfg,
            infer_cfg=icl_zh_infer_cfg,
            eval_cfg=icl_zh_eval_cfg)
    )

icl_en_datasets = []
icl_en_datasets_select = []
for k in [0, 4]:
    icl_en_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='dev')

    if k == 0:
        icl_en_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN',
                             prompt='Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        icl_en_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN',
                             prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A: The answer is {answer}.\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(role='HUMAN',
                             prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

    icl_en_eval_cfg = dict(evaluator=dict(type=ICLEvaluator), pred_role="BOT")

    icl_en_datasets.append(
        dict(
            type=ICLDataset,
            abbr='icl-en' if k == 0 else f'icl-en_{k}shot',
            path='./data/f_eval/logic/',
            name="en",
            reader_cfg=icl_en_reader_cfg,
            infer_cfg=icl_en_infer_cfg,
            eval_cfg=icl_en_eval_cfg)
    )
