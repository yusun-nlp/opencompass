"""This file is used to evaluate reference-based subjective datasets in F-Eval,
which are evaluating by API models.

Commonsense Triple, TextbookQA, Instruction and Fallacy Attack.
"""
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import DLCRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.summarizers.subjective import FEvalSummarizer
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.models import OpenAI

with read_base():
    from ..datasets.f_eval.commonsense import commonsense_triple_datasets, textbookqa_datasets, instruction_datasets
    from ..datasets.f_eval.logic import fallacy_attack_datasets
    from .models import base_models, api_models
model_dataset_combinations = []
work_dir = './outputs/f_eval/'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

# The infer and eval tasks are based on alillm, while other runners can be used as well.
# The instruction of other runner like slurm can be found in doc of OpenCompass.

alillm2_cfg = dict(
    bashrc_path="",  # Please fill your bashrc path
    conda_env_name="",  # Please fill your conda env name
    dlc_config_path="",  # Please fill your dlc config path
    workspace_id="",  # Please fill your workspace id
    worker_image="",  # Please fill your worker image
)

infer = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=alillm2_cfg,
        retry=2,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask),
    ),
)

# Please fill your openai API key
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        retry=2,
        max_num_workers=2,
        aliyun_cfg=alillm2_cfg,
        task=dict(
            type=SubjectiveEvalTask,
            judge_cfg=dict(
                abbr='GPT4.0',
                type=OpenAI, path='gpt-4-1106-preview',
                key="",  # Please fill your openai API key
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=2048, max_seq_len=2048, batch_size=4
            )
        )
    ),
)

summarizer = dict(
    type=FEvalSummarizer,
    judge_model='GPT4.0'
)

# You can customize the evaluation model and the sub-datasets for evaluation.
# If use api models, you should fill your openai API key in api_models in configs/eval_f_eval/models.py
models = base_models + api_models
datasets = commonsense_triple_datasets + textbookqa_datasets + instruction_datasets + fallacy_attack_datasets
for d in datasets:
    d["infer_cfg"]["inferencer"]["save_every"] = 1

model_dataset_combinations.append(
    dict(models=models, datasets=datasets))

# beforing running this file, please ensure a backend server is running and the address is assigned to the enviroment called OPNECOMPASS_LQ_BACKEND
# For example, export OPENCOMPASS_LQ_BACKEND = 'http://127.0.0.1:5001/'
# python -u run.py configs/eval_f_eval/f_eval_api.py -s -r
