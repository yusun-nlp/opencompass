"""This file is used to evaluate the remaining datasets in F-Eval, excluding
the reference-based subjective dataset."""
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.summarizers.subjective import FEvalSummarizer

with read_base():
    from ..datasets.f_eval.expression import expression_base, expression_api
    from ..datasets.f_eval.commonsense import commonsense_base, commonsense_api, commonsenseqa_ppl_datasets
    from ..datasets.f_eval.logic import logic_base, logic_api
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

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLEvalTask, dump_details=True)),
)

summarizer = dict(
    type=FEvalSummarizer,
    judge_model='GPT4.0'
)

# You can customize the evaluation model and the sub-datasets for evaluation.
# Considering that the inference methods of API models and base models are not exactly the same, models and datasets need to correspond.
# If use api models, you should fill your openai API key in api_models in configs/eval_f_eval/models.py
models = base_models[:1]
# models = api_models
datasets = commonsenseqa_ppl_datasets
# datasets = expression_base + commonsense_base + logic_base
# datasets = expression_api + commonsense_api + logic_api
for d in datasets:
    d["infer_cfg"]["inferencer"]["save_every"] = 1

model_dataset_combinations.append(
    dict(models=models, datasets=datasets))

# beforing running this file, please ensure a backend server is running and the address is assigned to the enviroment called OPNECOMPASS_LQ_BACKEND
# For example, export OPENCOMPASS_LQ_BACKEND = 'http://127.0.0.1:5001/'
# python -u run.py configs/eval_f_eval/f_eval_other.py -s -r
