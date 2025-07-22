from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.biodata import BiodataDataset
from opencompass.datasets.biodata import BiodataClsEvaluator, BiodataDictEvaluator, BiodataRMSEEvaluator, BiodataStringEvaluator, BiodataECNumberEvaluator

biodata_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth'
)

hint_dict = {
    'cls': 'Please use "yes" or "no" as your answer, and put your answer within \\boxed{}.',
    'number': 'Please put your answer number within \\boxed{}.',
    'dict': '',
    'Protein-string': 'Please put the final enzyme within \\boxed{} using "EC number", such as "EC1.2.3.4". Please split by "," if there are multiple enzymes.',
    'RNA-string': 'You should choose from [AtoI, m6A, none, m1A, ribozyme, m5C, m6Am, riboswitch, Um, Psi, m5U, 5S_rRNA, m7G, tRNA, miRNA, Intron_gpI, Cm, Intron_gpII, Am, 5_8S_rRNA, scaRNA, HACA-box, Gm, IRES, leader, CD-box]. Please put your final answer within \\boxed{}, split by "," if there are multiple answers.',
}

name_set = [
    'DNA-cls',
    'DNA-dict',
    'Multi_sequence-cls',
    'Multi_sequence-number',
    'Protein-cls',
    'Protein-number',
    'Protein-string',
    'RNA-number',
    'RNA-dict',
    'RNA-string',
]

biodata_datasets = []
for _name in name_set:
    task = _name.split('-')[-1]
    if _name in hint_dict:
        _hint = hint_dict[_name]
    else:
        _hint = hint_dict[task]
    biodata_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=f'{{prompt}}\n{_hint}'),
                dict(role='BOT', prompt='{ground_truth}\n')
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    if task == 'cls':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataClsEvaluator),
        )
    elif task == 'number':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataRMSEEvaluator),
        )
    elif task == 'dict':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataDictEvaluator),
        )
    elif _name == 'Protein-string':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataECNumberEvaluator),
        )
    elif _name == 'RNA-string':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataStringEvaluator),
        )
    else:
        raise NotImplementedError

    biodata_datasets.append(
        dict(
            abbr=f'{_name}-sample_2k',
            type=BiodataDataset,
            path='biodata/test-sample_2k',
            name=_name,
            reader_cfg=biodata_reader_cfg,
            infer_cfg=biodata_infer_cfg,
            eval_cfg=biodata_eval_cfg,
        )
    )

del _name
