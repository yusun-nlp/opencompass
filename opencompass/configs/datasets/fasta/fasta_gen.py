from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.fasta import FASTADataset, FASTAEvaluator

fasta_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth'
)

hint_dict = {
    'EC_number_prediction': 'Please output the EC number of the enzyme in \\boxed{x.x.x.x}.',
    'go_terms_prediction': 'Please output the Gene Ontology terms of the protein in \\boxed{}, split by ";" if there are multiple terms.',
    'keywords_prediction': 'Please output the keywords of the protein in \\boxed{}, split by ";" if there are multiple keywords.',
    'subcellular_loc_prediction': 'You should choose from [Nucleus, Mitochondrion, Lysosome/Vacuole, Extracellular, Peroxisome, Cytoplasm, Plastid, reticulum, membrane, apparatus]. Please output the subcellular location of the protein in \\boxed{}, split by ";" if there are multiple locations.'
}

name_set = [
    'EC_number_prediction',
    'go_terms_prediction',
    'keywords_prediction',
    'subcellular_loc_prediction',
]

fasta_datasets = []
for _name in name_set:
    _hint = hint_dict[_name]
    fasta_infer_cfg = dict(
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
    fasta_eval_cfg = dict(
        evaluator=dict(type=FASTAEvaluator),
    )

    fasta_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FASTADataset,
            path='fasta/test',
            name=_name,
            reader_cfg=fasta_reader_cfg,
            infer_cfg=fasta_infer_cfg,
            eval_cfg=fasta_eval_cfg,
        ))

del _name
