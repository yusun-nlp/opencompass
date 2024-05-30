from mmengine.config import read_base

with read_base():
    from .commonsense.commonsenseqa_gen import commonsenseqa_gen_datasets
    from .commonsense.commonsenseqa_ppl import commonsenseqa_ppl_datasets
    from .commonsense.commonsense_triple_gen import commonsense_triple_datasets
    from .commonsense.instruction_gen import instruction_datasets
    from .commonsense.textbookqa_gen import textbookqa_datasets
    from .commonsense.story_gen import story_gen_datasets
    from .commonsense.story_ppl import story_ppl_datasets

shared = commonsense_triple_datasets + instruction_datasets + textbookqa_datasets
commonsense_base = commonsenseqa_ppl_datasets + story_ppl_datasets
commonsense_api = commonsenseqa_gen_datasets + story_gen_datasets
