from mmengine.config import read_base

with read_base():
    from .logic.contradiction_gen import contradiction_datasets
    from .logic.coreference_gen import coreference_datasets
    from .logic.cot_gen import cot_datasets
    from .logic.fallacy_attack_gen import fallacy_attack_datasets
    from .logic.icl_gen import icl_zh_datasets, icl_en_datasets
    from .logic.anomaly_detection_gen import anomaly_detection_gen_datasets
    from .logic.anomaly_detection_ppl import anomaly_detection_ppl_datasets

shared = contradiction_datasets + coreference_datasets + cot_datasets + icl_zh_datasets + icl_en_datasets

logic_base = shared + anomaly_detection_ppl_datasets
logic_api = shared + anomaly_detection_gen_datasets
