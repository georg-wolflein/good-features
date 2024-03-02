RENAME_MODELS = {
    "AttentionMIL": "AttMIL",
    "Transformer": "Transformer",
    "MeanAveragePooling": "Mean pool",
}
RENAME_FEATURE_EXTRACTORS = {
    "swin": "Swin",
    "ctranspath": "CTransPath",
    "vits": "ViT-S",
    "dino_p16": "Lunit-DINO",
    "vit": "ViT-B",
    "owkin": "Phikon-S",
    "owkin_teacher": "Phikon-T",
    "resnet50": "ResNet-50",
    "retccl": "RetCCL",
    "bt": "Lunit-BT",
    "swav": "Lunit-SwAV",
    "mocov2": "Lunit-MoCo",
}
FEATURE_EXTRACTOR_REFERENCES = {
    "swin": "liu2021swin",
    "ctranspath": "wang2022transformer",
    "vits": "kolesnikov2021image",
    "dino_p16": "kang2023benchmarking",
    "vit": "kolesnikov2021image",
    "owkin": "filiot2023scaling",
    "owkin_teacher": "filiot2023scaling",
    "resnet50": "he2015deep",
    "retccl": "wang2023retccl",
    "bt": "kang2023benchmarking",
    "swav": "kang2023benchmarking",
    "mocov2": "kang2023benchmarking",
}
RENAME_FEATURE_EXTRACTORS_WITH_REFERENCES = {
    k: v if k not in FEATURE_EXTRACTOR_REFERENCES else f"{v}~\\cite{{{FEATURE_EXTRACTOR_REFERENCES[k]}}}"
    for k, v in RENAME_FEATURE_EXTRACTORS.items()
}
RENAME_TARGETS = {
    "subtype": "\\breasticon-Subtype",
    "CDH1": "\\breasticon-CDH1",
    "TP53": "\\breasticon-TP53",
    "PIK3CA": "\\breasticon-PIK3CA",
    "lymph": "\\breasticon-LN status",
    "MSI": "\\colonicon-MSI",
    "KRAS": "\\colonicon-KRAS",
    "BRAF": "\\colonicon-BRAF",
    "SMAD4": "\\colonicon-SMAD4",
}
RENAME_AUGMENTATIONS = {"rotate random angle": "random rotation"}
RENAME_AUGMENTATION_GROUPS = {
    "none": "Original",
    "Macenko_patchwise": "Macenko (patch)",
    "Macenko_slidewise": "Macenko (slide)",
    "all": "All",
    "simple_rotate": "Rotate/flip",
}
RENAME_MAGNIFICATIONS = {
    "low": "8x",
    "high": "20x",
}
FEATURE_EXTRACTOR_GROUPS = {
    "Swin": ["swin", "ctranspath"],
    "ViT-S": ["vits", "dino_p16"],
    "ViT-B": ["vit", "owkin", "owkin_teacher"],
    "ResNet-50": ["resnet50", "retccl", "bt", "swav", "mocov2"],
}
TARGET_GROUPS = {
    "Breast": ["subtype", "CDH1", "TP53", "PIK3CA"],
    "Breast (CAM17)": ["lymph"],
    "Colorectal": ["MSI", "KRAS", "BRAF", "SMAD4"],
}
