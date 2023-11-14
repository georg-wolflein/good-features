RENAME_MODELS = {
    "AttentionMIL": "AttMIL",
    "Transformer": "Transformer",
    "MeanAveragePooling": "Mean pool",
}
RENAME_FEATURE_EXTRACTORS = {
    "swin": "Swin",
    "ctranspath": "CTransPath",
    "vit": "ViT-B",
    "owkin": "Phikon",
    "vits": "ViT-S",
    "dino_p16": "Lunit-DINO",
    "resnet50": "ResNet-50",
    "retccl": "RetCCL",
    "bt": "Lunit-BT",
    "swav": "Lunit-SwAV",
}
RENAME_TARGETS = {
    "subtype": "Subtype",
    "CDH1": "CDH1",
    "TP53": "TP53",
    "PIK3CA": "PIK3CA",
    "lymph": "LN status",
    "MSI": "MSI",
    "KRAS": "KRAS",
    "BRAF": "BRAF",
    "SMAD4": "SMAD4",
}
RENAME_AUGMENTATIONS = {
    "rotate random angle": "random rotation",
}
RENAME_AUGMENTATION_GROUPS = {
    "none": "Original",
    "Macenko_patchwise": "Macenko (patch)",
    "Macenko_slidewise": "Macenko (slide)",
    "all": "All",
    "simple_rotate": "Rotate/flip",
}
FEATURE_EXTRACTOR_GROUPS = {
    "Swin": ["swin", "ctranspath"],
    "ViT-B": ["vit", "owkin"],
    "ViT-S": ["vits", "dino_p16"],
    "ResNet-50": ["resnet50", "retccl", "bt", "swav"],
}
TARGET_GROUPS = {
    "Breast": ["subtype", "CDH1", "TP53", "PIK3CA"],
    "Breast (CAM17)": ["lymph"],
    "Colorectal": ["MSI", "KRAS", "BRAF", "SMAD4"],
}
