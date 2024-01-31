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
    "owkin_teacher": "Phikon (teacher)",
    "vits": "ViT-S",
    "dino_p16": "Lunit-DINO",
    "resnet50": "ResNet-50",
    "retccl": "RetCCL",
    "bt": "Lunit-BT",
    "swav": "Lunit-SwAV",
    "mocov2": "Lunit-MoCo",
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
    "ViT-B": ["vit", "owkin", "owkin_teacher"],
    "ViT-S": ["vits", "dino_p16"],
    "ResNet-50": ["resnet50", "retccl", "bt", "swav", "mocov2"],
}
TARGET_GROUPS = {
    "Breast": ["subtype", "CDH1", "TP53", "PIK3CA"],
    "Breast (CAM17)": ["lymph"],
    "Colorectal": ["MSI", "KRAS", "BRAF", "SMAD4"],
}
