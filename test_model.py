
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# DistilBERT pretrained model
MODEL_NAME = "distilbert/distilbert-base-uncased"

print("Step 1: Loading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Step 2: Loading pretrained DistilBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={
        0: "claim",
        1: "evidence",
        2: "other",
    },
    label2id={
        "claim": 0,
        "evidence": 1,
        "other": 2,
    },
)

print("Step 3: DistilBERT loaded successfully!")
print("Model type:", model.config.model_type)
print("Number of labels:", model.config.num_labels)
print("Labels:", model.config.id2label)
