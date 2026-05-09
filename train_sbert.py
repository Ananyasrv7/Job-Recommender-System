import os
import torch
import random
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# FIX: Disable MPS (Apple GPU) — forces stable CPU training 
# Your crash: "MPS backend out of memory" — MPS can't handle this job size.
# These 3 lines disable it completely before anything else loads.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(torch.backends, "mps"):
    torch.backends.mps.enabled = False

DEVICE = "cpu"
print(f"Device: {DEVICE}  (MPS disabled — prevents out-of-memory crash)")

# Load ESCO data
base = "ESCO dataset - v1.1.1 - classification - en - csv"

occupations = pd.read_csv(f"{base}/occupations_en.csv", low_memory=False)
relations   = pd.read_csv(f"{base}/occupationSkillRelations_en.csv", low_memory=False)
skills_df   = pd.read_csv(f"{base}/skills_en.csv", low_memory=False)

occupations = occupations.dropna(subset=["preferredLabel", "description"])
occupations = occupationai s[occupations["description"].str.strip() != ""]
print(f"Occupations loaded: {len(occupations)}")

# Build training pairs
# Use (job_title, job_description) — NOT (text, text) which was the original bug.
# MNRL treats every other item in the batch as a negative automatically.
train_examples = []

for _, row in occupations.iterrows():
    title = str(row["preferredLabel"]).strip()
    desc  = str(row["description"]).strip()
    if not title or not desc:
        continue
    train_examples.append(InputExample(texts=[title, desc]))
    train_examples.append(InputExample(texts=[desc, title]))  # symmetric

# Add job → skill pairs for domain knowledge
try:
    merged = relations.merge(
        occupations[["conceptUri", "preferredLabel"]],
        left_on="occupationUri", right_on="conceptUri"
    ).merge(
        skills_df[["conceptUri", "preferredLabel"]],
        left_on="skillUri", right_on="conceptUri",
        suffixes=("_occ", "_skill")
    )
    for _, row in merged.iterrows():
        job   = str(row["preferredLabel_occ"]).strip()
        skill = str(row["preferredLabel_skill"]).strip()
        if job and skill:
            train_examples.append(
                InputExample(texts=[job, f"requires skill {skill}"])
            )
    print(f"Job-skill pairs added: {len(merged)}")
except Exception as e:
    print(f"Skipping skill pairs: {e}")

random.shuffle(train_examples)

# Train/val split 
split      = int(0.9 * len(train_examples))
train_data = train_examples[:split]
val_data   = train_examples[split:]
print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

# Model — pinned to CPU 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# DataLoader 
# FIX: batch_size reduced to 16 (safe for CPU RAM on a MacBook)
# pin_memory=False — MPS/CPU does not support pin_memory, avoids the UserWarning
train_dataloader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=16,
    pin_memory=False,   # fixes the UserWarning you saw
)

# Loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Evaluator 
val_s1 = [ex.texts[0] for ex in val_data[:1000]]
val_s2 = [ex.texts[1] for ex in val_data[:1000]]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    val_s1, val_s2,
    scores=[1.0] * len(val_s1),
    name="esco-val",
)

# Training
EPOCHS       = 4
total_steps  = len(train_dataloader) * EPOCHS
warmup_steps = int(0.10 * total_steps)

print(f"\nStarting: {EPOCHS} epochs | {total_steps} steps | {warmup_steps} warmup")
print("Expected time on MacBook CPU: ~30-90 mins depending on dataset size\n")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    evaluation_steps=500,
    output_path="esco-sbert",
    save_best_model=True,
    show_progress_bar=True,
)

print("\nDone. Model saved to: esco-sbert/")