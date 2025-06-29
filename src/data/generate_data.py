import random
from Bio import Align
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
import shutil

# --- CONFIGURATION ---
SEQ_LENGTH = 128
NUM_PAIRS = 1200
MUTATION_RATES = [0.2, 0.3, 0.6, 0.8, 0.9, 1.0]
TRAIN_RATIO = 0.67
VAL_RATIO = 0.17
TEST_RATIO = 0.16
OUTPUT_DIR = "data_dotplots"
PLOTS_DIR = "plots"
METADATA_DIR = "metadata"

# --- REPRODUCIBILITY ---
RANDOM_SEED = 27
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- SETUP ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, PLOTS_DIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, METADATA_DIR), exist_ok=True)
aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.1

def generate_random_sequence(length):
    return ''.join(random.choices("ACGT", k=length))

def mutate_sequence(seq, mutation_rate):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < mutation_rate:
            mutation_type = random.choice(['sub', 'del', 'ins'])
            if mutation_type == 'sub':
                seq[i] = random.choice([x for x in 'ACGT' if x != seq[i]])
            elif mutation_type == 'del' and i < len(seq) - 1:
                seq[i] = ''
            elif mutation_type == 'ins':
                seq[i] += random.choice('ACGT')
    return ''.join(seq)[:SEQ_LENGTH]
     
def compute_alignment_score(s1, s2):
    return aligner.score(s1, s2)

def compute_dotplot(s1, s2, k=3):
    dotplot = np.zeros((len(s1) - k + 1, len(s2) - k + 1), dtype=np.uint8)
    for i in range(len(s1) - k + 1):
        kmer1 = s1[i:i+k]
        for j in range(len(s2) - k + 1):
            kmer2 = s2[j:j+k]
            if kmer1 == kmer2:
                dotplot[i, j] = 255  # white pixel
    return dotplot

def save_dotplot_image(matrix, filename):
    img = Image.fromarray(matrix)
    img = img.convert('L')
    img.save(filename)

# --- GENERATE DATA ---
data = []
for _ in range(NUM_PAIRS):
    base_seq = generate_random_sequence(SEQ_LENGTH)
    mutation_rate = random.choice(MUTATION_RATES)
    mutated_seq = mutate_sequence(base_seq, mutation_rate)
    score = compute_alignment_score(base_seq, mutated_seq)
    dot = compute_dotplot(base_seq, mutated_seq)
    
    pair_id = f"{len(data):04d}"
    dotplot_path = os.path.join(OUTPUT_DIR, PLOTS_DIR,  f"dot_{pair_id}.png")
    save_dotplot_image(dot, dotplot_path)
    
    data.append({
        "id": pair_id,
        "seq1": base_seq,
        "seq2": mutated_seq,
        "score": score,
        "mutation_rate": mutation_rate,
        "dotplot": dotplot_path
    })


print(f"Generated {len(data)} sequence pairs with dotplots.")
print(data[0])  # Print first data entry for verification

# --- SPLIT DATA ---
train_val, test = train_test_split(data, test_size=TEST_RATIO, random_state=42)
train, val = train_test_split(train_val, test_size=VAL_RATIO/(TRAIN_RATIO + VAL_RATIO), random_state=42)

# --- SAVE METADATA ---

with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "train.json"), "w") as f:
    json.dump(train, f, indent=2)
with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "val.json"), "w") as f:
    json.dump(val, f, indent=2)
with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "test.json"), "w") as f:
    json.dump(test, f, indent=2)

print(f"Generated {len(train)} training, {len(val)} validation, {len(test)} test pairs.")

