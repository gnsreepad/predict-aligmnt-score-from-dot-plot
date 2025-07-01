import random
import os
import shutil
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Bio import Align

# --- CONFIGURATION ---
SEQ_LENGTH = 128
TARGET_PAIRS_PER_BIN = 1000   # Max samples per bin
SCORE_BINS = list(range(40, 121, 10))  # 40-50, 50-60, ..., 110-120
MUTATION_RATES = [0.09, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
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


# Custom k-mer alignment score function
# This function counts the number of matching k-mers between two sequences.
# It can be used to compute a similarity score based on k-mers.
def compute_kmer_alignment_score(s1, s2, k=3):
    max_i = min(len(s1), len(s2)) - k + 1
    score = sum(1 for i in range(max_i) if s1[i:i+k] == s2[i:i+k])
    return score

def compute_dotplot(s1, s2, k=3):
    dotplot = np.zeros((len(s1) - k + 1, len(s2) - k + 1), dtype=np.uint8)
    for i in range(len(s1) - k + 1):
        kmer1 = s1[i:i+k]
        for j in range(len(s2) - k + 1):
            kmer2 = s2[j:j+k]
            if kmer1 == kmer2:
                dotplot[i, j] = 255
    return dotplot

def save_dotplot_image(matrix, filename):
    img = Image.fromarray(matrix)
    img = img.convert('L')
    img.save(filename)

# --- BALANCED DATA GENERATION ---
bin_counts = {f"{lo}-{lo+10}": 0 for lo in SCORE_BINS}
data = []

print("Generating balanced samples...")
while sum(bin_counts.values()) < len(bin_counts) * TARGET_PAIRS_PER_BIN:
    base_seq = generate_random_sequence(SEQ_LENGTH)
    mutation_rate = random.choice(MUTATION_RATES)
    mutated_seq = mutate_sequence(base_seq, mutation_rate)
    score = compute_alignment_score(base_seq, mutated_seq)
    score = int(round(score))

    # Assign to bin
    for lo in SCORE_BINS:
        if lo <= score < lo + 10:
            bin_key = f"{lo}-{lo+10}"
            if bin_counts[bin_key] < TARGET_PAIRS_PER_BIN:
                pair_id = f"{len(data):05d}"
                dot = compute_dotplot(base_seq, mutated_seq)
                dotplot_path = os.path.join(OUTPUT_DIR, PLOTS_DIR, f"dot_{pair_id}.png")
                save_dotplot_image(dot, dotplot_path)

                data.append({
                    "id": pair_id,
                    "seq1": base_seq,
                    "seq2": mutated_seq,
                    "score": score,
                    "mutation_rate": mutation_rate,
                    "dotplot": dotplot_path
                })
                bin_counts[bin_key] += 1
            break

# --- PLOT HISTOGRAM ---
scores = [entry["score"] for entry in data]
plt.figure(figsize=(8, 5))
plt.hist(scores, bins=range(40, 131, 10), edgecolor='black')
plt.title("Balanced Histogram of Alignment Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_histogram.png"))
plt.close()

print(f"Generated {len(data)} balanced sequence pairs.")

# --- SPLIT DATA ---
train_val, test = train_test_split(data, test_size=TEST_RATIO, random_state=RANDOM_SEED)
train, val = train_test_split(train_val, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=RANDOM_SEED)

# --- SAVE METADATA ---
with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "train.json"), "w") as f:
    json.dump(train, f, indent=2)
with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "val.json"), "w") as f:
    json.dump(val, f, indent=2)
with open(os.path.join(OUTPUT_DIR, METADATA_DIR, "test.json"), "w") as f:
    json.dump(test, f, indent=2)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")