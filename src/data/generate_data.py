import random
from Bio import Align
import os

# --- CONFIGURATION ---
SEQ_LENGTH = 128
NUM_PAIRS = 1200
MUTATION_RATES = [0.0, 0.15, 0.2, 0.25, 0.35, 0.45, 0.55]
TRAIN_RATIO = 0.67
VAL_RATIO = 0.17
TEST_RATIO = 0.16
OUTPUT_DIR = "data_dotplots"

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
                seq[i] = random.choice([b for b in 'ACGT' if b != seq[i]])
            elif mutation_type == 'del' and i < len(seq) - 1:
                seq[i] = ''
            elif mutation_type == 'ins':
                seq[i] += random.choice('ACGT')
    return ''.join(seq)[:SEQ_LENGTH]
     
def compute_alignment_score(s1, s2):
    return aligner.score(s1, s2)

