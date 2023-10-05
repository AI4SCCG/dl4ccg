import utils
from rouge import Rouge
rouge = Rouge()

with open('best_result', 'r') as f:
    lines = f.readlines()

ids = []
reference_sentences = []
candidate_sentences = []

i = 0
while i < len(lines):
    # Check if the line starts with "Sample"
    if lines[i].startswith("Sample"):
        id_line = lines[i].strip().split()[1].strip(':')
        reference_line = lines[i + 1].strip().split()[1:]
        candidate_line = lines[i + 2].strip().split()[1:]

        ids.append(int(id_line))
        reference_sentences.append(reference_line )
        candidate_sentences.append(candidate_line)

        i += 3  # Move to the next sample
    else:
        i += 1  # Move to the next line
new_reference_sentences = [[list(sentence)] for sentence in reference_sentences]

blue_score, rouge_score, meteor_score = utils.eval_bleu_rouge_meteor(ids, candidate_sentences, new_reference_sentences)[: 3]

print('blue_score:', blue_score)
print('meteor_score', meteor_score)
print('rouge_score', rouge_score)

