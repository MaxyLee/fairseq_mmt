import sys
import sacrebleu
import evaluate as hf_evaluate

from transformers import AutoTokenizer
# from vizseq.scorers.meteor import METEORScorer

def read_file(path):
    i = 0
    toks = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            toks.append(line)
            i += 1
    return toks, i

sys_toks, i1 = read_file(sys.argv[1])
ref_toks, i2 = read_file(sys.argv[2])

assert i1 == i2, "error"

translations, ref = [], []
for k in range(i1):
    translations.append(sys_toks[k])
    ref.append(ref_toks[k])

# meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
#         translations, [ref]
#     )
# print(meteor_score)

bleu = sacrebleu.corpus_bleu(translations, [ref], lowercase=True, tokenize='zh')

print(bleu)
print(bleu.score)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

meteor = hf_evaluate.load('meteor')
        
predicts = [' '.join(tokenizer.tokenize(s)) for s in translations]
answers = [' '.join(tokenizer.tokenize(s)) for s in ref]

results = meteor.compute(predictions=predicts, references=answers)
print(results)