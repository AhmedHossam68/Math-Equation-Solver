import numpy as np
from PIL import Image
import os

# Generate vocabulary and char_to_idx dictionaries
vocab = open("Latex Vocabulary\\latex_vocab.txt").readlines()
formulae = open("normalized formulas\\formulas.norm.lst").readlines()
char_to_idx = {x.strip(): i for i, x in enumerate(vocab)}
char_to_idx.update({'#UNK': len(char_to_idx), '#START': len(char_to_idx) + 1, '#END': len(char_to_idx) + 2})

print("Vocabulary size: ", len(char_to_idx))

# Create train, test, validate files
set = "testing_list"  # Use train, valid or test to generate corresponding files
file_list = open(f"filtered formulas list\\{set}.lst").readlines()
set_list = []
for line in file_list:
    img_name, line_idx = line.strip().split()
    form = formulae[int(line_idx)].strip().split()
    out_form = [char_to_idx.get('#START')]
    for c in form:
        out_form.append(char_to_idx.get(c, char_to_idx['#UNK']))
    out_form.append(char_to_idx.get('#END'))
    set_list.append([img_name, out_form])

    buckets = {}
    file_not_found_count = 0
for img_name, out_form in set_list:
    if os.path.exists(f'.\Preprocessed_Images\{img_name}'):
        img_shp = Image.open(f'.\Preprocessed_Images\{img_name}').size
        buckets.setdefault(img_shp, []).append((img_name, out_form))
    else:
        file_not_found_count += 1

print(f"Num files found in {set} set: {len(set_list) - file_not_found_count}/{len(set_list)}")
    
properties = {
'vocab_size': len(vocab),
'vocab': vocab,
'char_to_idx': char_to_idx,
'idx_to_char': {y: x for x, y in char_to_idx.items()}
}
np.save('properties.npy', properties)

# Print 8 random formulas
np.random.seed(1234)
formulae = open("normalized formulas\\formulas.norm.lst").readlines()
for _ in range(8):
    print(np.random.choice(formulae))
    print()

np.save(set + '_buckets',buckets)

