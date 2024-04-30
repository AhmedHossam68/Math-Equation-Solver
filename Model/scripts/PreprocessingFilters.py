import os
from PIL import Image

def process_data(data_path, output_path, image_dir, label_path, max_width=500, max_height=160, max_tokens=150, postfix='.png'):
    num_discard = 0
    num_nonexist = 0

    labels = open(label_path).readlines()

    with open(output_path, 'w') as fout:
        with open(data_path, 'r') as fdata:
            for line in fdata:
                line_strip = line.strip()
                if len(line_strip) > 0:
                    line_idx, img_path, mod = line_strip.split()
                    img_path = os.path.join(image_dir, img_path) + postfix
                    if not os.path.exists(img_path):
                        print('%s does not exist!' % os.path.basename(img_path))
                        num_nonexist += 1
                        continue

                    old_im = Image.open(img_path)
                    old_size = old_im.size
                    w = old_size[0]
                    h = old_size[1]

                    if w > max_width or h > max_height:
                        print('%s discarded due to large image size!' % os.path.basename(img_path))
                        num_discard += 1
                        continue

                    label = labels[int(line_idx)]

                    if len(label.strip()) == 0 or len(label.strip().split()) > max_tokens:
                        print('%s discarded due to cannot-be-parsed formula!' % os.path.basename(img_path))
                        continue

                    fout.write('%s %s\n' % (os.path.basename(img_path), line_idx))

    print('%d discarded. %d not found in %s.' % (num_discard, num_nonexist, image_dir))

# Example usage
data_path = "D:\\Ahmed\\AI\\Project\\Model\\im2latex_validate.lst"
output_path = "D:\\Ahmed\\AI\\Project\\IM2LATEX(V2)\\validate_list.lst"
image_dir = "D:\\Ahmed\\AI\\Project\\IM2LATEX(V2)\\Preprocessed_Images"
label_path = 'D:\\Ahmed\\AI\\Project\\Model\\im2latex_formulas.lst'

process_data(data_path, output_path, image_dir, label_path)
