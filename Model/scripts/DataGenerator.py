import numpy as np
from PIL import Image

import tensorflow as tf
tf._tf_uses_legacy_keras = True
def data_loader(set_type, batch_size):
    # Load numpy buckets into dictionary
    train_dict = np.load("properties_buckets\\" + set_type + '_buckets.npy', allow_pickle= True).tolist()
    
    # Iterate over each bucket
    for img_size_group in train_dict.keys():
        # Create a list to hold image names and tokenized sequences
        train_list = train_dict[img_size_group]
        
        # Determine the number of files used for processing
        num_files = (len(train_list) // batch_size) * batch_size
        
        # Iterate over each batch
        for batch_idx in range(0, num_files, batch_size):
            # Create a sublist of the training list for the current batch
            train_sublist = train_list[batch_idx:batch_idx + batch_size]
            
            # Initialize lists for images and batch forms (labels)
            imgs = []
            batch_forms = []
            
            # Process each image and its corresponding tokenized sequence
            for img_name, tokenized_seq in train_sublist:
                # Load and preprocess the image
                img = np.asarray(Image.open('Preprocessed_Images//' + img_name).convert('YCbCr'))[:, :, 0][:, :, None]
                imgs.append(img)
                
                # Append the tokenized sequence to the batch forms list
                batch_forms.append(tokenized_seq)
            
            # Convert the list of images into a numpy array
            imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 3, 1, 2)
            
            # Calculate the lengths of all the formulas in the batch
            lens = [len(form) for form in batch_forms]
            
            # Initialize mask and Y arrays
            mask = np.zeros((batch_size, max(lens)), dtype=np.int32)
            Y = np.zeros((batch_size, max(lens)), dtype=np.int32)
            
            # Fill mask and Y arrays with data
            for i, form in enumerate(batch_forms):
                mask[i, :len(form)] = 1
                Y[i, :len(form)] = form
            
            # Yield the data to the model for processing
            yield imgs, Y, mask
