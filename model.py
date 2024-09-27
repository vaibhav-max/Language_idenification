#!/usr/bin/env python
# coding: utf-8

# In[2]:
import os
import csv
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
import pycountry
import torch.nn.functional as F 
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-4017"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)


# In[ ]:


# Converting the audio to mono is a common practice in audio processing pipelines, and it's often done for several reasons:

# Compatibility: Many models, including the one you're using, are designed to work with mono audio. Converting stereo audio to mono ensures compatibility with such models.

# Reduced complexity: Working with mono audio simplifies processing and reduces computational complexity. It ensures that the model is only considering one channel of audio data,
#  which can lead to faster inference and reduced memory requirements.

# Consistency: Converting stereo audio to mono ensures consistency in the input data format. It eliminates any potential discrepancies that may arise from stereo audio, 
# such as differences in volume or phase between channels.

# Improved performance: In many cases, stereo audio may not provide significant additional information for the task at hand. By converting it to mono, you focus on the 
# essential aspects of the audio relevant to the task, potentially leading to improved performance.

# Also in the above model mono audio is expected as input.


# In[5]:

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for processing.")
else:
    device = torch.device("cpu")
    print("Using CPU for processing.")

# Move the model to the appropriate device
model.to(device)

def get_language_name(iso_code):
    try:
        language = pycountry.languages.get(alpha_3=iso_code)
        return language.name
    except AttributeError:
        return "Language not found"
    
def process_folder(folder_path, model, processor, output_dir, parent_path):
    # Get the folder name
    folder_name = os.path.basename(folder_path)

    parent_folder_name = os.path.basename(parent_path)
    
    # Create a CSV file to store the results
    csv_save_path = os.path.join(output_dir, parent_folder_name)
    os.makedirs(csv_save_path, exist_ok=True)
    csv_file_path = os.path.join(csv_save_path, folder_name + "_predicted_labels.csv")


    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Asserted Language", "Detected Language", "Probability"]) 
        # List all files in the folder
        file_list = os.listdir(folder_path)
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(file_list, desc=f"Processing {folder_name}", unit="file")
        
        # Loop through each file
        for file_name in progress_bar:
            # Construct the full path to the audio file
            file_path = os.path.join(folder_path, file_name)

            # Skip non-audio files (e.g., previously generated CSV files)
            if not file_name.endswith(".wav"):
                continue
            
            # Load the audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample the audio to 16 kHz if the sampling rate is different
            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Ensure the audio is mono (if not already)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Squeeze the tensor to remove extra dimensions
            waveform = waveform.squeeze(0)
            # print("waveform shape " , waveform.shape)
            # # Pass the preprocessed waveform to your model
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

            # # Print the shape of each tensor in the inputs dictionary
            # for key, value in inputs.items():
            #     print(f"Shape of {key}: {value.shape}")
                
            with torch.no_grad():
                # Move inputs to the appropriate device
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs).logits
            
            probs = F.softmax(outputs, dim=-1)[0].tolist()
            lang_id = torch.argmax(outputs, dim=-1)[0].item()
            detected_lang_iso = model.config.id2label[lang_id]
            detected_lang = get_language_name(detected_lang_iso)

            detected_lang_prob = probs[lang_id]
            # Write the filename, foldername, and detected language to the CSV file
            csv_writer.writerow([file_name, folder_name, detected_lang, detected_lang_prob])
            
    print(f"CSV file saved for folder {folder_name}: {csv_file_path}")

# Path to the directory containing folders of audio files
directory_path = "/data/Vaani/Dataset/Audio_language_specific"
csv_save_path = "/data/Vaani/CSVs" 

# Iterate through each folder in the directory
for folder_name in tqdm(os.listdir(directory_path), desc="Main folders"):
    folder_path = os.path.join(directory_path, folder_name)
    print(folder_name)

    for subfolder in tqdm(os.listdir(folder_path), desc="Subfolders", leave=False):
        subfolder_path = os.path.join(folder_path, subfolder)
        # Check if the item in the directory is a folder
        if os.path.isdir(subfolder_path):
            print(subfolder)
            process_folder(subfolder_path, model, processor, csv_save_path, folder_path)
            




