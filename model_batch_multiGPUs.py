
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import csv
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
import pycountry
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from torch.nn.utils.rnn import pad_sequence

torchaudio.set_audio_backend("soundfile")  # or "sox" or "soundfile", depending on your preference and installation

# Define the model ID
model_id = "facebook/mms-lid-4017"

# Load model and processor
processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for processing.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set specific GPUs you want to use
    model = torch.nn.DataParallel(model)

model.to(device)

# Function to get language name from ISO code
def get_language_name(iso_code):
    try:
        language = pycountry.languages.get(alpha_3=iso_code)
        return language.name
    except AttributeError:
        return "Language not found"

# Function to process a folder of audio files in batches
def process_folder(folder_path, model, processor, output_dir, parent_path, batch_size=64):
    folder_name = os.path.basename(folder_path)
    parent_folder_name = os.path.basename(parent_path)
    csv_save_path = os.path.join(output_dir, parent_folder_name)
    os.makedirs(csv_save_path, exist_ok=True)
    csv_file_path = os.path.join(csv_save_path, folder_name + "_predicted_labels.csv")

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Asserted Language", "Detected Language", "Probability"]) 

        file_list = os.listdir(folder_path)
        audio_files = [file_name for file_name in file_list if file_name.endswith(".wav")]

        if not audio_files:
            print(f"No audio files found in folder {folder_name}.")
            return
        
        progress_bar = tqdm(range(0, len(audio_files), batch_size), desc=f"Processing {folder_name}", unit="batch")
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + batch_size, len(audio_files))
            batch_files = audio_files[batch_start:batch_end]
            batch_waveforms = []
            batch_filenames = []
            max_len = 0
            
            for file_name in batch_files:
                file_path = os.path.join(folder_path, file_name)

                waveform, sample_rate = torchaudio.load(file_path)
                if sample_rate != 16000:
                    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                max_len = max(max_len, len(waveform))

                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                processed_waveform = inputs['input_values'].squeeze(0)
                batch_waveforms.append(processed_waveform)
                batch_filenames.append(file_name)

            padded_waveforms = torch.nn.utils.rnn.pad_sequence(batch_waveforms, batch_first=True, padding_value=0)
            padded_waveforms = padded_waveforms[:, :max_len] 

            with torch.no_grad():
                padded_waveforms = padded_waveforms.to(device)
                outputs = model(padded_waveforms).logits

            # Inside the process_folder function
            for i, output in enumerate(outputs):
                probs = torch.nn.functional.softmax(output, dim=-1).tolist()


                lang_id = torch.argmax(output, dim=-1).item()
                
                # Accessing config from the underlying model within DataParallel
                model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
                detected_lang_iso = model_module.config.id2label[lang_id]
                
                detected_lang = get_language_name(detected_lang_iso)
                detected_lang_prob = probs[lang_id]
                            
                csv_writer.writerow([batch_filenames[i], folder_name, detected_lang, detected_lang_prob])

            
    print(f"CSV file saved for folder {folder_name}: {csv_file_path}")

# Path to the directory containing folders of audio files
directory_path = "/data/Vaani/Dataset/Audio_language_specific_2"
csv_save_path = "/data/Vaani/CSVs_2" 

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
