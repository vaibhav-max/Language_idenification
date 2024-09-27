from tqdm import tqdm
import json
import os
import shutil

# Path to the JSON file
json_file_path = r'/data/Vaani/Dataset/dataWmT.json'

# Directory containing all the audio files
audio_files_dir = r'/data/Vaani/Dataset/Audios_all_district_vaani_2'

# Directory to store the resulting audio files organized by language
result_dir = r'/data/Vaani/Dataset/Audio_language_specific'

# Read the JSON file with UTF-8 encoding
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

print("Json Loaded")

# Initialize progress counter
total_entries = len(data)
processed_entries = 0

print("Copying files...")

# Create tqdm instance with total set to the number of folders
progress_bar_folders = tqdm(os.listdir(audio_files_dir), desc="Folders")

for folder_name in progress_bar_folders:
    folder_path = os.path.join(audio_files_dir, folder_name)
    if os.path.isdir(folder_path):
        # Create tqdm instance with total set to the number of entries in the JSON data
        progress_bar_entries = tqdm(data, desc=f"{folder_name} Entries", total=total_entries)
        for entry in progress_bar_entries:
            audio_filename = entry['audioFilename']
            assert_language = entry['assertLanguage']
            
            # Create folder for the assert language if it doesn't exist
            language_folder = os.path.join(result_dir, assert_language)
            if not os.path.exists(language_folder):
                os.makedirs(language_folder)
            
            audio_filename = audio_filename.split('/')[2]
            # Copy the audio file to the corresponding assert language folder
            audio_file_source = os.path.join(folder_path, audio_filename)
            if os.path.exists(audio_file_source):
                shutil.copy(audio_file_source, language_folder)
            
            # Update progress counter for entries
            processed_entries += 1
            progress_bar_entries.update(1)
            progress_bar_entries.set_description(f"{folder_name} Progress: {processed_entries}/{total_entries}")

        progress_bar_entries.close()  # Close the progress bar for entries

progress_bar_folders.close()  # Close the progress bar for folders

print("Files organized successfully.")
