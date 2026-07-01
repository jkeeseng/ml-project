import pandas as pd
import os
from pydub import AudioSegment
import requests
import subprocess
from pathlib import Path


def download_and_convert_audio(id, label, base_path, max_size_kb=400):
    """
    Downloads an MP3 file using the given ID, converts it to WAV format, and saves it to a directory based on the label.
    Excludes files larger than 'max_size_kb'.

    Args:
    - id: The unique identifier for the audio file.
    - label: The label used to name the subdirectory where the WAV file will be stored.
    - base_path: The base path where the audio file will be saved.
    - max_size_kb: Maximum size of the WAV file in kilobytes to be included in the dataset.

    Returns:
    - path: The path to the converted WAV file, or None if the file is larger than the maximum size.
    """
    url = f"https://readup.com.au/api/recording/{id}/audio"
    temp_path = Path(base_path) / "temp"
    wav_path = Path(base_path) / "wav" / label
    temp_file_path = temp_path / f"{id}.mp3"
    wav_file_path = wav_path / f"{id}.wav"

    # Ensure directories exist
    temp_path.mkdir(parents=True, exist_ok=True)
    wav_path.mkdir(parents=True, exist_ok=True)

    if not wav_file_path.exists():
        # Download the MP3 file
        response = requests.get(url)
        with open(temp_file_path, 'wb') as file:
            file.write(response.content)

        # Convert MP3 to WAV
        subprocess.run(["ffmpeg", "-i", str(temp_file_path), str(wav_file_path)], check=True)

        # Clean up the temporary MP3 file
        temp_file_path.unlink()
    else:
        print(wav_file_path, 'already exists')

    # Check if the WAV file is larger than the max size
    if wav_file_path.exists() and wav_file_path.stat().st_size > max_size_kb * 1024:
        #wav_file_path.unlink()  # Delete the oversized file
        return None

    return str(wav_file_path)

def process_audio_dataset_with_questions(valid_data, questions, base_path, max_size_kb=400):
    """
    Processes a dataframe of valid data to prepare for audio dataset creation based on specific questions.
    Downloads, converts, and organizes audio files accordingly, excluding files larger than 'max_size_kb'.

    Args:
    - valid_data: DataFrame containing the valid data filtered from the dataset.
    - questions: List containing the occurrences in 'Word' that should be considered.
    - base_path: Path where the results will be saved.
    - max_size_kb: Maximum size of the WAV file in kilobytes to be included in the dataset.

    Returns:
    - Path to processed_files.csv and handles audio files.
    """

    filtered_data = valid_data[valid_data['Word'].isin(questions)]
    processed_data = filtered_data.copy()
    processed_data['Label'] = filtered_data.apply(lambda x: x['Word'] if x['HumanCorrect'] == 1 else "HumanIncorrect", axis=1)

    # Download and convert, excluding large files
    processed_data['Path'] = processed_data.apply(lambda row: download_and_convert_audio(row['ID'], row['Label'], base_path, max_size_kb), axis=1)
    processed_data = processed_data.dropna(subset=['Path'])  # Remove rows where conversion resulted in large files

    processed_data['Question'] = filtered_data['Word']
    processed_file_path = os.path.join(base_path, 'processed_files.csv')
    processed_data[['Label', 'Path', 'ID', 'Question']].to_csv(processed_file_path, index=False)

    return processed_file_path

def sample_per_question(group, correct_samples=1000, incorrect_samples=1000):
    """
    Samples exactly 'correct_samples' and 'incorrect_samples' from the group, ensuring the required number of each.
    """
    correct_count = group['HumanCorrect'].sum()
    incorrect_count = group['HumanIncorrect'].sum()

    # Sample correct and incorrect responses, with replacement if necessary
    correct = group[group['HumanCorrect'] == 1].sample(
        n=correct_samples, replace=(correct_count < correct_samples)
    )
    incorrect = group[group['HumanIncorrect'] == 1].sample(
        n=incorrect_samples, replace=(incorrect_count < incorrect_samples)
    )

    return pd.concat([correct, incorrect])


def filter_valid_questions(file_path, questions=None, min_samples=100, correct_samples = 300, incorrect_samples = 300):
    """
    Opens a CSV file, filters for valid data based on strict agreement among markers for 'Correct' or 'Incorrect' marks,
    ensures consistency across markers for each ResponseID with at least 3 marks, and identifies questions (words) with
    at least 'min_samples' samples. It returns a DataFrame with unique ResponseIDs and associated information, including
    binary values indicating whether the response was correct or incorrect, with strict enforcement of marker consensus.

    Args:
    - file_path: Path to the CSV file containing the dataset.
    - min_samples: Minimum number of samples required for a question to be considered valid.

    Returns:
    - DataFrame: Contains columns for ResponseID, Word, HumanCorrect, and HumanIncorrect,
                 with strict enforcement of no duplicate ResponseIDs and marker consensus.
    - List: A list of unique words from the valid questions.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    all_words = data['Word'].unique().tolist()
    print(all_words)
    print(len(all_words))

    if questions is not None: 
        data = data[data['Word'].isin(questions)]

    # Filter valid data where Human Mark is "Correct" or "Incorrect"
    valid_marks = data[data['Human Mark'].isin(['Correct', 'Incorrect'])]

    # Identify consistent ResponseIDs where all marks across markers are either "Correct" or "Incorrect"
    # and where each ResponseID has at least 3 markers
    consistent_response_ids = valid_marks.groupby('ResponseID').filter(
        lambda x: len(x['Marker'].unique()) == len(x) and len(x['Human Mark'].unique()) == 1 and len(x) >= 3)['ResponseID'].unique()

    # Filter the original data for consistent ResponseIDs
    consistent_data = valid_marks[valid_marks['ResponseID'].isin(consistent_response_ids)].copy()

    # Deduplicate based on ResponseID before further processing
    consistent_data.drop_duplicates(subset=['ResponseID'], inplace=True)

     # Convert Human Mark to binary indicators
    consistent_data['HumanCorrect'] = (consistent_data['Human Mark'] == 'Correct').astype(int)
    consistent_data['HumanIncorrect'] = (consistent_data['Human Mark'] == 'Incorrect').astype(int)

    sampled_data = consistent_data.groupby('Word').apply(
        sample_per_question, correct_samples=correct_samples, incorrect_samples=incorrect_samples
    ).reset_index(drop=True)

    for word in questions:
        correct_count = sampled_data[(sampled_data['Word'] == word) & (sampled_data['HumanCorrect'] == 1)].shape[0]
        incorrect_count = sampled_data[(sampled_data['Word'] == word) & (sampled_data['HumanIncorrect'] == 1)].shape[0]
        
        if correct_count != correct_samples or incorrect_count != incorrect_samples:
            raise ValueError(f"Word '{word}' does not have exactly {correct_samples} correct and {incorrect_samples} incorrect samples.")

    #print(valid_questions)

    # Aggregate data to ensure at least min_samples per word

    # Ensure the ResponseID is unique in the final dataset
    final_data = sampled_data[['ResponseID', 'Word', 'HumanCorrect', 'HumanIncorrect']].drop_duplicates(subset='ResponseID')
    final_data.rename(columns={'ResponseID': 'ID'}, inplace=True)

     # Extract a list of unique words
    unique_words = final_data['Word'].unique().tolist()

    print(final_data)

    return final_data, unique_words

def check_sample_availability(data, questions):
    for word in questions:
        correct_count = data[(data['Word'] == word) & (data['HumanCorrect'] == 1)].shape[0]
        incorrect_count = data[(data['Word'] == word) & (data['HumanIncorrect'] == 1)].shape[0]
        print(f"Word '{word}': {correct_count} Correct, {incorrect_count} Incorrect")




if __name__ == "__main__":

    questions = ['hl', 'ewe', 'molo', 'inja', 'ng', 'hayi', 'xela']
    # Adjust the paths as necessary
    base_path = 'data/filtered_dataset_isixhosa_remark_all'
    file_path = 'isixhosa converted final.csv'

    valid_data, unique_words = filter_valid_questions(file_path, questions = questions, correct_samples = 300, incorrect_samples = 300)
    check_sample_availability(valid_data, questions)


    if valid_data.empty:
        print("No valid data found after filtering.")
    else:
        print("Unique Words:", unique_words)
        print("Number of Unique Words:", len(unique_words))
        print(valid_data.head())

    
    process_audio_dataset_with_questions(valid_data, questions, base_path)
