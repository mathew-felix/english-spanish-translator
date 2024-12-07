import re
import os
import unicodedata
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_unique_characters(column):
    unique_chars = set()
    for text in column.dropna():  # Exclude NaN values
        unique_chars.update(text)  # Add characters to the set
    return unique_chars

def InspectDataset(file_path):
    """
    Inspects the English-Spanish translation dataset for basic statistics,
    missing values, duplicates, and unwanted characters.

    Parameters:
    - file_path (str): Path to the CSV file to inspect.

    Returns:
    - None
    """

    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns.\n")

        # Display basic information
        print("Dataset Information:")
        print(data.info())

        # Check for missing values
        print("\nMissing Values:")
        print(data.isnull().sum())

        # Check for duplicates
        print("\nDuplicate Rows:")
        duplicate_count = data.duplicated().sum()
        print(duplicate_count)
        if duplicate_count > 0:
            print(f"Total duplicate rows: {duplicate_count}")

        # Define function to check unwanted characters
        def contains_unwanted_characters(text):
            # Updated pattern to include numbers and additional punctuation
            unwanted_pattern = r"[^a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ.,!?;'\"()\-\u2013\u2014\u2015]"
            return bool(re.search(unwanted_pattern, text))

        # Add tqdm for progress tracking
        tqdm.pandas(desc="Checking unwanted characters")

        # Identify rows with unwanted characters in English column
        print("\nProcessing English column for unwanted characters...")
        data['English_Issues'] = data['English'].astype(str).progress_apply(contains_unwanted_characters)

        # Identify rows with unwanted characters in Spanish column
        print("\nProcessing Spanish column for unwanted characters...")
        data['Spanish_Issues'] = data['Spanish'].astype(str).progress_apply(contains_unwanted_characters)

        # Identify unique characters in each column
        unique_chars_english = get_unique_characters(data['English'])
        unique_chars_spanish = get_unique_characters(data['Spanish'])

        # Print unique characters
        print("\nUnique characters in the English column:")
        print(sorted(unique_chars_english))  # Sorted for better readability

        print("\nUnique characters in the Spanish column:")
        print(sorted(unique_chars_spanish))  # Sorted for better readability

        # Summarize issues
        print("\nRows with Unwanted Characters in English or Spanish:")
        issues = data[data['English_Issues'] | data['Spanish_Issues']]
        print(f"Total rows with issues: {len(issues)}")

        # Optionally, you can inspect a sample of problematic rows
        if len(issues) > 0:
            print("\nSample problematic rows:")
            print(issues.head())

        # Drop auxiliary columns used for analysis
        data = data.drop(columns=['English_Issues', 'Spanish_Issues'])

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def CleanDataset(file_path, output_path='./data/cleaned_dataset.csv'):
    """
    Cleans the English-Spanish translation dataset by handling missing values,
    removing duplicates, filtering unwanted characters, and saving the cleaned data.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - output_path (str): Path where the cleaned CSV will be saved.

    Returns:
    - None
    """

    # Check if the input file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns.")

        # Step 1: Handle missing values
        initial_count = len(data)
        data_cleaned = data.dropna(subset=['English', 'Spanish'])  # Drop rows where either column is NaN
        removed_na = initial_count - len(data_cleaned)
        print(f"Missing values handled. Removed {removed_na} rows with missing values.")

        # Step 2: Remove duplicate rows
        initial_count = len(data_cleaned)
        data_cleaned = data_cleaned.drop_duplicates()
        removed_duplicates = initial_count - len(data_cleaned)
        print(f"Duplicate rows removed. {removed_duplicates} duplicates dropped.")

        # Step 3: Refine valid characters and clean text
        # Define a comprehensive set of valid characters
        valid_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "áéíóúüñÁÉÍÓÚÜÑ"
            " .,!?'\"-()0123456789"
        )

        # Optional: Expand valid_chars if your dataset contains more necessary characters

        def clean_text(text):
            """
            Cleans the input text by normalizing Unicode characters, replacing specific
            characters, and removing any characters not in the valid_chars set.

            Parameters:
            - text (str): The text to be cleaned.

            Returns:
            - str: The cleaned text.
            """
            # Normalize Unicode characters to NFKC form
            text = unicodedata.normalize('NFKC', text)

            # Replace non-breaking spaces with regular spaces
            text = re.sub(r'\xa0', ' ', text)

            # Normalize dashes and quotes
            text = re.sub(r'[–—―]', '-', text)      # Normalize all dashes to a single dash
            text = re.sub(r'[‘’]', "'", text)       # Normalize single quotes
            text = re.sub(r'[“”]', '"', text)       # Normalize double quotes

            # Remove any character not in valid_chars
            cleaned = ''.join(char for char in text if char in valid_chars).strip()
            return cleaned

        # Apply the cleaning function to both columns
        print("Starting text cleaning for 'English' and 'Spanish' columns...")
        data_cleaned['English'] = data_cleaned['English'].astype(str).apply(clean_text)
        data_cleaned['Spanish'] = data_cleaned['Spanish'].astype(str).apply(clean_text)
        print("Unwanted characters removed and text normalized.")

        # Step 4: Drop rows that are completely empty after cleaning
        initial_count = len(data_cleaned)
        data_cleaned = data_cleaned[
            (data_cleaned['English'].str.strip() != '') |
            (data_cleaned['Spanish'].str.strip() != '')
        ]
        removed_empty = initial_count - len(data_cleaned)
        print(f"Dropped {removed_empty} rows that were empty after cleaning.")

        # Step 5: Save the cleaned dataset
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory '{output_dir}' for the cleaned dataset.")

        data_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved as '{output_path}' with {len(data_cleaned)} rows.")

    except Exception as e:
        print(f"An error occurred: {e}")

def SmallDataset(input_csv, output_csv, percent):
    # Read the original CSV
    df = pd.read_csv(input_csv)

    # Calculate 10% of the total rows
    sample_size = int(percent * len(df))

    # Randomly sample 10% of the rows without replacement
    df_sampled = df.sample(n=sample_size, random_state=42)  # random_state for reproducibility

    # Save the sampled rows to a new CSV
    df_sampled.to_csv(output_csv, index=False)

    print(f"Sampled {sample_size} rows and saved to {output_csv}")

def Split_data(file):

    train_file = "./data/train.csv"            # Path to save the training dataset
    test_file = "./data/test.csv"              # Path to save the test dataset

    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(file)

    # Ensure dataset has the required columns
    if "English" not in data.columns or "Spanish" not in data.columns:
        raise ValueError("The dataset must contain 'English' and 'Spanish' columns.")

    # Split the dataset
    print("Splitting dataset into training and testing sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the splits
    print(f"Saving training set to {train_file}...")
    train_data.to_csv(train_file, index=False)
    print(f"Saving test set to {test_file}...")
    test_data.to_csv(test_file, index=False)

    print(f"Dataset split completed. Training set: {len(train_data)}, Test set: {len(test_data)}")



def DatasetPreprocessing():
    InspectDataset('./data/english_spanish.csv')
    CleanDataset('./data/english_spanish.csv', './data/cleaned_dataset.csv')
    InspectDataset('./data/cleaned_dataset.csv')
    SmallDataset('./data/cleaned_dataset.csv','./data/small_dataset.csv', 1)
    Split_data('./data/small_dataset.csv')
