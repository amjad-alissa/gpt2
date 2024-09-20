import os
import pandas as pd

def read_file(file_path):
    """Read a file and return its lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def validate_line_counts(normal_lines, simple_lines):
    """Validate that both files have the same number of lines."""
    if len(normal_lines) != len(simple_lines):
        raise ValueError("The number of lines in the normal and simple files do not match.")

def parse_lines(normal_lines, simple_lines):
    """Parse the normal and simple lines into a list of dictionaries."""
    data = []
    last_normal_sentence = None
    last_simple_index = 0
    begin_simple_index = 0

    for normal in normal_lines:
        normal = normal.strip()

        if normal == last_normal_sentence:
            last_simple_index += 1
        else:
            if last_normal_sentence is not None:
                simple_sentences = [line.strip() for line in simple_lines[begin_simple_index:last_simple_index]]
                simplified_text = ' '.join(simple_sentences)
                data.append({'original': last_normal_sentence, 'simplified': simplified_text})

            last_normal_sentence = normal
            begin_simple_index = last_simple_index

            if last_simple_index < len(simple_lines):
                last_simple_index += 1

    if last_normal_sentence is not None:
        simple_sentences = [line.strip() for line in simple_lines[begin_simple_index:last_simple_index]]
        simplified_text = ' '.join(simple_sentences)
        data.append({'original': last_normal_sentence, 'simplified': simplified_text})

    return data

def process_file_pair(normal_file_path, simple_file_path):
    """Process a pair of normal and simple files and return the parsed data."""
    normal_lines = read_file(normal_file_path)
    simple_lines = read_file(simple_file_path)
    validate_line_counts(normal_lines, simple_lines)
    return parse_lines(normal_lines, simple_lines)

def save_data_to_csv(data, output_file):
    """Save the parsed data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data has been successfully parsed and saved to '{output_file}'.")

def gather_data_from_files(dataset_folder):
    """Gather data from all .normal and .simple files in the dataset folder."""
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.normal'):
            normal_file_path = os.path.join(dataset_folder, filename)
            simple_file_path = normal_file_path.replace('.normal', '.simple')

            if os.path.exists(simple_file_path):
                try:
                    parsed_data = process_file_pair(normal_file_path, simple_file_path)

                    # Create the output directory if it doesn't exist
                    output_folder = 'prepared_dataset'
                    os.makedirs(output_folder, exist_ok=True)

                    # Define the output file path
                    output_file = os.path.join(output_folder, filename.replace('.normal', '.csv'))

                    # Save the parsed data to a CSV file
                    save_data_to_csv(parsed_data, output_file)
                except ValueError as e:
                    print(e)
            else:
                print(f"Warning: Corresponding .simple file not found for {normal_file_path}")

def main():
    dataset_folder = 'dataset'  # Folder containing the .normal and .simple files
    gather_data_from_files(dataset_folder)

if __name__ == "__main__":
    main()
