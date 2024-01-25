def process_wav(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            first_part_augmented ='a' + parts[0]
            file_path = parts[1]

            # Modifying the path by replacing 'train-clean-100' with 'train-augmented'
            new_path = file_path.replace('train-clean-100', 'train-augmented')

            file.write(f"{first_part_augmented} {new_path}\n")


# Define the paths for the input and output files
input_file_path = 'wav.scp'  # The original file path
output_file_path = 'wav_new.scp' # The augmented file path

process_wav(input_file_path, output_file_path)



