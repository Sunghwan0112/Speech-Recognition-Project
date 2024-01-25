def augment_spk2utt(file_path):
    augmented_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            augmented_parts = ['a' + part for part in parts]
            augmented_line = ' '.join(augmented_parts)
            augmented_lines.append(augmented_line)

    return augmented_lines


# Specify the path to your spk2utt file
file_path = 'spk2utt'

# Get the augmented lines
augmented_content = augment_spk2utt(file_path)

# You can then write these lines back to a file or use them as needed
with open('spk2utt_new', 'w') as file:
    for line in augmented_content:
        file.write(line + '\n')
