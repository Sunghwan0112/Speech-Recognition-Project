def augment_spk2gender(file_path):
    augmented_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            number, gender = line.strip().split()
            augmented_number = 'a' + number
            augmented_line = f'{augmented_number} {gender}'
            augmented_lines.append(augmented_line)

    return augmented_lines


# Specify the path to your spk2gender file
file_path = 'spk2gender'

# Get the augmented lines
augmented_content = augment_spk2gender(file_path)

# You can then write these lines back to a file or use them as needed
with open('spk2gender_new', 'w') as file:
    for line in augmented_content:
        file.write(line + '\n')
