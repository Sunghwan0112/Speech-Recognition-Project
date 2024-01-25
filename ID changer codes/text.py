def augment_text_file(file_path):
    augmented_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            augmented_number = 'a' + parts[0]
            if len(parts) > 1:
                augmented_line = f'{augmented_number} {parts[1]}'
            else:
                augmented_line = augmented_number
            augmented_lines.append(augmented_line)

    return augmented_lines


# Specify the path to your text file
file_path = 'text'

# Get the augmented lines
augmented_content = augment_text_file(file_path)

# You can then write these lines back to a file or use them as needed
with open('text_new', 'w') as file:
    for line in augmented_content:
        file.write(line + '\n')
