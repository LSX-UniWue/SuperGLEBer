import csv

input_file = '/data/Germeval/2025/HarmfulContent/c2a/test.csv'
output_file = '/Users/juliawunderle/SuperGLEBer/data/Germeval/2025/HarmfulContent/c2a/test_add.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile, delimiter=';')
    fieldnames = reader.fieldnames

    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()

    for row in reader:
        if not row['C2A']:  # Check if 'C2A' is empty
            row['C2A'] = 'False'
        writer.writerow(row)

print(f"Updated CSV written to {output_file}")
