import csv

input_file = '/data/Germeval/2025/FlauschErkennung/task1/test.csv'
output_file = '/Users/juliawunderle/SuperGLEBer/data/Germeval/2025/FlauschErkennung/task1/test_tap.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        if not row['flausch']:  # Check if 'flausch' is empty
            row['flausch'] = 'no'
        writer.writerow(row)

print(f"Updated CSV written to {output_file}")
