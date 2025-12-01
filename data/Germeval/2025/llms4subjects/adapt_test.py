input_file = '/Users/juliawunderle/SuperGLEBer/data/Germeval/2025/llms4subjects/old/test_de.txt'
output_file = '/Users/juliawunderle/SuperGLEBer/data/Germeval/2025/llms4subjects/test.txt'

with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        text = parts[1].strip()
        outfile.write(f'__label__rel\t{text}\n')

print(f"Labeled text written to {output_file}")
