with open('/Users/juliawunderle/translator/data/raw/NER_News/Wiki_News/train.txt', 'r') as input_file, open('train.tsv', 'w') as output_file:
    for line in input_file:
        if not line.startswith('#'):
            output_file.write(line)
