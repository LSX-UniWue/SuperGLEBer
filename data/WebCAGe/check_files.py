with open('train.txt', 'r') as file:
    train_lines = set(file.readlines())

with open('test.txt', 'r') as file:
    test_lines = set(file.readlines())

common_lines = train_lines.intersection(test_lines)
print(common_lines)
print(f'Number of common lines between train and test: {len(common_lines)}')
