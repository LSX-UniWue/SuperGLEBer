import random

import pandas as pd


def get_counts(data):
    counts = {}
    for d in data:
        target = d[-2] + " " + d[-1]
        counts[target] = counts.get(target, 0) + 1
    return counts


def write_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for d in data:
            line = '\t'.join(d[:-1]) + '\t' + d[-1] + '\n'
            file.write(line)


def split_data(data: str, train_split: float = 0.7, dev_split: float = 0.15, test_split: float = 0.15):
    with open(data, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = [line.strip().split('\t') for line in lines]
    print(f"Length of dataset {len(data)}")
    df = pd.DataFrame(data, columns=['Sentence1', 'Sentence2', 'Word', 'Target'])

    df['Phrase_Target'] = df['Word'] + ' ' + df['Target'].astype(str)

    unique_targets = df['Phrase_Target'].unique()
    concrete_samples_list = {}
    for target in unique_targets:
        sample = df[df['Phrase_Target'] == target][['Sentence1', 'Sentence2', 'Word', 'Target']].values.tolist()
        concrete_samples_list[target] = sample

    counts = {k: len(v) for k, v in concrete_samples_list.items()}
    train = []
    dev = []
    test = []

    for k, v in concrete_samples_list.items():
        num_samples = len(v)

        if num_samples <= 3:
            for element in v:
                selected_list = random.choices([train, dev, test], [train_split, dev_split, test_split])[0]
                selected_list.append(element)

        elif num_samples == 3:
            train.append(v[0])
            test.append(v[1])
            dev.append(v[2])

        elif num_samples > 3:
            num_train = int(train_split * num_samples)
            num_dev = int(dev_split * num_samples)
            num_test = num_samples - num_train - num_dev

            train.extend(v[:num_train])
            dev.extend(v[num_train:num_train + num_dev])
            test.extend(v[num_train + num_dev:num_train + num_dev + num_test])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test, counts, unique_targets


if __name__ == '__main__':
    path = "/Users/juliawunderle/translator/data/raw/WebCAGe/preprocessed/train.txt"
    train_data, dev_data, test_data, counts_orig, unique = split_data(path)
    print(f"Train {len(train_data)} - {len(train_data) / 9350:.3f}; Dev {len(dev_data)} - {len(dev_data) / 9350:.3f}; Test {len(test_data)} - {len(test_data) / 9350:.3f}")
    counts_train = get_counts(train_data)
    counts_dev = get_counts(dev_data)
    counts_test = get_counts(test_data)

    comparison_table = pd.DataFrame({
        'Element': list(unique),
        'All': [counts_orig.get(element, 0) for element in unique],
        'Train': [counts_train.get(element, 0) for element in unique],
        'Dev': [counts_dev.get(element, 0) for element in unique],
        'Test': [counts_test.get(element, 0) for element in unique],
    })

    comparison_table.to_csv('comparison_table.csv', index=False)

    write_file(train_data, "train.txt")
    write_file(dev_data, "dev.txt")
    write_file(test_data, "test.txt")
