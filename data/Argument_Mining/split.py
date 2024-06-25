import csv
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def write_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for d in data:
            line = f"{' '.join((f'__label__{l}' for l in d[1].split('+')))}\t{d[2].strip()}\t{d[0].strip()}\n"
            f.write(line)


def get_label_distribution(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    labels = [line.strip().split("\t")[4] for line in lines]
    label_distribution = Counter(labels)
    return label_distribution


def split_data(path):
    reader = csv.reader(open(path, "r", encoding="utf-8"), delimiter="\t")
    data = list(reader)
    header = data[0]
    data = data[1:]

    labels = [d[header.index("code")] for d in data]
    content = [
        (d[header.index("id")].strip(), d[header.index("code")].strip(), d[header.index("content")].strip())
        for d in data
    ]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([label.split("+") for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(content, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    # train_data = [d for id_train in X_train for d in data if d[0] == id_train]
    # dev_data = [d for id_dev in X_dev for d in data if d[0] == id_dev]
    # test_data = [d for id_test in X_test for d in data if d[0] == id_test]

    print(f"Train: {len(X_train)} samples, Dev: {len(X_dev)} samples, Test: {len(X_test)} samples")

    write_file(X_train, "data/raw/Argument_Mining/train.txt")
    write_file(X_dev, "data/raw/Argument_Mining/dev.txt")
    write_file(X_test, "data/raw/Argument_Mining/test.txt")

    return X_train, X_dev, X_test


if __name__ == "__main__":
    file = "/Users/janpf/Downloads/train.txt"

    train, dev, test = split_data(file)

    train_dist = get_label_distribution("data/raw/Argument_Mining/train.txt")

    print(
        f"Train: mpos: {train_dist['mpos']} - {train_dist['mpos'] / 5356:.3f}; premise: {train_dist['premise']} - {train_dist['premise'] / 9615:.3f};  non-arg: {(train_dist['non-arg'])} - {train_dist['non-arg'] / 2335:.3f}; mpos+premise: {train_dist['mpos+premise']} - {train_dist['mpos+premise'] / 546:.3f}"
    )

    dev_dist = get_label_distribution("data/raw/Argument_Mining/dev.txt")
    print(
        f"Dev: mpos: {dev_dist['mpos']} - {dev_dist['mpos'] / 5356:.3f}; premise: {dev_dist['premise']} - {dev_dist['premise'] / 9615:.3f};  non-arg: {(dev_dist['non-arg'])} - {dev_dist['non-arg'] / 2335:.3f}; mpos+premise: {dev_dist['mpos+premise']} - {dev_dist['mpos+premise'] / 546:.3f}"
    )

    test_dist = get_label_distribution("data/raw/Argument_Mining/test.txt")
    print(
        f"Test: mpos: {test_dist['mpos']} - {test_dist['mpos'] / 5356:.3f}; premise: {test_dist['premise']} - {test_dist['premise'] / 9615:.3f};  non-arg: {(test_dist['non-arg'])} - {test_dist['non-arg'] / 2335:.3f}; mpos+premise: {test_dist['mpos+premise']} - {test_dist['mpos+premise'] / 546:.3f}"
    )
