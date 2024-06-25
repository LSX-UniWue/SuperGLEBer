from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def write_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(data))


def split(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Dataset size: {len(lines)}")
    data = [line.strip().split("\t") for line in lines]

    labels = [d[0] for d in data]
    text = [d[1] for d in data]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([label.split(" ") for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    y_train_labels = mlb.inverse_transform(y_train)
    y_dev_labels = mlb.inverse_transform(y_dev)
    y_test_labels = mlb.inverse_transform(y_test)

    train_data = ["\t".join([" ".join(labels), x]) for labels, x in zip(y_train_labels, X_train)]
    dev_data = ["\t".join([" ".join(labels), x]) for labels, x in zip(y_dev_labels, X_dev)]
    test_data = ["\t".join([" ".join(labels), x]) for labels, x in zip(y_test_labels, X_test)]

    print(f"Train: {len(train_data)} samples, Dev: {len(dev_data)} samples, Test: {len(test_data)} samples")

    write_file(train_data, "/Users/janpf/projects/Superkleber/data/raw/Hotel_Aspect/train.txt")
    write_file(dev_data, "/Users/janpf/projects/Superkleber/data/raw/Hotel_Aspect/dev.txt")
    write_file(test_data, "/Users/janpf/projects/Superkleber/data/raw/Hotel_Aspect/test.txt")


if __name__ == "__main__":
    file = "/Users/janpf/Downloads/train.txt"
    split(file)
