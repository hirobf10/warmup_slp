import argparse
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def get_text_label(file):
    X, y = [], []
    for line in file:
        y_i, x_i = line.split("\t")
        X.append(x_i)
        assert int(y_i) == -1 or int(y_i) == 1
        if int(y_i) == -1:  # NON-PERSON
            y.append(0)
        elif int(y_i) == 1:  # PERSON
            y.append(1)
    return X, y


def get_scores(tn, fp, fn, tp):
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * prec * recall / (prec + recall)

    return acc, prec, recall, f1


def main(args):
    with open(args.train_file_path, "r", encoding="utf-8") as f:
        train_file = [line.splitlines()[0] for line in f.readlines()]

    with open(args.test_file_path, "r", encoding="utf-8") as f:
        test_file = [line.splitlines()[0] for line in f.readlines()]

    text_train, y_train = get_text_label(train_file)
    text_test, y_test = get_text_label(test_file)

    cv = CountVectorizer()
    X_train = cv.fit_transform(text_train)
    X_test = cv.transform(text_test)

    model = LogisticRegression(random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc, prec, recall, f1 = get_scores(*confusion_matrix(y_test, y_pred).flatten())

    print("=" * 15, "Results", "=" * 15)
    print(
        f"Accuracy: {acc : .3f}, Precision: {prec : .3f}, Recall: {recall : .3f}, f-score: {f1 : .3f}"
    )

    idx = random.randint(0, len(text_test) - 1)
    print("=" * 15, "Example", "=" * 15)
    print(text_test[idx])
    print("-" * 39)
    pred = model.predict(cv.transform([text_test[idx]]))
    print("The model predicted this sentence as: ", end="")
    if pred[0] == 0:
        print("NON-PERSON")
    else:
        print("PERSON")
    print("=" * 39)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    args = parser.parse_args()

    SEED = 42

    main(args)
