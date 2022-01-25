from collections import deque, defaultdict
import math
import argparse

from sacremoses import MosesTokenizer


class Ngram:
    def __init__(self, n):
        if not isinstance(n, int):
            raise TypeError
        if n < 1:
            raise ValueError
        self.n = n
        self._trained = False
        self.counts = None
        self.context_counts = None
        self.counts_inv = None
        self.prob = None

        self.mt = MosesTokenizer(lang="en")

    def __call__(self, text: str):
        if not self._trained:
            print("Call train before predicting")
            return
        return self.mle_with_entropy(text)

    def train(self, corpus):
        """
        Train Ngram with the corpus.
        """
        counts = defaultdict(lambda: defaultdict(int))
        counts_inv = defaultdict(lambda: defaultdict(int))

        for line in corpus:
            tokens = deque(self.mt.tokenize(line))
            if self.n == 1:
                tokens.appendleft("<s>")
                tokens.append("</s>")
            else:
                for _ in range(self.n - 1):
                    tokens.appendleft("<s>")
                    tokens.append("</s>")
            tokens = list(tokens)
            for i in range(len(tokens) - self.n + 1):
                counts[" ".join(tokens[i : i + self.n - 1])][
                    tokens[i : i + self.n][-1]
                ] += 1
                counts_inv[tokens[i : i + self.n][-1]][
                    " ".join(tokens[i : i + self.n - 1])
                ] += 1
            self.counts = counts
            self.counts_inv = counts_inv

        self.prob = self.kneser_ney_smoothing()
        self._trained = True

    def kneser_ney_smoothing(self, d=0.75):
        prob = defaultdict(lambda: defaultdict(int))
        num_ngram_types = len(
            [
                context
                for context_count in self.counts_inv.values()
                for context in context_count.keys()
            ]
        )
        for context, word_count in self.counts.items():
            context_count = sum([count for count in word_count.values()])
            num_context = sum([count for count in word_count.values()])
            for word, count in word_count.items():
                lmd = d * len(word_count) / num_context
                p_cont = len(self.counts_inv[word]) / num_ngram_types
                prob[context][word] = max(count - d, 0) / context_count + lmd * p_cont
        return prob

    def mle_with_entropy(self, text: str) -> float:
        prob = 0
        tokens = deque(self.mt.tokenize(text))
        if self.n == 1:
            tokens.appendleft("<s>")
            tokens.append("</s>")
        else:
            for _ in range(self.n - 1):
                tokens.appendleft("<s>")
                tokens.append("</s>")
        tokens = list(tokens)
        for i in range(len(tokens) - self.n + 1):
            p = self.prob[" ".join(tokens[i : i + self.n - 1])][
                tokens[i : i + self.n][-1]
            ]
            if p == 0:
                continue
            else:
                prob += math.log(p)
        return -prob / len(tokens)


def main(args):
    with open(args.train_file_path, "r", encoding="utf-8") as f:
        train_file = [line.splitlines()[0] for line in f.readlines()]

    with open(args.test_file_path, "r", encoding="utf-8") as f:
        test_file = [line.splitlines()[0] for line in f.readlines()]

    ngram = Ngram(args.n)
    ngram.train(train_file)
    entropy_list = [ngram(line) for line in test_file]
    print(
        f"The mean of entropy when {args.n}-gram:",
        sum(entropy_list) / len(entropy_list),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("n", type=int, help="Number of n-gram that you want to model")
    args = parser.parse_args()

    main(args)
