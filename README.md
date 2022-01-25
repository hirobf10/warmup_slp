# Warm-up
## Traditional Language model
I implemented n-gram models in [src/language_model.py](https://github.com/hirobf10/warmup_slp/blob/main/src/language_model.py)  

### Command
You can run the n-gram model with following command; this command creates trigram.

```bash
python src/language_model.py "data/wiki-en-train.word" "data/wiki-en-test.word" 3
```

### Result
I report each average of entropy on [wiki-en-test.word](https://github.com/hirobf10/warmup_slp/blob/main/data/wiki-en-test.word) when adopting bigram, trigram or 5-gram.

|                    | bigram | trigram | 5-gram |
|--------------------|--------|---------|--------|
| Average of entropy | 1.3908 | 0.3258  | 0.1608 |

### Simple explanation of my codes
1. Insertion of special symbols `<s>` and `<\s>`
    ```py
    if self.n == 1:
        tokens.appendleft("<s>")
        tokens.append("</s>")
    else:
        for _ in range(self.n - 1):
            tokens.appendleft("<s>")
            tokens.append("</s>")
    ```
    I inserted n-1 `<s>` to the head of each sentence and `</s>` to the end (except for unigram), where **n** corresponds to n of **n**-gram.
    This is because if only one `<s>` is inserted to each sentence, the n-gram model need additional n-2 tokens as context for generation. So the n-gram model needs n-1 `<s>`.

2. Kneser-Ney Smoothing ([codes](https://github.com/hirobf10/warmup_slp/blob/7c57e28170bd8598ce6347c46b6637466aa1d94b/src/language_model.py#L59))  
    To implement this smoothing method. I referred to Eq. 3.35 in [this text](https://web.stanford.edu/~jurafsky/slp3/3.pdf).


## Class prediction
- Features: CountVector  
- Model: LogisticRegression  

In [src/class_prediction.py.py](https://github.com/hirobf10/warmup_slp/blob/main/src/class_prediction.py), I implemented a model which predicts whether a sentence is related to PERSON or not.  
I used as features counts of tokens appering in documents.  
To count up frequencies and create vocabulary, I exploited [CountVector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).  
Then, a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model recieves the vector as input and makes predictions.

### Command
```bash
python src/class_prediction.py "data/titles-en-train.labeled" "data/titles-en-test.labeled"
```

### Result

| Accuracy | Precision | Recall | f-score |
|----------|-----------|--------|---------|
| 0.942    | 0.952     | 0.925  | 0.1608  |