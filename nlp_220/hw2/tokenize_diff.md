# Difference between Word Tokenizers in NLTK and SpacCy

## Introduction

NLTK and Spacy are both useful tools when processing texts. They both have tokenizers for extracting words out of sentences as tokens. But what is the difference between them?
(For NLTK tokenizer, here is using `wordpunct_tokenizer`)

## Difference

After examining couple of the tokenized results of both tokenizers, there are some differences between NLTK and SpaCy tokenizers are observed. They are illustrated as the following:

### Space and New Line

This is the most common case. The SpaCy will extract the new line character as a token and NLTK will not consider it. Moreover, the SpaCy will also take consideration of the space characters. After looking at the raw data, it turns out the space characters that SpaCy kept are the space after the end of a sentence(after a period).

```bash
[] # NLTK
[' ', ' \n'] # SpaCy
```

### Apostrophe

Apostrophe is a difficult case to address and two tokenizers have their own way to deal with it. The NLTK tokenizer will just split them regardless the lexical meaning of the word. However, it looks like SpaCy is able to recognize the `o'clock` is a single term that may not able to be split. It also kept the `n't` which indicates the meaning of "not" which may be an important information for the model.

```bash
['o', "'", 'clock'] # NLTK
["o'clock"] # SpaCy
```

```bash
['won', "'", 't'] # NLTK
['wo', "n't", ] # SpaCy
```

### Abbreviation

The abbreviation term can be an important named entity for the recognition. The NLTK will separate them all and remove the punctuation. But the SpaCy just kept them as a whole token for all abbreviation cases.

```bash
['C', 'U', 'H', 'K'] # NLTK
['C.U.H.K'] # SpaCy
```

```bash
['B', 'B', 'Q'] # NLTK
['B.B.Q'] # SpaCy
```

### Some certain entity

In NLTK, the tokenizer will extract the month and year separately. However, SpaCy are able to extract them together as a single token. These two methods may lead to different prediction for the model and which is better may depend on the task and the domain we want to focus on. There may be some other entities that SpaCy is able to recognize but there is only the type of time entity being observed.

```bash
['1983', 'August'] # NLTK
['August,1983'] # SpaCy
```

### Special Case

I am not really sure this is a special case since it is just like the previous case `won't` but without apostrophe. It can extract out the token of `not` which can provide important information separately with `can`.

```bash
['cannot'] # NLTK
['can', 'not'] # SpaCy
```

## Conclusion

The NLTK tokenizer usually directly split the tokens apart from any space and punctuations such as period or apostrophe without any other considerations. However, the SpaCy may consider more on the meaning of the term and then decide whether it should be split as different tokens. This may have huge effect on the prediction of the model, either increasing or decreasing the performance. There may not be a perfect way to perform tokenization, but it will be more practical and helpful if we can tokenize sentences based on the lexical meaning.
