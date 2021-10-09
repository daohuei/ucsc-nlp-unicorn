## **Tokenization in the book corpus** & **NLTK `word_tokenize` function**

When tokenizing the corpus with `word_tokenize()` function or `words()` in the book corpus, the total amounts of tokens from them are slightly different.

```python
from nltk.tokenize import word_tokenize
tokenized_words = word_tokenize(gutenberg.raw())
origin_words = gutenberg.words()
len(tokenized_words) # 2538215
len(origin_words) # 2621613
```

After examining a couple of tokens, the main difference between these two ways of tokenization is how they consider the punctuation.

The tokenizer from book corpus will tokenize the punctuation independently from its compounding word. For instance, it splitted the word "twenty-one years" into "twenty", "-", "one", and "years", which make every number separate tokens.

-   Raw Text: `twenty-one years`
    -   **Built-in words callback in the corpus**:
    ```
    'twenty', '-', 'one', 'years'
    ```

On the other hand, the `word_tokenizer` function from NLTK considered the "twenty-one" as a single token.

-   Raw Text: `twenty-one years`
    -   **Word Tokenizer from NLTK**:
    ```
    'twenty-one', 'years'
    ```

For other similar cases, there is: <br/>

-   **Raw Text**: `Mr. Woodhouse`

    -   **Built-in words callback in the corpus**:

    ```
    'Mr', '.', 'Woodhouse'
    ```

    -   **Word Tokenizer from NLTK**:

    ```
    'Mr.', 'Woodhouse'
    ```

In addition, there is a noteworthy behavior that happens when tokenizing the possessive term such as `"Emma's"`. The built-in `words()` function still extract the punctuation out of the compounding word, however, there is a different result from `word_tokenize()` function which split the apostrophe along with its following "s" character.

-   Raw Text: `sister's`

    -   **Built-in words callback in the corpus**:

    ```
    'sister', "'", 's'
    ```

    -   **Word Tokenizer from NLTK**:

    ```
    'sister', "'s"
    ```

-   Raw Text: `Emma's power`

    -   **Built-in words callback in the corpus**:

    ```
    'Emma', "'", 's', "power"
    ```

    -   **Word Tokenizer from NLTK**:

    ```
    'Emma', "'s", "power"
    ```

However, there is a special case for the processive term which may occur when dealing with long sequence tokenization. The "Emma's" will be fully tokenized when the newline character appears right after the term and also there exists some text after the character. For instance, "Emma's\n Some text" will have "Emma's" be fully extracted as a token.

```python
word_tokenize("Emma's\n I am a edge case") # ["Emma's", 'I', 'am', 'a', 'edge', 'case']
```

After digging into the source code, these two tokenizers were being defined with different class. The tokenizer in the corpus is actually a `WordPunctTokenizer` which will tokenize texts into sequences of alphabetic and non-alphabetic characters with simple regular expression and that is why it will consider punctuation as a single token. On the other hand, the `word_tokenize()` function is a tokenizer that based on `TreebankWordTokenizer`, which is using much more complicated regular expression for tokenization. And this is the reason why it can process complicated compounding words like possessive term.

By the way, the two ways of sentence tokenization has a slightly difference as well. The built-in `sents()` function is able to split titles, chapters, and volumes into different sentences, however, the `sent_tokenize` imported from NLTK consider them as a single sentence.

-   Raw Text: first sentence of the Gutenberg Book `"austen-emma.txt"` document

    -   **Built-in `sents()` callback in the corpus**:

        ```
        ['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']']
        ['VOLUME', 'I']
        ['CHAPTER', 'I']
        ['Emma', 'Woodhouse', ',', 'handsome', ',', 'clever', ',', 'and'....]
        ```

    -   **Sentence Tokenizer from NLTK**:

        ```
        ['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', 'VOLUME', 'I', 'CHAPTER', 'I', 'Emma', 'Woodhouse', ',', 'handsome', ',', 'clever', ',', 'and'....]
        ```
