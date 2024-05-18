"""
Susbtring with Concatenation of All Words - 
1. All words in the words list are of same length.
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Create a wordmap with Counter(words)
Now, since the words are of same size, 
the combination of words should each be in the string s. 
so we need to check for occurance of each len(words)*lenght of each word
in each of the s in steps. If it does occur, then we record the starting index of the occurance
then move to the next step. 
So it's a sliding window of lenght len(words)*length of each word. 
But within each window, we need to confirm if every occurance of the wordmap exists.
if it doens't we break out of the look that is checking if there is a match.
We also break out if the occurance of the word in the current window exceeds the occurance in words list.
"""

s = "barfoothefoobarman"
words = ["foo", "bar"]
from collections import Counter


def substring_occurances(s: str, words: list[str]) -> list[int]:
    word_count = Counter(words)
    word_len = len(words[0])
    window_span = len(words[0]) * len(words)
    res = []
    if window_span > len(s):
        return []
    for i in range(len(s) - window_span + 1):
        seen_map = Counter()
        for j in range(i, i + window_span, word_len):
            current_word = s[j : j + word_len]
            if current_word not in word_count:
                break
            if current_word in word_count:
                seen_map[current_word] += 1
            if seen_map[current_word] > word_count[current_word]:
                break
        else:
            res.append(i)
    return res


print(substring_occurances(s, words))
