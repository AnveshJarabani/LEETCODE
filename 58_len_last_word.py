"""
Given a string s consisting of words and spaces, return the length of the last word in the string.
A word is a maximal substring consisting of non-space characters only.
Example 1:
Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.
Example 2:
Input: s = "   fly me   to   the moon  "
Output: 4
Explanation: The last word is "moon" with length 4.
Example 3:
Input: s = "luffy is still joyboy"
Output: 6
Explanation: The last word is "joyboy" with length 6.
"""


def len_lastword(s: str) -> int:
    return len(s.split()[-1]) if s.split()[-1] else 0


print(
    len_lastword(s="Hello World"),
    'Explanation: The last word is "World" with length 5.',
)
print(
    len_lastword(s="   fly me   to   the moon  "),
    'Explanation: The last word is "moon" with length 4.',
)
print(
    len_lastword(s="luffy is still joyboy"),
    'Explanation: The last word is "joyboy" with length 6.',
)
