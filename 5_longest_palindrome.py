"""
Approach - 
o(n^2) solution.
For each value in the string, 
travel left and right until it's not a palindrome or if it's bigger than the current palindrome.
Just handle things for both cases - even and odd lenghted palindromic substrings. 
"""


def longest_palindrome(s: str) -> str:
    def substr(s: str, substring: str, l: int, r: int) -> str:
        while l >= 0 and r < span and s[l] == s[r]:
            if r - l + 1 > len(substring):
                substring = s[l : r + 1]
            l -= 1
            r += 1
        return substring

    span = len(s)
    substring = ""
    for i in range(span):
        # Handling even lengthed substrings -
        substring = substr(s, substring, i, i)
        # Handling even lengthed substrings -
        substring = substr(s, substring, i, i + 1)
    return substring


s = "cbbbbbddask"
print(longest_palindrome(s))
