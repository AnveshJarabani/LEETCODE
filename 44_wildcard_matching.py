"""
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
Example 2:
Input: s = "aa", p = "*"
Output: true
Explanation: '*' matches any sequence.
Example 3:
Input: s = "cb", p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
"""


def wildcard_match(s: str, p: str) -> bool:
    dp = [[False for _ in range(len(p) + 1)] for _ in range(len(s) + 1)]
    dp[0][0] = True
    for j in range(len(p)):
        if p[j] == "*":
            dp[0][j + 1] = dp[0][j]
    for i in range(len(s)):
        for j in range(len(p)):
            if p[j] == s[i] or p[j] == "?":
                dp[i + 1][j + 1] = dp[i][j]
            elif p[j] == "*":
                dp[i + 1][j + 1] = dp[i][j + 1] or dp[i + 1][j]
    return dp[len(s)][len(p)]


print(wildcard_match(s="aa", p="a"), "false")
print(wildcard_match(s="aa", p="*"), "true")
print(wildcard_match(s="cb", p="?a"), "false")
print(wildcard_match(s="adceb", p="*a*b"), "true")
