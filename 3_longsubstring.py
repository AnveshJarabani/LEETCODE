""" 
This is a classic sliding window problem. 
When we need to find "longest" something, try to use sliding window. 
you start with no set and then keep updating the set as you traverse a given string/list
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start, max_substr = 0, 0
        seen = set()
        for end in range(len(s)):
            while s[end] in seen:
                seen.remove(s[start])
                start += 1
            seen.add(s[end])
            max_substr = max(max_substr, end - start + 1)
        return max_substr
