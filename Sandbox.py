'''
Longest substring without repeating chars - 
Approach - 
hash set. 
why set? 
o(1) to find if a char exists. 
Keep track of the hash set seen. 
abcddabc
01234567
if max(max_length, )
'''



def longest_substring(s: str) -> int:
    start, max_substr = 0, 0 
    seen = set()
    for end in range(len(s)):
        while s[end] in seen:
            seen.remove(s[start])
            start+= 1
        seen.add(s[end])
        max_substr = max(max_substr, end - start + 1)
    return max_substr
s = "pwwkaslkfjasdf;lkew"
print(longest_substring(s))