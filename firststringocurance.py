"""
Find the index of the first occurance in a string. 
Approach - 
Keep removing the first value of the current string 
until the string "startswith" the target / given string, then return  the index value
Keep track of the index value with a counter. 
"""


def first_occurance(haystack: str, needle: str) -> int:
    counter = 0
    while haystack:
        if haystack.startswith(needle):
            return counter
        haystack = haystack[1:]
        counter += 1
    return -1


haystack = "sadbutsad"
needle = "sad"
print(first_occurance(haystack, needle))
