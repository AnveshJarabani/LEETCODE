"""
Longest Common Prefix - 
If the list is empty, return an empty string. There's no common prefix in an empty list.
If the list has only one string, return that string. The common prefix of a list with one string is the string itself.
Sort the list of strings. The common prefix of all strings will also be the common prefix of the first and last string in the sorted list.
Initialize the common prefix as an empty string.
"""


def prefix(strs: list[str]) -> str:
    if len(strs) == 0 or "" in strs:
        return ""
    if len(strs) == 1:
        return strs[0]
    pre = ""
    strs.sort()
    for i in range(len(strs[0])):
        if strs[0][i] == strs[len(strs) - 1][i]:
            pre += strs[0][i]
        else:
            return pre
    return pre


strs = ["flower", "flow", "flight"]
strs1 = ["dog", "racecar", "car"]
strs2 = ["", ""]
strs2 = ["ab", "a"]

print(prefix(strs))
print(prefix(strs1))
print(prefix(strs2))
