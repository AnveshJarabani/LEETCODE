"""
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:
Input: strs = [""]
Output: [[""]]
Example 3:
Input: strs = ["a"]
Output: [["a"]]

Approach - 
Create a dictionary with keys as sorted list of the str chars in each value, 
then for each value, if the sorted(str) is the key, then append to the value. 
return the dict values as a list.
"""


def group_anagrams(strs: list[str]) -> list[list[str]]:
    sorted_dict = (
        {}
    )  # initialize a dictionary to group the anagrams as values of the dict.
    for s in strs:
        key = "".join(sorted(s))
        sorted_dict.setdefault(key, []).append(key)
    return list(sorted_dict.values())


print(
    group_anagrams(strs=["eat", "tea", "tan", "ate", "nat", "bat"]),
    'Output: [["bat"],["nat","tan"],["ate","eat","tea"]]',
)
print(group_anagrams(strs=[""]), 'Output: [[""]]')
print(group_anagrams(strs=["a"]), 'Output: [["a"]]')
