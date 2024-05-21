"""
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
Example 1:
Input: nums = [1,1,2]
Output: [[1,1,2],[1,2,1],[2,1,1]]
Example 2:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""

from collections import Counter


def unique_permutations(nums: list[int]) -> list[list[int]]:
    hash_map = Counter(nums)

    def dfs():
        if len(temp) == len(nums):
            res.append(temp.copy())
            return
        for n in hash_map:
            if hash_map[n] > 0:
                temp.append(n)
                hash_map[n] -= 1
                dfs()
                hash_map[n] += 1
                temp.pop()

    res, temp = [], []
    dfs()
    return res


print(unique_permutations(nums=[1, 1, 2]), "res = [[1,1,2],[1,2,1],[2,1,1]]")
print(
    unique_permutations(nums=[1, 2, 3]),
    "res = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]",
)
