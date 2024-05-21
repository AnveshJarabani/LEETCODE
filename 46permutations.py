"""
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:
Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:
Input: nums = [1]
Output: [[1]]

Approach - 
use backtrack algo to find all combinations and just return
"""


def permute(nums: list[int]) -> list[list[int]]:
    def backtrack(nums, path):
        if not nums:
            result.append(path)
        for i in range(len(nums)):
            backtrack(nums[:i] + nums[i + 1 :], path + [nums[i]])

    result = []
    backtrack(nums, [])
    return result


print(
    permute(nums=[1, 2, 3]), " res = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]"
)
print(permute(nums=[0, 1]), "[[0,1],[1,0]]")
print(permute(nums=[1]), "[[1]]")
