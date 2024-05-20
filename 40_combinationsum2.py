"""
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.
Note: The solution set must not contain duplicate combinations.
Example 1:
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:
Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]

approach - 
use backtrack, but don't use the values already used if the values result in a good result/ appending to result.
"""


def combination_sum_uniques(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()

    def dfs(cur_sum: int, combination: list[int], idx: int):
        if cur_sum == 0:
            result.append(list(combination))
            return
        if cur_sum < 0:
            return
        for i in range(idx, len(candidates)):
            if i > idx and candidates[i] == candidates[i - 1]:
                continue
            val = candidates[i]
            dfs(cur_sum - val, combination + [val], i + 1)

    result = []
    dfs(target, [], 0)
    return result


print(combination_sum_uniques(candidates=[2, 5, 2, 1, 2], target=5))
print(combination_sum_uniques(candidates=[10, 1, 2, 7, 6, 1, 5], target=8))
