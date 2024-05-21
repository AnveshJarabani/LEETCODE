"""
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
Each element nums[i] represents the maximum length of a forward jump from index i. 
In other words, if you are at nums[i], you can jump to any nums[i + j] where:
0 <= j <= nums[i] and
i + j < n
Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].
Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:
Input: nums = [2,3,0,1,4]
Output: 2
use variables - current_jumps, jumps, farthest. 
So whenever the current_jumps is the most we can reach current index, then we increment the jump by one and update the 
current_jump with the farthest value. 
"""


def jump_game(nums: list[int]) -> int:
    jumps, current_jumps, farthest = 0, 0, 0
    for i in range(len(nums) - 1):
        farthest = max(i + nums[i], farthest)
        if current_jumps == i:
            jumps += 1
            current_jumps = farthest
    return jumps


print(jump_game(nums=[2, 3, 1, 1, 4]), 2)
print(jump_game(nums=[2, 3, 0, 1, 4]), 2)
