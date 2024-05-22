"""
Given an integer array nums, find the subarray with the largest sum, and return its sum.
Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
Example 2:
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.
Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
Approach - 
Dynamic programming implementation 
"""
def max_sum(nums: list[int]) -> int:
    cur_sum =  max_sum = nums[0]
    for num in nums[1:]:
        cur_sum = max(cur_sum+num,num) # choose either to keep the current sum so far or add current num whicheever is bigger
        max_sum = max(max_sum, cur_sum) # update the max sum with current_sum consideration for max value
    return max_sum

print(max_sum(nums = [-2,1,-3,4,-1,2,1,-5,4]),
"Output: 6")
print(max_sum(nums = [1]),
"Output: 1")
print(max_sum(nums = [5,4,-1,7,8]),
"Output: 23")