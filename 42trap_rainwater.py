"""
Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.
Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9
Approach - 
two pointers. 
max rain water trapped at each index will be the min of current max. so keep doing this from both sides. to sum it up.
"""


def max_trapped_water(nums: list[int]) -> int:
    water = 0
    left, right = 0, len(nums) - 1
    max_left, max_right = 0, 0
    while left < right:
        if nums[left] <= nums[right]:
            max_left = max(max_left, nums[left])
            water += max_left - nums[left]
            left += 1
        else:
            max_right = max(max_right, nums[right])
            water += max_right - nums[right]
            right -= 1
    return water


print(max_trapped_water(nums=[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)
print(max_trapped_water(nums=[4, 2, 0, 3, 2, 5]), 9)
