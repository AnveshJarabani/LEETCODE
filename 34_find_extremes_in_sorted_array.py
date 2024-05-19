"""
Given an array of integers nums sorted in non-decreasing order, 
find the starting and ending position of a given target value.
If target is not found in the array, return [-1, -1].
You must write an algorithm with O(log n) runtime complexity.
Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
Example 3:
Input: nums = [], target = 0
Output: [-1,-1]

Approach -

Since the values are sorted,
use two bsts - one to find left extreme and one to find right extreme. 
then return the result. 
"""


def find_extremes(nums: list[int], target: int) -> list[int]:
    if not nums:
        return [-1, -1]

    def left_extreme() -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left if left < len(nums) and nums[left] == target else -1

    def right_extreme() -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right if right >= 0 and nums[right] == target else -1

    if left_extreme() == -1:
        return [-1, -1]
    return [left_extreme(), right_extreme()]


# nums = [5, 7, 7, 8, 8, 10]
nums = [2, 2]
target = 8
print(find_extremes(nums, target))
