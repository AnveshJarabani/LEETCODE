"""
Given an unsorted integer array nums. Return the smallest positive integer that is not present in nums.
You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.
Example 1:
Input: nums = [1,2,0]
Output: 3
Explanation: The numbers in the range [1,2] are all in the array.
Example 2:
Input: nums = [3,4,-1,1]
Output: 2
Explanation: 1 is in the array but 2 is missing.
Example 3:
Input: nums = [7,8,9,11,12]
Output: 1
Explanation: The smallest positive integer 1 is missing.
"""


def smallest_positive_integer(nums: list[int]) -> int:
    if not nums:
        return 1
    span = len(nums)
    for i in range(span):
        while 0 < nums[i] <= span and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(span):
        if nums[i] != i + 1:
            return i + 1
    return span + 1


print(smallest_positive_integer(nums=[1, 2, 0]), 3)
print(smallest_positive_integer(nums=[3, 4, -1, 1]), 2)
print(smallest_positive_integer(nums=[7, 8, 9, 11, 12]), 1)
