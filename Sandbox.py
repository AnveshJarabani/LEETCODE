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

Since the values are sorted, use bst to find the first value of target.
Then move left and right to see if that value exists if it does keep going until the extremes / indexes are found.
"""
def find_extremes(nums: list[int], target: int) -> list[int]:
    #Check if the value exists. Since the values are sorted, we can use bst
    def bst(left, right, target):
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left += 1
            else:
                right -=1
        return -1
    if (answer := bst(nums, target)) == -1:
        return [-1, -1]
    
        while 0 <= answer < len(nums):
            
            

nums = [5,7,7,8,8,10]
target = 8