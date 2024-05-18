"""
Search in rotated array - 
Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:
Input: nums = [1], target = 0
Output: -1
1. Just do bst (binary search tree on left and right. left and right being two arrays split at the pivot point)
First find the pivot point. 
How? Start with two pointers, if the left value is greater than nums[-1] then the pivot is to the right,
so l+=1
else r-=1
so keep going until the left is less than or equal to the right. 
Then once the mid which is the pivot is found, just apply bst on both sides.
"""


def search_rotated_array(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    # Find Pivot point -
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > nums[-1]:
            left = mid + 1
        else:
            right = mid - 1

    def bst(left, right, target):
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        return -1

    if (answer := bst(left, mid, target)) != -1:
        return answer
    return bst(mid, right, target)


nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_rotated_array(nums, target))
