"""
26 - Remove duplicates from sorted array. 
keep a counter. 
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
two pointers left and right. 
Keep moving the right pointer until the nums[left] != nums[right]
then just swap values. 
Then go left +=1 and right+=1
do the same thing until the right value hits the last element in the array.
"""


def swap_duplicates(nums: list[int]) -> list[int]:
    if not nums:
        return 0
    left, right = 0, 1
    span = len(nums)
    while right < len(nums):
        if nums[left] != nums[right]:
            left += 1
            nums[left] = nums[right]
        right += 1
    return nums


nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
print(swap_duplicates(nums))
