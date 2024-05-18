'''
Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

Solution -
Have two pointers. Starting at either sides. Now, if the sum of the two pointers is greater than the target, move the right pointer to the left. If the sum is less than the target, move the left pointer to the right. If the sum is equal to the target, return the indices of the two pointers.
Sorting happens in O(nlogn) time. The two pointer approach happens in O(n) time. So, the overall time complexity is O(nlogn) + O(n) = O(nlogn).
'''

def twosum(nums,target):
    nums.sort()
    i,j = 0, len(nums)-1
    if nums[i]+nums[j] > target:
        j-=1
    elif nums[i]+nums[j] < target:
        i+=1
    else:
        return [nums[i], nums[j]]


nums = [2,7,11,15]
target = 9
print(twosum(nums,target))


