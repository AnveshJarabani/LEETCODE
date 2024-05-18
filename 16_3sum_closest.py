"""
3sum closest - 
Start with a target value of infinity. 
Then keep the delta of the current 3sum with target and if's less than the current, udpate the 3sum value
keep going and return the value. 
Use two loops. with 3 pointers similar to 3sum problem. 
"""


def threesum_closest(nums: list[int], target: int) -> int:
    if len(nums) == 3:
        return sum(nums)
    closest_sum = float("inf")
    span = len(nums)
    nums.sort()
    for start in range(span - 2):
        mid, end = start + 1, span - 1
        while mid < end:
            s = nums[start] + nums[mid] + nums[end]
            if abs(target - s) < abs(closest_sum - target):
                closest_sum = s
            if s < target:
                mid += 1
            elif s > target:
                end -= 1
            else:
                return s
    return closest_sum


nums = [4, 0, 5, -5, 3, 3, 0, -4, -5]
target = -2
print(threesum_closest(nums, target))
