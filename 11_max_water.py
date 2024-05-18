"""
Approach - 
start from 0 and len(list) - 1 with two pointers. 
keep traversing both left and right only if the next pointer height (list[l/r] is greater than the existing height)
then find the difference in the pointers (width) and multiply by smallest height

"""


def max_water(lst: list[int]) -> int:
    l, r = 0, len(lst) - 1
    max_water = 0
    while l < r:
        max_water = max(max_water, min(lst[l], lst[r]) * (r - l))
        if lst[l] < lst[r]:
            l += 1
        else:
            r -= 1
    return max_water


height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(max_water(height))
