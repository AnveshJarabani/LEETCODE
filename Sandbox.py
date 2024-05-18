"""
median of two sorted arrays - 
o(m+n) solution - 
use the 
"""

nums1 = [1, 2, 7]
nums2 = [3, 4, 5, 6]


def median(nums1: list[int], nums2: list[int]) -> int:
    merged = []
    while nums1 and nums2:
        if nums1[0] <= nums2[0]:
            merged.append(nums1.pop(0))
        else:
            merged.append(nums2.pop(0))
    if nums1:
        merged.extend(nums1)
    elif nums2:
        merged.extend(nums2)
    n = len(merged) - 1
    if len(merged) % 2 == 0:
        return (merged[n // 2] + merged[n // 2 + 1]) / 2
    else:
        return merged[n // 2]


print(median(nums1, nums2))
