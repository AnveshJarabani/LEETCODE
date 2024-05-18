from typing import List


# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#         s=[]
#         p1,p2=0,0
#         while p1<len(nums1) and p2<len(nums2):
#             if nums1[p1]<nums2[p2]:
#                 s.append(nums1[p1])
#                 p1+=1
#             else:
#                 s.append(nums2[p2])
#                 p2+=1
#         if p1==len(nums1):
#             s.extend(nums2[p2:])
#         else:
#             s.extend(nums1[p1:])
#         n=len(s)-1
#         if (n + 1)% 2 == 0:
#             return (s[n//2]+s[(n//2)+1])/2
#         else:
#             return s[n//2]
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # nums1 = [1, 2, 7]
        # nums2 = [3, 4, 5, 6]

        # def median(nums1: list[int], nums2: list[int]) -> int:
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


nums1 = []
nums2 = [1]
print(Solution().findMedianSortedArrays(nums1, nums2))
