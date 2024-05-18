'''
Approach - 
3Sum = 0  Problem --
Keep track of three pointers - start, mid, end
two loops - 
first loop - 
start moves from 0 to len(list) - 3 & mid and end moves from mid + 1 to end -1 and end moves from end to mid -1
second loop - 
mid and end travels towards each other while checking if the list[start] + list[mid] + list[end] ==0
in which case, it will add the start, mid, end values to the result subset list and append to the final list. 

edge cases - 
if the list is only 3 elements -> check if it's sum to zero then return the result. 

if the start and mid elements have same values after the existing element, 
keep traversing until the next value is not the same as the start / mid. 
'''
def three_sum(lst: list[int]) -> list[list[int]]:
    if len(lst) < 3: return []
    span = len(lst)
    lst.sort()
    result_list = []
    for start in range(span - 2):
        if start > 0 and lst[start] == lst[start - 1]: continue
        mid , end = start + 1, span - 1
        while mid < end:
            s = lst[start] + lst[mid] + lst[end]
            if  s < 0:
                mid+=1
            elif s > 0:
                end-=1
            else:
                result_list.append([lst[start], lst[mid], lst[end]])
                while  mid < end and lst[mid] == lst[mid+1]:
                    mid+=1
                while  mid < end and lst[end] == lst[end-1]:
                    end-=1
                mid+=1
                end-=1
    return result_list
#nums = [-1, 0, 1, 2, -1, -4]
nums = [1, -1 , -1 ,0]
print(three_sum(nums))
