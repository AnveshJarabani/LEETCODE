"""
Merge K sorted lists - 
Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:
Input: lists = []
Output: []

Appraoch - 
Build a heap with tuples. 
Tuple will have - 
(1.the value of the first node element , 2.the current node itself)
This is how the heap priority is defined by the value of the first elemnt.
so the heap is of size k. k being the number of listnodes in the input list.
so from that point, we can heappop each element and build our result list. 
As we build our result list, we can keep updating the heap with the next node,
and it's value if it exists.
We continue this until all the nodes don't have anymore next nodes at whichpoint
we just return the result linkedlist / listnode head.
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def build_nodelist(lst: list):
        dummy = ListNode()
        current_node = dummy
        for i in lst:
            current_node.next = ListNode(val=i)
            current_node = current_node.next
        return dummy.next

    @staticmethod
    def print_nodelist(node):
        lst = []
        while node:
            lst.append(str(node.val))
            node = node.next
        return "->".join(lst)


import heapq


def merge_k_sorted_lists(lsts: list[ListNode]) -> ListNode:
    heap = []
    count = 0
    for node in lsts:
        if node:
            heap.append((node.val, count, node))
            count += 1
    heapq.heapify(heap)
    head = ListNode()
    current_node = head
    while heap:
        val, _, node = heapq.heappop(heap)
        current_node.next = ListNode(val=val)
        current_node = current_node.next
        if node.next:
            heapq.heappush(heap, (node.next.val, count, node.next))
            count += 1
    return head.next


lst1 = [1, 4, 5]
lst2 = [1, 3, 4]
lst3 = [2, 6]
lsts = [
    ListNode.build_nodelist(lst1),
    ListNode.build_nodelist(lst2),
    ListNode.build_nodelist(lst3),
]
# print(ListNode.print_nodelist(lst1))
res_node = merge_k_sorted_lists(lsts)
print(ListNode.print_nodelist(res_node))
