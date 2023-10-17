from utility.simulator.queue import *
from rich import print

object_list = [0, 1, 2, 3, 4]
object_list2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
object_list3 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

objects = [object_list, object_list2, object_list3]

queues1 = [PriorityQueue() for _ in range(len(objects))]

for i, q in enumerate(queues1):
    for obj in objects[i]:
        q.put((obj, obj))

queue_dict1 = dict(zip([f"queue_{i}" for i in range(len(queues1))], queues1))

object_list = ["a", "b", "c", "d", "e"]
object_list2 = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"]
object_list3 = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj"]

objects = [object_list, object_list2, object_list3]

queues2 = [PriorityQueue() for _ in range(len(objects))]

for i, q in enumerate(queues2):
    for obj in objects[i]:
        q.put((obj, obj))

queue_dict2 = dict(zip([f"queue_{i}" for i in range(len(queues2))], queues2))

queue_dict = queue_dict1  # {"A": queue_dict1, "B": queue_dict2}


print(queue_dict)

NextObject = MultiQueueIterator(queue_dict, peek=True)
for i, o in enumerate(NextObject):
    print(o)
    print(NextObject.get_current_keys())

    if i == 0 or i == 2:
        NextObject.fail()
    else:
        NextObject.success()

# NextObject = QueueIterator(q, peek=True)
# for i in NextObject:
#     print(i)

#     NextObject.pop()

# print(NextObject)
