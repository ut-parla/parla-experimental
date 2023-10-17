from __future__ import annotations

import heapq
from ..types import TaskID, Time
from .events import Event
from typing import Tuple, TypeVar, Optional, Self, Dict, Iterable

# (completion_time, event)
EventPair = Tuple[Time, Event]
# (order, task)
TaskPair = Tuple[int, TaskID]


def length(q: Iterable | PriorityQueue) -> int:
    if isinstance(q, PriorityQueue):
        return len(q)
    elif isinstance(q, Iterable):
        if isinstance(q, Dict):
            return sum([length(qi) for qi in q.values()])
        else:
            return sum([length(qi) for qi in q])
    else:
        raise TypeError(f"Cannot get length of {q}")


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.tiebreaker = 0

    def put(self, item):
        self.tiebreaker += 1
        item = (item[0], self.tiebreaker, item[1])
        heapq.heappush(self.queue, item)

    def get(self):
        result_with_tiebreaker = heapq.heappop(self.queue)
        return (result_with_tiebreaker[0], result_with_tiebreaker[2])

    def peek(self):
        result_with_tiebreaker = self.queue[0]
        return (result_with_tiebreaker[0], result_with_tiebreaker[2])

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return f"PriorityQueue({self.queue})"

    def __repr__(self):
        return self.__str__()

    def remove(self, item):
        self.queue.remove(item)
        heapq.heapify(self.queue)


class QueueIterator(object):
    def __init__(self, q: PriorityQueue, maxiter: int = -1, peek: bool = True):
        self.q = q
        self.iter = 0
        self.maxiter = maxiter
        self.peek = peek
        self.success_count = 0

        self.failed_count = 0
        self.fail_threshold = 1

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.q) == 0:
            self.iter = 0
            self.failed_count = 0
            self.success_count = 0
            # print("Queue is empty!")
            raise StopIteration

        if self.failed_count >= self.fail_threshold:
            self.iter = 0
            self.failed_count = 0
            self.success_count = 0
            # print("Queue failed!", self.q)
            raise StopIteration

        if self.maxiter != -1 and self.iter >= self.maxiter:
            self.iter = 0
            self.failed_count = 0
            self.success_count = 0
            # print("Queue reached maxiter!")
            raise StopIteration

        self.iter += 1

        if self.peek:
            return self.q.peek()
        else:
            self.success_count += 1
            return self.q.get()

    def success(self):
        self.success_count += 1
        return self.q.get()

    def fail(self):
        self.failed_count += 1
        return self.q.peek()

    def __len__(self):
        return len(self.q)

    def __str__(self):
        return f"QueueIterator({self.q}, ITER={self.iter}, MAXITER={self.maxiter}, PEEK={self.peek})"

    def __repr__(self):
        return self.__str__()


class MultiQueueIterator(object):
    @staticmethod
    def make_iterator(q: PriorityQueue | Dict, maxiter: int = -1, peek: bool = True):
        if isinstance(q, PriorityQueue):
            return QueueIterator(q, maxiter, peek)
        elif isinstance(q, Dict):
            return MultiQueueIterator(q, maxiter, peek)

    def __init__(self, queues: Dict, maxiter: int = -1, peek: bool = True):
        self.dict = queues
        self.keys = list(queues.keys())
        self.iterators = [
            MultiQueueIterator.make_iterator(q, -1, peek) for q in queues.values()
        ]

        self.iter = 0
        self.maxiter = maxiter
        self.peek = peek

        self.current_idx = 0
        self.success_count = 0
        self.failed_count = 0

    def __iter__(self):
        return self

    @property
    def current_iterator(self):
        return self.iterators[self.current_idx]

    @property
    def current_key(self):
        return self.keys[self.current_idx]

    def get_current_keys(self):
        current_key = self.current_key
        if isinstance(self.current_iterator, MultiQueueIterator):
            return current_key, self.current_iterator.get_current_keys()
        else:
            return current_key

    def __len__(self):
        return sum([len(q) for q in self.iterators])

    def __next__(self):
        if self.maxiter != -1 and self.iter >= self.maxiter:
            self.iter = 0
            self.failed_count = 0
            raise StopIteration

        self.iter += 1
        next = None
        while len(self.iterators) > 0 and next is None:
            try:
                next = self.iterators[self.current_idx].__next__()

            except StopIteration:
                self.iterators.pop(self.current_idx)
                self.keys.pop(self.current_idx)
                if len(self.iterators) > 0:
                    self.current_idx = self.iter % len(self.iterators)

        if len(self.iterators) == 0 or next is None:
            raise StopIteration

        if not self.peek:
            self.current_idx = self.iter % len(self.iterators)

        return next

    def success(self):
        popped = self.iterators[self.current_idx].success()
        self.success_count += 1
        self.current_idx = self.iter % len(self.iterators)
        return popped

    def fail(self):
        peeked = self.iterators[self.current_idx].fail()
        self.failed_count += 1
        self.current_idx = self.iter % len(self.iterators)
        return peeked

    def __str__(self):
        return f"MultiQueueIterator({self.dict}, ITER={self.iter}, MAXITER={self.maxiter}, PEEK={self.peek})"


class TaskQueue(PriorityQueue):
    def put(self, task: "SimulatedTask"):
        super().put((task.info.order, task.name))

    def put_id(self, task_id: TaskID, priority: int | Time):
        super().put((priority, task_id))

    def get(self) -> Optional[TaskPair]:
        pair = super().get()
        return pair

    def peek(self) -> Optional[TaskPair]:
        pair = super().peek()
        return pair

    def __str__(self):
        return f"TaskQueue({self.queue})"


class TaskIterator(QueueIterator):
    def __init__(self, q: TaskQueue, maxiter: int = -1, peek: bool = True):
        super().__init__(q, maxiter, peek)

    def __iter__(self) -> Self:
        return super().__iter__()

    def __next__(self) -> TaskPair:
        return super().__next__()

    def success(self) -> TaskPair:
        return super().success()

    def fail(self) -> TaskPair:
        return super().fail()

    def __str__(self):
        return f"TaskIterator({self.q}, ITER={self.iter}, MAXITER={self.maxiter}, PEEK={self.peek})"


class MultiTaskIterator(MultiQueueIterator):
    def __init__(self, queues: Dict, maxiter: int = -1, peek: bool = True):
        super().__init__(queues, maxiter, peek)

    def __iter__(self) -> Self:
        return super().__iter__()

    def __next__(self) -> TaskPair:
        return super().__next__()

    def success(self) -> TaskPair:
        return super().success()

    def fail(self) -> TaskPair:
        return super().fail()

    def __str__(self):
        return f"MultiTaskIterator({self.dict}, ITER={self.iter}, MAXITER={self.maxiter}, PEEK={self.peek})"


class EventQueue(PriorityQueue):
    def put(self, event: Event, completion_time: Time):
        super().put((completion_time, event))

    def get(self) -> Optional[EventPair]:
        return super().get()

    def peek(self) -> Optional[EventPair]:
        pair = super().peek()
        if pair:
            return pair
        else:
            return None

    def __str__(self):
        return f"EventQueue({self.queue})"


class EventIterator(QueueIterator):
    def __init__(self, q: EventQueue, maxiter: int = -1, peek: bool = True):
        super().__init__(q, maxiter, peek)

    def __iter__(self) -> Self:
        return super().__iter__()

    def __next__(self) -> EventPair:
        return super().__next__()

    def pop(self) -> EventPair:
        return super().success()

    def __str__(self):
        return f"EventIterator({self.q}, ITER={self.iter}, MAXITER={self.maxiter}, PEEK={self.peek})"
