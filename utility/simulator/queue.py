import heapq
from ..types import TaskID, Time
from .events import Event
from typing import Tuple, TypeVar, Optional, Self
from .task import SimulatedTask

# (completion_time, event)
EventPair = Tuple[Time, Event]
# (order, task)
TaskPair = Tuple[int, TaskID]


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, item):
        heapq.heappush(self.queue, item)

    def get(self):
        return heapq.heappop(self.queue)

    def peek(self):
        return self.queue[0]

    def __len__(self):
        return len(self.queue)


class QueueDrainer(object):
    def __init__(self, q: PriorityQueue, maxiter: int = -1):
        self.q = q
        self.iter = 0
        self.maxiter = maxiter

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.q) == 0:
            raise StopIteration

        if self.maxiter != -1 and self.iter >= self.maxiter:
            self.iter = 0
            raise StopIteration

        self.iter += 1
        return self.q.get()


class TaskQueue(PriorityQueue):
    def put(self, task: SimulatedTask):
        super().put((task.info.order, task.name))

    def get(self) -> Optional[TaskID]:
        pair = super().get()
        if pair:
            return pair[1]
        else:
            return None

    def peek(self) -> Optional[TaskID]:
        pair = super().peek()
        if pair:
            return pair[1]
        else:
            return None


class GetNextTask(QueueDrainer):
    def __init__(self, q: TaskQueue, maxiter: int = -1):
        super().__init__(q, maxiter)

    def __iter__(self) -> Self:
        return super().__iter__()

    def __next__(self) -> TaskPair:
        return super().__next__()


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


class GetNextEvent(QueueDrainer):
    def __init__(self, q: EventQueue, maxiter: int = -1):
        super().__init__(q, maxiter)

    def __iter__(self) -> Self:
        return super().__iter__()

    def __next__(self) -> EventPair:
        return super().__next__()
