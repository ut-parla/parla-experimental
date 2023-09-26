import heapq
from ..types import TaskID
from .events import Event
from typing import Tuple


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
    # TODO(hc): better name?
    def __init__(self, q: PriorityQueue, maxiter: int = None):
        self.q = q
        self.iter = 0
        self.maxiter = maxiter

    def __iter__(self):
        while True:
            try:
                if len(self.q) == 0:
                    break
                if self.maxiter is not None and self.iter >= self.maxiter:
                    break

                self.iter += 1
                yield self.q.get()
            except IndexError:
                break


class TaskQueue(PriorityQueue):

    def put(self, task: TaskID):
        super().put((task.info.order, task))

    def get(self) -> TaskID:
        pair = super().queue.get()
        if pair:
            return pair[1]
        else:
            return None

    def peek(self) -> TaskID:
        pair = super().queue.peek()
        if pair:
            return pair[1]
        else:
            return None


class GetNextTask(QueueDrainer):
    pass


class EventQueue(PriorityQueue):

    def put(self, event: Event, completion_time: float):
        super().put((completion_time, event))

    def get(self) -> Tuple[float, Event]:
        return super().queue.get()

    def peek(self) -> Event:
        pair = super().queue.peek()
        if pair:
            return pair[1]
        else:
            return None


class GetNextEvent(QueueDrainer):
    pass
