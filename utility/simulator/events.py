from ..types import TaskID, TaskInfo, TaskState, Optional
from dataclasses import dataclass, field


@dataclass(slots=True)
class Event:
    func: str
    time: Optional[float] = None

    def __eq__(self, other):
        return self.func == other.func and self.time == other.time

    def __lt__(self, other):
        return self.time < other.time

    def __hash__(self):
        return hash((self.func, self.time))


@dataclass(slots=True)
class PhaseEvent(Event):
    max_tasks: int | None = None


@dataclass(slots=True)
class TaskEvent(Event):
    task: TaskID = field(default_factory=TaskID)


@dataclass(slots=True)
class Mapper(PhaseEvent):
    func: str = "mapper"


@dataclass(slots=True)
class Reserver(PhaseEvent):
    func: str = "reserver"


@dataclass(slots=True)
class Launcher(PhaseEvent):
    func: str = "launcher"


@dataclass(slots=True)
class TaskCompleted(TaskEvent):
    func: str = "complete_task"
