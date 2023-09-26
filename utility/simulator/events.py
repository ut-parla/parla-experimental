from ..types import TaskID, TaskInfo, TaskState
from dataclasses import dataclass


@dataclass(slots=True)
class Event():
    func: str = None


@dataclass(slots=True)
class PhaseEvent(Event):
    max_tasks: int | None = None


@dataclass(slots=True)
class TaskEvent(Event):
    task: TaskID | None = None


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
    pass
