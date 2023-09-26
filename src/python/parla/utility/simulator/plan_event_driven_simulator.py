class Mapper:
  def get_duration():
  def run(self):
    new_event = []
    any_work = True
    is_mapped = False
    while any_work:
      task, any_work = self.check_and_pop_spawned_task()
      is_mapped = self.map_task(task)
    if is_mapped:
      new_event.append(Event(self.resource_reserver))
    return new_event

class ResReserver:
  def get_duration():
  def run(self):
    new_event = []
    any_work = True
    is_reserved = False
    while any_work:
      task, any_work = self.checK_and_pop_reservable_task()
      is_reserved = self.reserce_resource(task)
    # Launcher can be added by "TaskCleaner"
    return new_event

class Launcher:
  def get_duration():
  def run(self):
    ...

class TaskCleaner:
  def get_duration():
  def run(self):
    ...

class Event:
  def __init__(self, _event):
    self.callback = _event.run
    self.duration = _event.get_duration()

  def run_callback(self):
    return self.callback()

  def advance_time(self, current_time):
    return current_time + self.duration


class Scheduler:

  def __init__(self):
    self.mapper = Mapper()
    self.resource_reserver = ResReserver()
    self.launcher = Launcher()
    self.task_cleaner = TaskCleaner()
    self.current_time = 0

  def process_event(self, event):
    new_events = event.run_callback()
    self.current_time = event.advance_time(self.current_time)
    self.event_enqueue.add_events(new_events)

  def run(self):
    
    # This also increases the number of remaining tasks.
    self.fill_spawned_tasks()
    # At the beginning, push all phase events.
    self.event_queue.add_event(
        Event(self.mapper))
    self.event_queue.add_event(
        Event(self.resource_reserver))
    self.event_queue.add_event(
        Event(self.launcher))
    self.event_queue.add_event(
        Event(self.task_cleaner))
    while self.num_remaining_tasks > 0:
      event = self.pop_event()
      self.process_event(event)
      
      
      
      
      
