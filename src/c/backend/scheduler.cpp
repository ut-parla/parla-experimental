#include "include/runtime.hpp"
#include "include/phases.hpp"

//Worker Implementation

//nothing here, currently header only

//WorkerPool Implementation

template<typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::enqueue_worker(InnerWorker* worker){
    this->active_workers.push_back(worker);
}

template<typename AllWorkers_t, typename ActiveWorkers_t>
InnerWorker* WorkerPool<AllWorkers_t, ActiveWorkers_t>::dequeue_worker(){
    InnerWorker* worker = this->active_workers.back_and_pop();
    return worker;
}

template<typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::add_worker(InnerWorker* worker){
    this->all_workers.push_back(worker);
    assert(this->all_workers.size() <= this->max_workers);
}

template<typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::get_num_available_workers(){
    return this->active_workers.size();
}

template<typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::get_num_workers(){
    return this->max_workers;
}

template<typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::set_num_workers(int nworkers){
    this->max_workers = nworkers;
}

template class WorkerPool<WorkerQueue, WorkerQueue>;


//Scheduler Implementation

InnerScheduler::InnerScheduler(){

    // A dummy task count is used to keep the scheduler alive. 
    // NOTE: At least one task must be added to the scheduler by the main thread, otherwise the runtime will finish immediately
    this->increase_num_active_tasks();

    this->workers.set_num_workers(1);

    //Initialize the phases
    this->ready_phase = new ReadyPhase(this);
    this->launcher = new LauncherPhase(this);
    this->resources = new InnerResourcePool<float>();
    //TODO: Clean these up
}

void InnerScheduler::set_num_workers(int nworkers){
    this->workers.set_num_workers(nworkers);
}

void InnerScheduler::set_resources(std::string resource_name, float resource_value){
    this->resources->set(resource_name, resource_value);
}

void InnerScheduler::set_py_scheduler(void *py_scheduler){
    this->py_scheduler = py_scheduler;
}

void InnerScheduler::set_stop_callback(stopfunc_t stop_callback){
    this->stop_callback = stop_callback;
}

void InnerScheduler::set_launch_callback(launchfunc_t launch_callback){
    this->launcher->set_launch_callback(launch_callback, this->py_scheduler);
}


void InnerScheduler::run(){
    unsigned long long iteration_count = 0;
    while(this->should_run){
        auto status= this->activate();
        if (this->sleep_flag){
            std::this_thread::sleep_for(std::chrono::milliseconds(this->sleep_time));
        }
    }
}

void InnerScheduler::stop(){
    this->should_run = false;
    launch_stop_callback(this->stop_callback, this->py_scheduler);
}

Scheduler::Status InnerScheduler::activate(){
    //this->spawned_phase->run(this->mapped_phase);
    //this->mapped_phase->run(this->reserved_phase);
    //this->reserved_phase->run(this->ready_phase);
    this->ready_phase->run(this->launcher);
    //this->launcher->run();

    return this->status;
}

void InnerScheduler::activate_wrapper(){
    this->activate();
}

void InnerScheduler::enqueue_task(InnerTask* task){
    //TODO: Change this to appropriate phase as it becomes implemented
    this->ready_phase->enqueue(task);
}

void InnerScheduler::enqueue_tasks(std::vector<InnerTask*>& tasks){
    this->ready_phase->enqueue(tasks);
}

void InnerScheduler::add_worker(InnerWorker* worker){
    this->workers.add_worker(worker);
}

void InnerScheduler::enqueue_worker(InnerWorker* worker){
    this->workers.enqueue_worker(worker);
}

void InnerScheduler::task_cleanup(InnerWorker *worker, InnerTask *task, int state){
    /* Task::States are: spawned, mapped, reserved, ready, running, complete */

    //This will be called by EVERY thread that finishes a task
    //Everything in here needs to be thread safe

    //TODO: for runahead, we need to do this AFTER the task body is complete
    //      Need to add back to the pool after notify_dependents
    //      Movin this below but leaving my original placement here for now
    //this->resources->increase(task->resources);
    //this->launcher->num_running_tasks--;
    //this->workers.enqueue_worker(worker);

    if(state == Task::spawned){
        //Do continuation handling 
    }

    if (state == Task::complete){
        //When a task completes we need to notify all of its dependents
        //and enqueue them if they are ready

        auto& enqueue_buffer = worker->enqueue_buffer;
        task->notify_dependents(enqueue_buffer);
        this->enqueue_tasks(enqueue_buffer);

        //TODO: Wait on CUDA events here for runahead
        //TODO: Should probably split this into two functions here
        //      Then they can be called separately in Python

        //We also need to decrease the number of active tasks
        //If this is the last active task, the scheduler is stopped
        this->decrease_num_active_tasks();
    }

    //TODO: for runahead, we need to do this AFTER the task body is complete
    //      Need to add back to the pool after notify_dependents
    this->resources->increase(task->resources);
    this->launcher->num_running_tasks--;
    this->workers.enqueue_worker(worker);

    //NOTE: Task::complete is NOT equivalent to task->complete
    // Task->complete:
    //      - only signals that notify_dependencies has been completed
    //      - Has a valid state for use in dependency handling

    // Task::complete:
    //     - signals that the task has finished everything

    task->set_state(state);
}

int InnerScheduler::get_num_active_tasks(){
    return this->num_active_tasks;
}

void InnerScheduler::increase_num_active_tasks(){
    this->num_active_tasks++;
}

void InnerScheduler::decrease_num_active_tasks(){
    int count = this->num_active_tasks.fetch_sub(1) - 1;

    if (count == 0){
        this->stop();
    }
}

int InnerScheduler::get_num_running_tasks(){
    return this->launcher->num_running_tasks;
}

int InnerScheduler::get_num_ready_tasks(){
    return this->ready_phase->get_count();
}











