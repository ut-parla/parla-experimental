#include "include/runtime.hpp"
#include "include/phases.hpp"

/**************************/
//Spawned Phase implementation

void SpawnedPhase::run(MappedPhase* ready){
    //NOT IMPLEMENTED
}


/**************************/
//Mapped Phase implementation

void MappedPhase::run(ReservedPhase* ready){
    //NOT IMPLEMENTED
}

/**************************/
//Reserved Phase implementation
void ReservedPhase::run(ReadyPhase* ready){
    //NOT IMPLEMENTED
}

/**************************/
//Ready Phase implementation

void ReadyPhase::enqueue(InnerTask* task){
    this->ready_tasks.push_back(task);
}

void ReadyPhase::enqueue(std::vector<InnerTask*>& tasks){
    //NOT IMPLEMENTED
}

int ReadyPhase::get_count(){
    return this->ready_tasks.atomic_size();
}

bool ReadyPhase::condition(){
    //NOT IMPLEMENTED
    return true;
}

void ReadyPhase::run(LauncherPhase* launcher){

    //TODO: Refactor this so its readable without as many nested conditionals

    //This is a critical region
    //Mutex needed only if it is called from multiple threads (not just scheduler thread)

    //Assumptions:
    //Scheduler resources are ONLY decreased here
    //Available workers are ONLY decreased here

    //Assumptions to revisit:
    //Ready tasks must be launched in order
    //If the task at the head cannot be launched (eg. not enough resources, no available workers)
    //, then no other tasks can be launched
    //TODO: Revisit this design decision

    //TODO: Currently this drains the whole queue. Use Ready::condition() to set a better policy?
    //TODO: This stops at a single failure.
    //TODO: Maybe failure of a phase means it should wait on events to try again. Instead of just spinning?

    this->mtx.lock();
    
    bool has_task = true;

    while(has_task){

        has_task = this->get_count() > 0;

        if(has_task){
            auto task = this->ready_tasks.front();
            bool has_resources = scheduler->resources->check_greater(task->resources);

            if(has_resources){

                bool has_thread = scheduler->workers.get_num_available_workers() > 0;

                if (has_thread){

                    InnerTask* task = this->ready_tasks.front_and_pop();
                    InnerWorker* worker = scheduler->workers.dequeue_worker();

                    launcher->enqueue(task, worker);

                    this->status.update(Ready::success);
                }
                else{
                    this->status.update(Ready::worker_miss);
                    break; //No more workers available
                }
            }
            else{
                this->status.update(Ready::resource_miss);
                break; //No more resources available
            }
        }
        else{
            this->status.update(Ready::task_miss);
            break; //No more tasks available
        }
    }

    this->mtx.unlock();

}

/**************************/
//Launcher Phase implementation

void LauncherPhase::enqueue(InnerTask *task, InnerWorker *worker){
    //Immediately launch task

    void* py_task = task->py_task;
    void* py_worker = worker->py_worker;

    //Acquire GIL to assign task to worker and notify worker through python callback
    launch_task_callback(this->launch_callback, py_scheduler, py_task, py_worker);

    //TODO: Replace this with a condition variable update on the InnerWorker
    //      Then we won't need to acquire the GIL to launch from the scheduler thread

    this->num_running_tasks++;
}


void LauncherPhase::run(){
    //NOT IMPLEMENTED
}