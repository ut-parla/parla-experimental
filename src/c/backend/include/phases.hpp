#pragma once
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "containers.hpp"
#include "runtime.hpp"
#include <string.h>

//TODO(will): This is a mock implementation of the phases. They may need to be rearranged, renamed, etc. Just a baseline organization.

namespace Spawned{
    enum State { failure, task_miss, success };
    class Status{
      private:
        const static int size = 3;
      public:
        int status[size];

        void reset() {
          for (int i = 0; i < size; ++i) {
            this->status[i] = 0;
          }
        }

        void set(int index, int value) {
          this->status[index] = value;
        }

        int get(int index) {
          return this->status[index];
        }

        void update(State state) {
          this->status[state]++;
        }

        void print() {
          std::cout << "Ready Status: (";
          for (int i = 0; i < size; ++i) {
            std::cout << this->status[i];
          }
          std::cout << "\n";
        }
    };
}

class SpawnedPhase {
  public:
    SpawnedPhase() : dummy_dev_idx_{0} {}

    void enqueue(InnerTask* task);
    void enqueue(std::vector<InnerTask*>& tasks);

    size_t get_count();

    /* This is the mapper. It moves stuff from spawned to mapped.*/
    //void run(MappedPhase* ready);
    void run(ReadyPhase* ready, DeviceManager* device_manager);

  private:
    std::string name = "Spawned Phase";
    Spawned::Status status;
    TaskQueue spawned_tasks;
    uint64_t dummy_dev_idx_;
};

namespace Mapped{

    enum State { failure, success };

    class Status{
        private:
            const static int size = 2;
        public:
            int status[size];
    };
}

class MappedPhase {
    public:
        std::string name = "Mapped Phase";
        Mapped::Status status;

        //TODO: Add any counters and internal state here.

        MappedPhase() = default;

        /* This is the reserver. It moves stuff from mapped to reserved.*/
        void run(ReservedPhase* ready);
};

namespace Reserved{

    enum State { failure, success};

    class Status{
        private:
            const static int size = 2;
        public:
            int status[size];
    };
}

class ReservedPhase {
    public:
        std::string name = "Reserved Phase";
        Reserved::Status status;

        //TODO: Add any counters and internal state here.

        ReservedPhase() = default;

        /* This is the readier. It it is a no op. As tasks move themselves to the Ready Phase. */
        void run(ReadyPhase* ready);
};

namespace Ready{

    enum State { entered, task_miss, resource_miss, worker_miss, success};

    class Status{
        private:
            const static int size = 5;
        public:
            int status[size];

            void reset(){
                for(int i = 0; i < size; i++){
                    this->status[i] = 0;
                }
            }

            void set(int index, int value){
                this->status[index] = value;
            }

            int get(int index){
                return this->status[index];
            }

            void update(State state){
                this->status[state]++;
            }

            void print(){
                std::cout << "Ready Status: (" << this->status[0] << " " << this->status[1] << " " << this->status[2] << " " << this->status[3] << " " << this->status[4] << ")" << std::endl;
            }
    };
}

#ifdef PARLA_ENABLE_LOGGING
    LOG_ADAPT_STRUCT(Ready::Status, status)
#endif

class ReadyPhase {

    public:
        TaskQueue ready_tasks;
        
        std::string name = "Ready Phase";

        InnerScheduler *scheduler;
        Ready::Status status;

        std::mutex mtx;

        ReadyPhase() = default;
        ReadyPhase(InnerScheduler *scheduler){
            this->scheduler = scheduler;
        }

        void set_scheduler(InnerScheduler *scheduler){
            this->scheduler = scheduler;
        }

        void enqueue(InnerTask* task);
        void enqueue(std::vector<InnerTask*>& tasks);

        int get_count();

        bool condition();

        void run(LauncherPhase* launcher);
};

#ifdef PARLA_ENABLE_LOGGING
    LOG_ADAPT_STRUCT(ReadyPhase, status)
#endif

namespace Launcher{

    enum State { failure, success};

    class Status{
        public:
            int status[2];

            void reset(){
                status[0] = 0;
                status[1] = 0;
            }

            void set(int index, int value){
                status[index] = value;
            }

            int get(int index){
                return status[index];
            }

            void update(State state){
                status[state]++;
            }

            void print(){
                std::cout << "Launcher Status: (" << this->status[0] << " " << this->status[1] << ")" << std::endl;
            }
    };
}

class LauncherPhase {
    public:
        std::string name = "Launcher Phase";
        Launcher::Status status;

        launchfunc_t launch_callback;
        
        /*Buffer to store not yet launched tasks. Currently unused. Placeholder in case it becomes useful.*/
        TaskList task_buffer;
        WorkerList worker_buffer;

        InnerScheduler *scheduler;
        void* py_scheduler;

        /*Number of running tasks. A task is running if it has been assigned to a worker and is not complete*/
        std::atomic<int> num_running_tasks;

        LauncherPhase() = default;
        LauncherPhase(InnerScheduler *scheduler){
            this->scheduler = scheduler;
        }

        LauncherPhase(InnerScheduler *scheduler, launchfunc_t launch_callback){
            this->scheduler = scheduler;
            this->launch_callback = launch_callback;
        }

        /* Pointer to the C++ scheduler */
        void set_scheduler(InnerScheduler *scheduler){
            this->scheduler = scheduler;
        }

        /* Set Python launch callback */
        void set_launch_callback(launchfunc_t launch_callback, void* py_scheduler){
            this->launch_callback = launch_callback;
            this->py_scheduler = py_scheduler;
        }

        /*Add a task to the launcher. Currently this acquires the GIL and dispatches the work to a Python Worker for each task */
        void enqueue(InnerTask* task, InnerWorker* worker);

        /* A placeholder function in case work needs to be done at this stage. For example, dispatching a whole buffer of tasks*/
        void run();

        /* Number of running tasks. A task is running if it has been assigned to a worker and is not complete */
        int get_num_running_tasks(){
            return this->num_running_tasks.load();
        }
};



#endif // PARLA_PHASES_HPP
