



```python

class PArray:

    def __init__(self):
        self.cy_parray = CyPArray()

    ## The rest of normal PArray

cdef class CyParray:
    cdeff CppPArray cpp_parray

    def __cinit__(self):
        self.cpp_parray = new CppPArray()

    def __init__(self):
        pass

    #Wrappers for the methods of CppPArray

    //set_id
    //set_size
    //set_state
    //get_state


```

```cpp

class CppPArray{

    public:
        CppPArray();

        //Minimal Methods needed by Mapper & Eviction Manager

        //Get the unique ID of the PArray
        size_t get_id();
        //Get current size of the PArray on a specific device
        size_t get_size(device_id);
        //Query the state of the PArray on a specific device
        PArrayState get_state(device_id);
        //Get the set of all current valid devices
        container<device_id> get_valid_devices();

        //Some others like `is_last` may be useful? 

        //Methods needed to track data depenendencies and task graph partition costs
        //TODO(wlr): Not related to PArray coherency features. Used by the runtime. 
        //           Could be a separate class?
        void add_task(InnerTask*);    //require O(1) access
        void remove_task(InnerTask*); //require O(1) access
        std::vector<InnerTask*> get_tasks();
        

        //Methods likely needed by Python PArray layer
        void set_id(size_t);
        size_t set_size(device_id, size_t);
        PArrayState set_state(device_id, PArrayState);



    protected:
        size_t id;
        //TODO(yy): coherence table
        //TODO(yy): other metadata about data (size of each chunk, etc.)

        //TODO(wlr): List of tasks that are currently using this PArray
};

```

```python

```