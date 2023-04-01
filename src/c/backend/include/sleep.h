#include<unistd.h>
#include<time.h>
#include<chrono>

using namespace std;
using namespace chrono;

/*
void busy_sleep(const unsigned milli){
    clock_t time_end;
    time_end = clock() + milli * CLOCKS_PER_SEC/1000;
    while(clock() < time_end)
    {
    }
}
*/

void busy_sleep(const unsigned micro){
    //int count = 0;
    auto block = chrono::microseconds(micro);
    auto time_start = chrono::high_resolution_clock::now();

    auto now = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(now - time_start);

    do{

        now = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast<chrono::microseconds>(now - time_start);
    }
    while(elapsed.count() < micro);
    //return count;
}


void sleeper(const unsigned int t){
    sleep(t);
}
