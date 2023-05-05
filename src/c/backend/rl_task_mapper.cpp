#include "include/rl_task_mapper.hpp"

RLTaskMappingPolicy::RLTaskMappingPolicy(
    DeviceManager *device_manager, PArrayTracker *parray_tracker,
    Mapper *mapper)
    : MappingPolicy(device_manager, parray_tracker) {
  size_t num_devices = device_manager->get_num_devices();
  this->rl_agent_ = new RLAgent(num_devices, num_devices);
  this->rl_env_ = new RLEnvironment(this->device_manager_, mapper);
}

bool RLTaskMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    const Mapper &mapper, Score_t *score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) {
  return true;
}

bool RLTaskMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
        &parray_list,
    std::vector<bool> *is_dev_assigned) {
  return true;
}

bool RLTaskMappingPolicy::calc_score_mdevplacement(
    InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
    const Mapper &mapper,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score,
    const std::vector<
        std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list) {
  return true;
}
