#ifndef PARLA_BACKEND_RL_UTILS_HPP
#define PARLA_BACKEND_RL_UTILS_HPP

static bool check_valid_tasks(std::string task_name) {
  if (task_name.find("global_0") == std::string::npos &&
      task_name.find("begin_rl_task") == std::string::npos &&
      task_name.find("end_rl_task") == std::string::npos &&
      task_name.find("Reset") == std::string::npos &&
      task_name.find("CopyBack") == std::string::npos) {
    return true;
  }
  return false;
}

#endif
