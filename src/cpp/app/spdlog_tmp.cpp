#include "include/pch.hpp"

std::shared_ptr<spdlog::logger> logger_;

void initLogger()
{
    logger_ = spdlog::stdout_color_mt("console");
    logger_->set_level(spdlog::level::debug);
}