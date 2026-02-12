cmake_minimum_required(VERSION 3.14)
project(FlightSolver LANGUAGES CXX)

# Настройка стандарта
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Определение компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(MINGW)
        message(STATUS "MinGW detected - using safe optimizations")
        add_compile_options(-O2)  # Безопасный уровень оптимизации
    else()
        add_compile_options(-O3 -march=native)  # Агрессивные оптимизации только для Linux
    endif()
endif()

# Сначала создаем исполняемый файл
add_executable(solver
    app/solver.cpp
    app/spdlog_tmp.cpp
)

# Только ПОСЛЕ создания цели можно настраивать предкомпилированные заголовки
set(PCH_FILE "${PROJECT_SOURCE_DIR}/include/pch.hpp")
if(EXISTS "${PCH_FILE}")
    message(STATUS "Using precompiled header: ${PCH_FILE}")
    target_precompile_headers(solver PRIVATE "${PCH_FILE}")
else()
    message(WARNING "pch.hpp not found: ${PCH_FILE}")
endif()

# Настройки включения
target_include_directories(solver PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/include
)

# Для MinGW явно указываем необходимые библиотеки
if(MINGW)
    target_link_libraries(solver PRIVATE
        -static
        -static-libgcc
        -static-libstdc++
        -Wl,-Bstatic,--whole-archive
        -lwinpthread
        -Wl,--no-whole-archive
    )
endif()
