#include <iostream>
#include <string>

#include "app/spdlog_tmp.hpp"
#include "core/fdp_solver.hpp"
#include "core/instance.hpp"
#include "core/parsed_data.hpp"
#include "core/parser.hpp"

int main(int argc, char* argv[])
{
    if (argc != 7)
    {
        std::cerr << "Usage: solver <input_json_path> <output_path> <timelimit> <return_first>" << std::endl;
        return 1;
    }

    initLogger();

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    size_t max_seconds = std::stoul(argv[3]);
    int return_first = std::stoul(argv[4]);
    int fast_comp_feasible = std::stoul(argv[5]);
    int fast_comp_opt = std::stoul(argv[6]);

    try
    {
        auto data = FlightDistributionParser::parse_from_json(input_path);
        std::cout << "Parsed successfully from: " << input_path << std::endl;
        auto problem = data.getInstance();

        auto solver = Solver();
        solver.setup(problem);
        solver.solve(problem, max_seconds, return_first, fast_comp_feasible, fast_comp_opt);
        auto solution = solver.getSolution(problem);

        FlightDistributionParser::serialize_to_json(solver.getObjective(), solution, solver.getStatus(), output_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Parsing failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}