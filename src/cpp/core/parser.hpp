#pragma once
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "core/parsed_data.hpp"

class FlightDistributionParser
{
   public:
    static FlightDistributionData parse_from_json(std::string const& filepath)
    {
        FlightDistributionData data;

        // Чтение файла в строку
        std::ifstream input_file(filepath);
        if (!input_file.is_open())
        {
            throw std::runtime_error("Cannot open JSON file: " + filepath);
        }

        std::stringstream buffer;
        buffer << input_file.rdbuf();
        std::string json_content = buffer.str();

        // Парсинг строки
        rapidjson::Document j;
        j.Parse(json_content.c_str());

        if (j.HasParseError())
        {
            throw std::runtime_error("JSON parse error");
        }

        // alt_nodes
        for (auto const& outer : j["alt_nodes"].GetArray())
        {
            std::vector<int> inner_vec;
            for (auto const& inner : outer.GetArray())
            {
                inner_vec.push_back(inner.GetInt());
            }
            data.alt_nodes.push_back(std::move(inner_vec));
        }

        // start_point
        for (auto const& id : j["start_point"].GetArray())
        {
            data.start_point.push_back(id.GetInt());
        }

        // edges
        for (auto const& edge : j["edges"].GetArray())
        {
            auto const& arr = edge.GetArray();
            int from = arr[0].GetInt();
            int to = arr[1].GetInt();
            data.edges.emplace_back(from, to);
        }

        // num_aircrafts
        int num_aircrafts = 0;
        const auto& aircraft_nodes = j["aircraft_nodes"];
        for (auto itr = aircraft_nodes.MemberBegin(); itr != aircraft_nodes.MemberEnd(); ++itr)
        {
            num_aircrafts = std::max(num_aircrafts, std::stoi(itr->name.GetString()));
        }
        num_aircrafts += 1;

        // aircraft_nodes
        data.aircraft_nodes.resize(num_aircrafts);
        for (auto itr = aircraft_nodes.MemberBegin(); itr != aircraft_nodes.MemberEnd(); ++itr)
        {
            int key = std::stoi(itr->name.GetString());
            auto const& arr = itr->value.GetArray();
            std::vector<int> values;
            for (auto const& val : arr)
            {
                values.push_back(val.GetInt());
            }
            data.aircraft_nodes[key] = std::move(values);
        }

        // num_nodes
        int num_nodes = 0;
        const auto& node_cost = j["node_cost"];
        for (auto itr = node_cost.MemberBegin(); itr != node_cost.MemberEnd(); ++itr)
        {
            num_nodes = std::max(num_nodes, std::stoi(itr->name.GetString()));
        }
        num_nodes += 1;

        // node_cost
        data.node_cost.resize(num_nodes);
        for (auto itr = node_cost.MemberBegin(); itr != node_cost.MemberEnd(); ++itr)
        {
            int key = std::stoi(itr->name.GetString());
            double value = itr->value.GetDouble();
            data.node_cost[key] = value;
        }

        return data;
    }

    static void serialize_to_json(Score objective,
                                  DictInt<NodeId> const& solution,
                                  std::string const& status,
                                  std::string const& output_path)
    {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

        doc.AddMember("objective", objective, allocator);
        doc.AddMember("status", rapidjson::Value().SetString(status.c_str(), allocator), allocator);

        rapidjson::Value sol_array(rapidjson::kArrayType);
        for (int val : solution)
        {
            sol_array.PushBack(val, allocator);
        }

        doc.AddMember("solution", sol_array, allocator);

        // Сериализация в строку
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);

        // Запись в файл
        std::ofstream out_file(output_path);
        if (!out_file.is_open())
        {
            throw std::runtime_error("Cannot open output file: " + output_path);
        }
        out_file << buffer.GetString();
        out_file.close();
    }
};
