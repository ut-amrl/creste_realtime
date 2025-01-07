#ifndef PLANNER_H_
#define PLANNER_H_

#include <queue>
#include <cmath>
#include <unordered_set>
#include <iostream>
#include "yaml-cpp/yaml.h"
#include "utils.h"

namespace lsmap {

struct Node {
    int x, y;       // Grid coordinates
    float theta;    // Orientation
    float v, w;     // Linear and angular velocity
    float t;        // Time
    float g, h, f;  // Cost values
    Node *parent;

    Node(int x_, int y_, float theta_, float v_ = 0.0f, float w_ = 0.0f, float t_ = 0.0f, Node *parent_ = nullptr)
        : x(x_), y(y_), theta(theta_), v(v_), w(w_), t(t_), g(INFINITY), h(INFINITY), f(INFINITY), parent(parent_) {}

    bool operator<(const Node &other) const {
        return f > other.f;  // Priority queue (min-heap) based on `f` value
    }
};

struct HashNode {
    size_t operator()(const std::tuple<int, int, float> &node) const {
        return std::hash<int>()(std::get<0>(node)) ^ std::hash<int>()(std::get<1>(node)) ^ std::hash<float>()(std::get<2>(node));
    }
};

class CarrotPlanner {
public:
    CarrotPlanner(const YAML::Node &config) {
        LoadMapParams(config);
        LoadPlannerParams(config);
    }

    void LoadMapParams(const YAML::Node &config) {
        const auto &node = config["map_params"];
        map_params_.resolution = node["resolution"].as<float>();
        map_params_.height = node["height"].as<float>() / map_params_.resolution;
        map_params_.width = node["width"].as<float>() / map_params_.resolution;
        map_params_.origin_x = node["origin_x"].as<float>() / map_params_.resolution;
        map_params_.origin_y = node["origin_y"].as<float>() / map_params_.resolution;
    }

    void LoadPlannerParams(const YAML::Node &config) {
        const auto &node = config["planner_params"];
        planner_params_.max_v = node["max_v"].as<float>();
        planner_params_.max_w = node["max_w"].as<float>();
        planner_params_.max_dv = node["max_dv"].as<float>();
        planner_params_.max_dw = node["max_dw"].as<float>();
        planner_params_.partitions = node["partitions"].as<int>();
        planner_params_.dt = node["dt"].as<float>();
    }

    Path PlanPath(const std::vector<std::vector<float>> &traversability_map, const Pose2D &carrot) {
        int H = traversability_map.size();  // Height
        int W = traversability_map[0].size();  // Width
        float resolution = map_params_.resolution;

        int goal_x = static_cast<int>(std::round(carrot.x / resolution + W / 2));
        int goal_y = static_cast<int>(std::round(carrot.y / resolution + H / 2));

        Node *start = new Node(map_params_.origin_x, map_params_.origin_y, 0.0f, 0.2f);
        start->g = 0.0f;
        start->h = heuristic(start, goal_x, goal_y);
        start->f = start->g + start->h;

        std::priority_queue<Node *> open_list;
        open_list.push(start);

        std::unordered_set<std::tuple<int, int, float>, HashNode> closed_set;

        while (!open_list.empty()) {
            Node *current = open_list.top();
            open_list.pop();

            if (isGoal(current, goal_x, goal_y)) {
                return reconstructPath(current);
            }

            closed_set.insert(std::make_tuple(current->x, current->y, current->theta));

            for (const auto &neighbor : getNeighbors(current, traversability_map)) {
                if (closed_set.count(std::make_tuple(neighbor->x, neighbor->y, neighbor->theta))) {
                    continue;  // Skip already visited nodes
                }

                float tentative_g = current->g + computeCost(current, neighbor, traversability_map);
                if (tentative_g < neighbor->g) {
                    neighbor->parent = current;
                    neighbor->g = tentative_g;
                    neighbor->h = heuristic(neighbor, goal_x, goal_y);
                    neighbor->f = neighbor->g + neighbor->h;
                    open_list.push(neighbor);
                }
            }
        }

        return Path();
    }

private:
    MapParams map_params_;
    PlannerParams planner_params_;

    bool isGoal(Node *node, int goal_x, int goal_y) {
        const float goal_radius = 5.0; // 5 cells (0.5 meters)
        return std::hypot(node->x - goal_x, node->y - goal_y) <= goal_radius;
    }

    float heuristic(Node *node, int goal_x, int goal_y) {
        return std::hypot(node->x - goal_x, node->y - goal_y);
    }

    float computeCost(Node *from, Node *to, const std::vector<std::vector<float>> &map) {
        float cost = map[to->y][to->x];
        float dist = std::hypot(to->x - from->x, to->y - from->y);
        return dist + std::exp(cost);
    }

    std::pair<std::vector<float>, std::vector<float>> getValidVW(float current_v, float current_w) {
        // Calculate the valid range of linear and angular velocities
        float min_v = std::max(0.1f, current_v - planner_params_.max_dv);  // Only allow forward motion
        float max_v = std::min(planner_params_.max_v, current_v + planner_params_.max_dv);

        float min_w = std::max(-planner_params_.max_w, current_w - planner_params_.max_dw);
        float max_w = std::min(planner_params_.max_w, current_w + planner_params_.max_dw);

        std::vector<float> valid_v, valid_w;
        for (int i = 0; i < planner_params_.partitions; ++i) {
            float v = min_v + i * (max_v - min_v) / (planner_params_.partitions - 1);
            float w = min_w + i * (max_w - min_w) / (planner_params_.partitions - 1);
            valid_v.push_back(v);
            valid_w.push_back(w);
        }

        return {valid_v, valid_w};
    }

    std::vector<Node *> getNeighbors(Node *node, const std::vector<std::vector<float>> &map) {
        std::vector<Node *> neighbors;
        auto [valid_v, valid_w] = getValidVW(node->v, node->w);
        float dt = planner_params_.dt;

        for (float v : valid_v) {
            for (float w : valid_w) {
                float new_theta = node->theta + w * dt;
                float new_x = (w != 0.0f) ? (node->x + v / w * (std::sin(new_theta) - std::sin(node->theta)))
                                          : (node->x + v * std::cos(node->theta) * dt);
                float new_y = (w != 0.0f) ? (node->y - v / w * (std::cos(new_theta) - std::cos(node->theta)))
                                          : (node->y + v * std::sin(node->theta) * dt);

                int grid_x = static_cast<int>(std::round(new_x));
                int grid_y = static_cast<int>(std::round(new_y));

                if (grid_x < 0 || grid_x >= int(map[0].size()) || grid_y < 0 || grid_y >= int(map.size())) {
                    continue;
                }

                float cost = map[grid_y][grid_x];
                if (cost >= 0.9f) {
                    continue;
                }

                float new_t = node->t + dt;
                neighbors.push_back(new Node(grid_x, grid_y, new_theta, v, w, new_t, node));
            }
        }

        return neighbors;
    }

    Path reconstructPath(Node *node) {
        Path path;
        while (node) {
            Pose2D pose{node->x * map_params_.resolution - map_params_.width / 2,
                         node->y * map_params_.resolution - map_params_.height / 2,
                         node->theta};
            path.poses.push_back(pose);
            node = node->parent;
        }
        std::reverse(path.poses.begin(), path.poses.end());
        return path;
    }
};

}  // namespace lsmap

#endif  // PLANNER_H_
