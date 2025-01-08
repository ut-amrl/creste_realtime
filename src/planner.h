#ifndef PLANNER_H_
#define PLANNER_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_set>

#include "shared/math/math_util.h"
#include "utils.h"
#include "yaml-cpp/yaml.h"

using math_util::Clamp;
using math_util::Sq;
namespace lsmap {

struct PathPoint {
  float x, y, theta;
  float v, w;
  int t_idx;
};

static std::vector<std::pair<int, int>> bresenham(int x0, int y0, int x1,
                                                  int y1) {
  std::vector<std::pair<int, int>> points;
  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);
  bool steep = (dy > dx);

  if (steep) {
    std::swap(x0, y0);
    std::swap(x1, y1);
    std::swap(dx, dy);
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }
  int error = dx / 2;
  int ystep = (y0 < y1) ? 1 : -1;
  int y = y0;

  for (int x = x0; x <= x1; x++) {
    if (steep) {
      points.emplace_back(y, x);
    } else {
      points.emplace_back(x, y);
    }
    error -= dy;
    if (error < 0) {
      y += ystep;
      error += dx;
    }
  }
  return points;
}

/**
 * Node class for Hybrid A*.
 */
class Node {
 public:
  float x, y;
  float theta;
  int t_idx;
  float v, w;
  Node* parent;

  float g;  // cost from start
  float h;  // heuristic cost to goal
  float f;  // total cost
  int depth;
  float weight;

  Node(float x_, float y_, float theta_, int t_idx_, float v_, float w_,
       Node* parent_ = nullptr)
      : x(x_),
        y(y_),
        theta(theta_),
        t_idx(t_idx_),
        v(v_),
        w(w_),
        parent(parent_) {
    g = std::numeric_limits<float>::infinity();
    h = std::numeric_limits<float>::infinity();
    f = std::numeric_limits<float>::infinity();
    depth = 0;
    weight = 1.0f;
  }

  // For priority_queue. We want the smallest f on top => invert compare.
  bool operator>(const Node& other) const { return f > other.f; }
};

/**
 * Min-heap comparator that compares Node pointers by their f value
 */
struct NodeCompare {
  bool operator()(Node* a, Node* b) {
    // We want the node with *smallest* f on top => a->f > b->f
    return a->f > b->f;
  }
};

class CarrotPlanner {
 public:
  CarrotPlanner(const YAML::Node& config) {
    LoadMapParams(config);
    LoadPlannerParams(config);
    TensorToVec2D(
        createTrapezoidalFovMask(map_params_.map_height, map_params_.map_width)
            .to(torch::kFloat32),
        fov_mask_);
  }

  void LoadMapParams(const YAML::Node& config) {
    const auto& node = config["map_params"];
    map_params_.resolution = node["resolution"].as<float>();
    map_params_.width = node["width"].as<float>();
    map_params_.height = node["height"].as<float>();
    map_params_.origin_x = node["origin_x"].as<float>();
    map_params_.origin_y = node["origin_y"].as<float>();
    map_params_.map_width = map_params_.width / map_params_.resolution;
    map_params_.map_height = map_params_.height / map_params_.resolution;
  }

  void LoadPlannerParams(const YAML::Node& config) {
    const auto& node = config["planner_params"];
    planner_params_.max_v = node["max_v"].as<float>();
    planner_params_.max_w = node["max_w"].as<float>();
    planner_params_.max_dv = node["max_dv"].as<float>();
    planner_params_.max_dw = node["max_dw"].as<float>();
    planner_params_.partitions = node["partitions"].as<int>();
    planner_params_.dt = node["dt"].as<float>();
    planner_params_.max_iters = node["max_iters"].as<int>();
    planner_params_.max_time = node["max_time"].as<float>();
    planner_params_.goal_tolerance = node["goal_tolerance"].as<float>();
    planner_params_.cost_weight = node["cost_weight"].as<float>();

    // Precompute valid v and w ranges
    v_range_ = linspace(-planner_params_.max_v, planner_params_.max_v,
                        planner_params_.partitions);
    w_range_ = linspace(-planner_params_.max_w, planner_params_.max_w,
                        planner_params_.partitions);

    // Print the valid ranges
    std::cout << "Valid v range: ";
    for (const auto& v : v_range_) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    std::cout << "Valid w range: ";
    for (const auto& w : w_range_) {
      std::cout << w << " ";
    }
    std::cout << std::endl;
  }

  /**
   * Plan a path from start_node to goal_node using A*.
   */
  std::vector<PathPoint> PlanPath(
      const std::vector<std::vector<float>>& traversability_map,
      const Pose2D& carrot) {
    // Initialize
    std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_heap;
    std::set<std::tuple<int, int, int>> closed_set;  // for visited states
    closed_set_.clear();                             // for debugging
    traversability_map_ = traversability_map;        // for path checking

    Node* start_node = new Node(0.5f, 0.0f, 0.0f, 0, 0.2f, 0.0f);
    Node* goal_node = new Node(carrot.x, carrot.y, 0.0f, 0, 0.0f, 0.0f);

    // Initialize start
    start_node->g = 0.0f;
    start_node->h = heuristic(start_node, goal_node);
    start_node->f = start_node->g + start_node->h;

    // Clamp goal to closest valid point
    goal_node->x = Clamp(goal_node->x, -map_params_.height / 2.0f,
                         map_params_.height / 2.0f);
    goal_node->y = Clamp(goal_node->y, -map_params_.width / 2.0f,
                         map_params_.width / 2.0f);

    // Check if start is out of bounds
    if (start_node->x < -map_params_.height / 2.0f ||
        start_node->x > map_params_.height / 2.0f ||
        start_node->y < -map_params_.width / 2.0f ||
        start_node->y > map_params_.width / 2.0f) {
      std::cout << "Start node out of bounds\n";
      return {};
    }

    open_heap.push(start_node);

    // Initialize closest node for timeout
    float best_distance_so_far = distance(start_node, goal_node);
    Node* best_node_so_far = start_node;  // Track best node so far

    int timeout_count = 0;
    while (!open_heap.empty()) {
      Node* curr_node = open_heap.top();
      open_heap.pop();

      // Update the closest node if the current node is nearer to the goal
      float curr_distance = distance(curr_node, goal_node);
      if (curr_distance < best_distance_so_far) {
        best_distance_so_far = curr_distance;
        best_node_so_far = curr_node;
      }

      if (goal_reached(curr_node, goal_node)) {
        return reconstruct_path(curr_node);
      }

      // If already visited
      if (is_in_close_set(closed_set, curr_node)) {
        continue;
      }
      // Mark visited
      add_to_close_set(closed_set, curr_node);

      timeout_count++;
      if (timeout_count > planner_params_.max_iters) {
        std::cout << "Astar planning timeout\n";
        return reconstruct_path(best_node_so_far);
      }

      // Expand neighbors
      std::vector<Node*> neighbors =
          get_neighbors(curr_node, planner_params_.max_time);
      for (Node* neighbor_node : neighbors) {
        if (goal_reached(neighbor_node, goal_node)) {
          // Return path from neighbor
          neighbor_node->parent = curr_node;
          return reconstruct_path(neighbor_node);
        }

        if (is_in_close_set(closed_set, neighbor_node)) {
          delete neighbor_node;
          continue;
        }

        neighbor_node->parent = curr_node;
        neighbor_node->depth = curr_node->depth + 1;
        neighbor_node->weight = curr_node->weight;

        // compute cost
        neighbor_node->g = compute_g_cost(curr_node, neighbor_node);
        neighbor_node->h = heuristic(neighbor_node, goal_node);
        neighbor_node->f = neighbor_node->g + neighbor_node->h;

        open_heap.push(neighbor_node);
      }
    }

    std::cout << "Exits without finding a path\n";
    return reconstruct_path(best_node_so_far);
  }

  void printExploredNodes() {
    // We'll use map_params_.map_height as rows, map_params_.map_width as cols
    // so image(row, col) => image(xCell, yCell).
    int rows = static_cast<int>(map_params_.map_height);
    int cols = static_cast<int>(map_params_.map_width);

    // Create a single-channel image with black background
    cv::Mat visitedImg(rows, cols, CV_8UC1, cv::Scalar(0));

    // Mark visited cells in white
    for (auto& key : closed_set_) {
      // key = (cell_x, cell_y, t_idx)
      int cell_x = std::get<0>(key);
      int cell_y = std::get<1>(key);

      // Safety check so we don't go out of bounds
      if (cell_x >= 0 && cell_x < rows && cell_y >= 0 && cell_y < cols) {
        visitedImg.at<uchar>(cell_x, cell_y) = 255;  // white pixel
      }
    }

    // Save the image
    cv::imwrite("explored_nodes.png", visitedImg);
  }

  void printPath(const std::vector<PathPoint>& path) {
    int rows = static_cast<int>(map_params_.map_height);
    int cols = static_cast<int>(map_params_.map_width);

    // Create a single-channel image with black background
    cv::Mat pathImg(rows, cols, CV_8UC1, cv::Scalar(0));

    // Mark path cells in white
    for (const auto& pt : path) {
      // Convert (x, y) to cell coordinates
      auto [cell_x, cell_y] = to_cell(pt.x, pt.y);

      // Safety check
      if (cell_x >= 0 && cell_x < rows && cell_y >= 0 && cell_y < cols) {
        pathImg.at<uchar>(cell_x, cell_y) = 255;  // white pixel for path
      }
    }

    // Save the image
    cv::imwrite("reconstructed_path.png", pathImg);
  }

  void publishPathOnMap(
      const std::vector<std::vector<float>>& traversability_map,
      const std::vector<PathPoint>& path, ros::Publisher& image_pub) {
    // Dimensions from the traversability map
    int rows = static_cast<int>(traversability_map.size());
    if (rows == 0) return;
    int cols = static_cast<int>(traversability_map[0].size());

    // Create a color image from the traversability map
    cv::Mat mapImage(rows, cols, CV_8UC3);

    // Fill the image with grayscale values based on traversability
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        float val = traversability_map[i][j];
        int intensity = static_cast<int>(Clamp(val * 255.0f, 0.0f, 255.0f));

        // Set values outside of fov mask to black
        if (fov_mask_[i][j] == 0.0f) {
          intensity = 0.0f;
        }
        mapImage.at<cv::Vec3b>(i, j) =
            cv::Vec3b(intensity, intensity, intensity);
      }
    }

    // Overlay the reconstructed path onto the map image in red
    for (size_t k = 1; k < path.size(); ++k) {
      const auto& prev = path[k - 1];
      const auto& curr = path[k];

      // Convert world coordinates to cell/grid indices
      auto [x1, y1] = to_cell(prev.x, prev.y);
      auto [x2, y2] = to_cell(curr.x, curr.y);

      // Draw a line segment between consecutive points in the path
      cv::line(mapImage, cv::Point(y1, x1), cv::Point(y2, x2),
               cv::Scalar(0, 0, 255), 2);  // Red color, thickness 2
    }

    // Convert the OpenCV image to a ROS Image message using cv_bridge
    cv_bridge::CvImage cv_image;
    cv_image.encoding = "bgr8";  // 8-bit color image
    cv_image.image = mapImage;
    sensor_msgs::ImagePtr msg = cv_image.toImageMsg();

    // Publish the image
    image_pub.publish(msg);
  }

 private:
  MapParams map_params_;
  PlannerParams planner_params_;

  // Precompute a list of values for v and w
  std::vector<float> v_range_;
  std::vector<float> w_range_;
  std::set<std::tuple<int, int, int>> closed_set_;
  std::vector<std::vector<float>> traversability_map_;
  std::vector<std::vector<float>> fov_mask_;

  /**
   * Generate a linearly spaced vector from start to end with 'n' points.
   */
  std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> v;
    if (n <= 1) {
      v.push_back(start);
      return v;
    }
    float step = (end - start) / (float)(n - 1);
    for (int i = 0; i < n; i++) {
      v.push_back(start + i * step);
    }
    return v;
  }

  /**
   * Compute the Euclidean distance between node1 and node2.
   */
  float distance(Node* node1, Node* node2) {
    float dx = node1->x - node2->x;
    float dy = node1->y - node2->y;
    return std::sqrt(dx * dx + dy * dy);
  }

  /**
   * The main heuristic function. (Currently uses simple distance.)
   */
  float heuristic(Node* curr_node, Node* goal_node) {
    float d = distance(curr_node, goal_node);
    // If you want to incorporate `heuristic_multiplier_` or randomness, do it
    // here.
    return d;
  }

  /**
   * Transform from world coordinates (x,y) -> grid (i,j).
   * Equivalent to the Python:
   *   int(-x / map_res[0] + grid_size[0]//2), int(-y / map_res[1] +
   * grid_size[1]//2)
   */
  std::pair<int, int> to_cell(float x, float y) {
    int cx =
        (int)(-x / map_params_.resolution + (map_params_.map_height / 2.0f));
    int cy =
        (int)(-y / map_params_.resolution + (map_params_.map_width / 2.0f));
    return {cx, cy};
  }

  /**
   * Transform from grid (i,j) -> world coordinates (x,y).
   */
  std::pair<float, float> to_coord(int i, int j) {
    float x = -(i - (map_params_.map_height / 2.0f)) * map_params_.resolution;
    float y = -(j - (map_params_.map_width / 2.0f)) * map_params_.resolution;
    return {x, y};
  }

  /**
   * Check if current node is within goal_radius of goal_node
   */
  bool goal_reached(Node* curr_node, Node* goal_node) {
    return (distance(curr_node, goal_node) < planner_params_.goal_tolerance);
  }

  /**
   * Add node to closed set
   */
  void add_to_close_set(std::set<std::tuple<int, int, int>>& closed_set,
                        Node* node) {
    auto [cx, cy] = to_cell(node->x, node->y);
    int t_idx = node->t_idx;
    closed_set.insert(std::make_tuple(cx, cy, t_idx));
    // Also store for debugging
    closed_set_.insert(std::make_tuple(cx, cy, t_idx));
  }

  /**
   * Check if node is in closed set
   */
  bool is_in_close_set(const std::set<std::tuple<int, int, int>>& closed_set,
                       Node* node) {
    auto [cx, cy] = to_cell(node->x, node->y);
    int t_idx = node->t_idx;
    auto key = std::make_tuple(cx, cy, t_idx);
    return (closed_set.find(key) != closed_set.end());
  }

  /**
   * Reconstruct path by backtracking from current node up to the start
   */
  std::vector<PathPoint> reconstruct_path(Node* curr_node) {
    std::vector<PathPoint> path;
    float curr_v = 0.0f;
    float curr_w = 0.0f;
    while (curr_node != nullptr) {
      PathPoint p;
      p.x = curr_node->x;
      p.y = curr_node->y;
      p.theta = curr_node->theta;
      p.t_idx = curr_node->t_idx;
      p.v = curr_v;
      p.w = curr_w;
      // Then update for next iteration
      curr_v = curr_node->v;
      curr_w = curr_node->w;
      path.push_back(p);

      curr_node = curr_node->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  /**
   * Compute the cost g when moving from curr_node to neighbor_node
   */
  float compute_g_cost(Node* curr_node, Node* neighbor_node) {
    float curr_cost = curr_node->g;
    float dist_cost = distance(curr_node, neighbor_node);

    // Summation of cost map along line from curr_node to neighbor_node
    float learned_cost = get_learned_cost(curr_node, neighbor_node);

    float g = curr_cost + dist_cost + learned_cost;
    return g;
  }

  /**
   * Summation of cost_map cells along the line from (x1,y1)->(x2,y2).
   * Equivalent to Python's get_learned_cost.
   */
  float get_learned_cost(Node* curr_node, Node* neighbor_node) {
    float x1 = curr_node->x;
    float y1 = curr_node->y;
    float x2 = neighbor_node->x;
    float y2 = neighbor_node->y;

    // Just an assertion that we are in bounds
    // (Adjust or remove if you want less strict checks)
    assert(x2 >= -map_params_.height / 2.0f && x2 < map_params_.height / 2.0f &&
           "Neighbor node x out of bounds");
    assert(y2 >= -map_params_.width / 2.0f && y2 < map_params_.width / 2.0f &&
           "Neighbor node y out of bounds");

    // Collect cells between the two nodes
    auto [cx1, cy1] = to_cell(x1, y1);
    auto [cx2, cy2] = to_cell(x2, y2);

    std::vector<std::pair<int, int>> cells = bresenham(cx1, cy1, cx2, cy2);

    // Reverse the cells
    std::reverse(cells.begin(), cells.end());

    float total_cost = 0.0f;
    for (auto& c : cells) {
      int i = c.first;
      int j = c.second;
      // clamp i,j
      i = std::max(0, std::min(i, map_params_.map_height - 1));
      j = std::max(0, std::min(j, map_params_.map_width - 1));

      // cost map access => cost_map_[i][j],
      // then sum using exponent, etc.
      float cval = traversability_map_[i][j];
      total_cost += std::exp(planner_params_.cost_weight * cval);
    }
    return total_cost;
  }

  /**
   * Forward motion rollout from (x,y,theta) using constant curvature
   * for time = planning_dt_.
   */
  void forward_motion_rollout(Node* node, float v, float w, float& x_new,
                              float& y_new, float& theta_new) {
    float dt = planner_params_.dt;
    if (std::fabs(w) > 1e-6) {
      theta_new = node->theta + w * dt;
      x_new = node->x + v / w * (std::sin(theta_new) - std::sin(node->theta));
      y_new = node->y - v / w * (std::cos(theta_new) - std::cos(node->theta));
    } else {
      theta_new = node->theta;
      x_new = node->x + v * std::cos(node->theta) * dt;
      y_new = node->y + v * std::sin(node->theta) * dt;
    }
  }

  /**
   * Return the list of feasible velocities & angular velocities
   * for the next step.
   */
  std::pair<std::vector<float>, std::vector<float>> get_valid_vw(
      float current_v, float current_w) {
    float min_v = std::max(0.1f, current_v - planner_params_.max_dv);
    float max_v =
        std::min(planner_params_.max_v, current_v + planner_params_.max_dv);

    std::vector<float> valid_v;
    for (auto v : v_range_) {
      if (v >= min_v && v <= max_v) {
        valid_v.push_back(v);
      }
    }

    float min_w =
        std::max(-planner_params_.max_w, current_w - planner_params_.max_dw);
    float max_w =
        std::min(planner_params_.max_w, current_w + planner_params_.max_dw);

    std::vector<float> valid_w;
    for (auto w : w_range_) {
      if (w >= min_w && w <= max_w) {
        valid_w.push_back(w);
      }
    }
    return {valid_v, valid_w};
  }

  /**
   * Generate neighbors by expanding from 'node' with all feasible (v,w).
   */
  std::vector<Node*> get_neighbors(Node* node, int max_time_idx) {
    std::vector<Node*> neighbors;
    auto vw = get_valid_vw(node->v, node->w);
    auto valid_v = vw.first;
    auto valid_w = vw.second;

    for (float v : valid_v) {
      for (float w : valid_w) {
        float x_new, y_new, theta_new;
        forward_motion_rollout(node, v, w, x_new, y_new, theta_new);

        // Check bounds
        if (x_new <= -map_params_.height / 2.0f ||
            x_new >= map_params_.height / 2.0f ||
            y_new <= -map_params_.width / 2.0f ||
            y_new >= map_params_.width / 2.0f) {
          continue;
        }
        int t_idx = std::min(node->t_idx + 1, max_time_idx);

        // Collisions can be checked here if desired
        // e.g. check cost_map_ or do a bresenham line from curr_x,curr_y to
        // x_new,y_new
        int curr_x, curr_y;
        std::tie(curr_x, curr_y) = to_cell(node->x, node->y);
        bool collided = fabs(fov_mask_[curr_x][curr_y]) < 1e-6;

        if (!collided) {
          Node* nbr = new Node(x_new, y_new, theta_new, t_idx, v, w);
          neighbors.push_back(nbr);
        }
      }
    }
    return neighbors;
  }
};

}  // namespace lsmap

#endif  // PLANNER_H_
