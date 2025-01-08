#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <vector>

// A simple struct to hold path data if needed
struct PathPoint {
  float x, y, theta;
  float v, w;
  int t_idx;
};

/**
 * Bresenham line drawing from (x0,y0) to (x1,y1).
 * Returns a list of (x,y) cells that the line passes through.
 */
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

/**
 * AStarPlanner class
 */
class AStarPlanner {
 public:
  // Configuration / parameters
  std::vector<std::vector<float>> cost_map_;
  float cost_weight_;
  std::vector<float> map_size_;  // e.g. [25.6, 25.6]
  std::vector<float> map_res_;   // e.g. [0.1, 0.1]
  std::vector<int> grid_size_;   // e.g. [256, 256]

  float max_v_;
  float max_w_;
  float max_dv_;
  float max_dw_;
  int partitions_;

  std::vector<float> v_range_;
  std::vector<float> w_range_;
  float planning_dt_;

  float heuristic_multiplier_;
  float linear_acc_weight_mean_;
  float angular_acc_weight_mean_;
  bool random_motion_cost_;
  float heuristic_annealing_;
  float heuristic_direction_weight_;

  // For debugging / visualization
  std::set<std::tuple<int, int, int>> closed_set_;

  AStarPlanner(const std::vector<std::vector<float>>& cost_map,
               float cost_weight, const std::vector<float>& map_size,
               const std::vector<float>& map_res, float max_v, float max_w,
               float max_dv, float max_dw, int partitions, float planning_dt,
               float heuristic_multiplier, float linear_acc_weight,
               float angular_acc_weight, bool random_motion_cost = false,
               float heuristic_annealing = 1.0f,
               float heuristic_direction_weight = 1.0f) {
    cost_map_ = cost_map;
    cost_weight_ = cost_weight;
    map_size_ = map_size;
    map_res_ = map_res;
    max_v_ = max_v;
    max_w_ = max_w;
    max_dv_ = max_dv;
    max_dw_ = max_dw;
    partitions_ = partitions;
    planning_dt_ = planning_dt;
    heuristic_multiplier_ = heuristic_multiplier;
    linear_acc_weight_mean_ = linear_acc_weight;
    angular_acc_weight_mean_ = angular_acc_weight;
    random_motion_cost_ = random_motion_cost;
    heuristic_annealing_ = heuristic_annealing;
    heuristic_direction_weight_ = heuristic_direction_weight;

    // Compute grid_size
    // e.g. if map_size_ = [25.6, 25.6], map_res_ = [0.1, 0.1], then grid_size_
    // = [256, 256]
    int gx = (int)(map_size_[0] / map_res_[0]);
    int gy = (int)(map_size_[1] / map_res_[1]);
    grid_size_ = {gx, gy};

    // Create velocity & angular velocity ranges
    v_range_ = linspace(-max_v_, max_v_, partitions_);
    w_range_ = linspace(-max_w_, max_w_, partitions_);
  }

  /**
   * Plan a path from start_node to goal_node using A*.
   */
  std::vector<PathPoint> plan(Node* start_node, Node* goal_node,
                              int max_time_idx, float goal_radius,
                              int max_expand_node_num) {
    // Initialize
    std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_heap;
    std::set<std::tuple<int, int, int>> closed_set;  // for visited states
    closed_set_.clear();                             // for debugging

    // Initialize start
    start_node->g = 0.0f;
    start_node->h = heuristic(start_node, goal_node);
    start_node->f = start_node->g + start_node->h;

    // Check if start is out of bounds
    if (start_node->x < -map_size_[0] / 2.0f ||
        start_node->x > map_size_[0] / 2.0f ||
        start_node->y < -map_size_[1] / 2.0f ||
        start_node->y > map_size_[1] / 2.0f) {
      std::cout << "Start node out of bounds\n";
      return {};
    }

    open_heap.push(start_node);

    int timeout_count = 0;
    while (!open_heap.empty()) {
      Node* curr_node = open_heap.top();
      open_heap.pop();

      if (goal_reached(curr_node, goal_node, goal_radius)) {
        return reconstruct_path(curr_node);
      }

      // If already visited
      if (is_in_close_set(closed_set, curr_node)) {
        continue;
      }
      // Mark visited
      add_to_close_set(closed_set, curr_node);

      timeout_count++;
      if (timeout_count > max_expand_node_num) {
        std::cout << "Astar planning timeout\n";
        return {};
      }

      // Expand neighbors
      std::vector<Node*> neighbors = get_neighbors(curr_node, max_time_idx);
      for (Node* neighbor_node : neighbors) {
        if (goal_reached(neighbor_node, goal_node, goal_radius)) {
          neighbor_node->parent = curr_node;
          // Return path from neighbor
          return reconstruct_path(neighbor_node);
        }

        if (is_in_close_set(closed_set, neighbor_node)) {
          delete neighbor_node;
          continue;
        }

        neighbor_node->parent = curr_node;
        neighbor_node->depth = curr_node->depth + 1;
        neighbor_node->weight = curr_node->weight * heuristic_annealing_;

        // compute cost
        neighbor_node->g = compute_g_cost(curr_node, neighbor_node);
        neighbor_node->h = heuristic(neighbor_node, goal_node);
        neighbor_node->f = neighbor_node->g + neighbor_node->h;

        open_heap.push(neighbor_node);
      }
    }

    std::cout << "Exits without finding a path\n";
    return {};
  }

 private:
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
    int cx = (int)(-x / map_res_[0] + (grid_size_[0] / 2.0f));
    int cy = (int)(-y / map_res_[1] + (grid_size_[1] / 2.0f));
    return {cx, cy};
  }

  /**
   * Transform from grid (i,j) -> world coordinates (x,y).
   */
  std::pair<float, float> to_coord(int i, int j) {
    float x = -(i - (grid_size_[0] / 2.0f)) * map_res_[0];
    float y = -(j - (grid_size_[1] / 2.0f)) * map_res_[1];
    return {x, y};
  }

  /**
   * Check if current node is within goal_radius of goal_node
   */
  bool goal_reached(Node* curr_node, Node* goal_node, float goal_radius) {
    return (distance(curr_node, goal_node) < goal_radius);
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
    assert(x2 >= -map_size_[0] / 2.0f && x2 < map_size_[0] / 2.0f &&
           "Neighbor node x out of bounds");
    assert(y2 >= -map_size_[1] / 2.0f && y2 < map_size_[1] / 2.0f &&
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
      i = std::max(0, std::min(i, grid_size_[0] - 1));
      j = std::max(0, std::min(j, grid_size_[1] - 1));

      // cost map access => cost_map_[i][j],
      // then sum using exponent, etc.
      float cval = cost_map_[i][j];
      total_cost += std::exp(cost_weight_ * cval);
    }
    return total_cost;
  }

  /**
   * Forward motion rollout from (x,y,theta) using constant curvature
   * for time = planning_dt_.
   */
  void forward_motion_rollout(Node* node, float v, float w, float& x_new,
                              float& y_new, float& theta_new) {
    if (std::fabs(w) > 1e-6) {
      theta_new = node->theta + w * planning_dt_;
      x_new = node->x + v / w * (std::sin(theta_new) - std::sin(node->theta));
      y_new = node->y - v / w * (std::cos(theta_new) - std::cos(node->theta));
    } else {
      theta_new = node->theta;
      x_new = node->x + v * std::cos(node->theta) * planning_dt_;
      y_new = node->y + v * std::sin(node->theta) * planning_dt_;
    }
  }

  /**
   * Return the list of feasible velocities & angular velocities
   * for the next step.
   */
  std::pair<std::vector<float>, std::vector<float>> get_valid_vw(
      float current_v, float current_w) {
    // Example from python: only positive velocity
    float min_v = std::max(0.1f, current_v - max_dv_);
    float max_v = std::min(max_v_, current_v + max_dv_);

    std::vector<float> valid_v;
    for (auto v : v_range_) {
      if (v >= min_v && v <= max_v) {
        valid_v.push_back(v);
      }
    }

    float min_w = std::max(-max_w_, current_w - max_dw_);
    float max_w = std::min(max_w_, current_w + max_dw_);

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

    // float curr_x = node->x;
    // float curr_y = node->y;

    for (float v : valid_v) {
      for (float w : valid_w) {
        float x_new, y_new, theta_new;
        forward_motion_rollout(node, v, w, x_new, y_new, theta_new);

        // Check bounds
        if (x_new <= -map_size_[0] / 2.0f || x_new >= map_size_[0] / 2.0f ||
            y_new <= -map_size_[1] / 2.0f || y_new >= map_size_[1] / 2.0f) {
          continue;
        }
        int t_idx = std::min(node->t_idx + 1, max_time_idx);

        // Collisions can be checked here if desired
        // e.g. check cost_map_ or do a bresenham line from curr_x,curr_y to
        // x_new,y_new
        bool collided = false;
        if (!collided) {
          Node* nbr = new Node(x_new, y_new, theta_new, t_idx, v, w);
          neighbors.push_back(nbr);
        }
      }
    }
    return neighbors;
  }
};

// -----------------------------------------------------------------------------
// Example main() usage
// -----------------------------------------------------------------------------
int main() {
  // Example usage:
  // Suppose your cost_map is 256x256 (for a 25.6 x 25.6 area at 0.1
  // resolution). We'll just fill it with zeros for demonstration.
  int rows = 256;
  int cols = 256;
  std::vector<std::vector<float>> cost_map(rows,
                                           std::vector<float>(cols, 0.0f));

  // Example obstacles: set a rectangle with cost
  for (int i = 100; i < 120; i++) {
    for (int j = 100; j < 120; j++) {
      cost_map[i][j] = 1.0f;  // an obstacle region
    }
  }

  // Construct AStarPlanner
  std::vector<float> map_size = {25.6f, 25.6f};
  std::vector<float> map_res = {0.1f, 0.1f};
  float cost_weight = 2.0f;
  float max_v = 1.0f;
  float max_w = (float)M_PI / 2.0f;
  float max_dv = 0.2f;
  float max_dw = (float)M_PI / 4.0f;
  int partitions = 11;
  float planning_dt = 1.0f;
  float heuristic_multiplier = 0.0f;
  float linear_acc_weight = 0.0f;
  float angular_acc_weight = 0.0f;
  bool random_motion_cost = false;
  float heuristic_annealing = 1.0f;

  AStarPlanner planner(
      cost_map, cost_weight, map_size, map_res, max_v, max_w, max_dv, max_dw,
      partitions, planning_dt, heuristic_multiplier, linear_acc_weight,
      angular_acc_weight, random_motion_cost, heuristic_annealing);

  // Create start and goal nodes
  Node* start_node = new Node(0.0f, 0.0f, 0.0f, 0, 0.2f, 0.0f);
  Node* goal_node = new Node(10.14f, 3.51f, 0.21f, 0, 0.0f, 0.0f);

  // Plan
  int max_time_idx = 0;
  float goal_radius = 0.5f;
  int max_expand_node_num = 100000;
  std::vector<PathPoint> path = planner.plan(
      start_node, goal_node, max_time_idx, goal_radius, max_expand_node_num);
  if (!path.empty()) {
    std::cout << "Path found! Size=" << path.size() << "\n";
    for (auto& p : path) {
      std::cout << "(" << p.x << ", " << p.y << ", " << p.theta << ") v=" << p.v
                << " w=" << p.w << " t=" << p.t_idx << "\n";
    }
  } else {
    std::cout << "No path found.\n";
  }

  // Cleanup
  delete start_node;
  delete goal_node;

  return 0;
}
