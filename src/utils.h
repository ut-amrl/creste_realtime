#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>

inline std::vector<std::vector<bool>> createTrapezoidalFovMask(
    int H, int W, float fovTopAngle = 70, float fovBottomAngle = 70, float near = 0, float far = 200) {
    
    // Initialize the mask
    std::vector<std::vector<bool>> mask(H, std::vector<bool>(W, false));
    
    // Center coordinates
    float centerX = W / 2.0;
    float centerY = H / 2.0;

    // Loop through each pixel in the grid
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Calculate distance and angle
            float dx = x - centerX;
            float dy = centerY - y;
            float distance = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dx, dy) * 180.0 / M_PI;

            // Adjust angles to be in the range [-180, 180]
            if (angle < -180.0) {
                angle += 360.0;
            }

            // Determine angular spread
            float angularSpread;
            if (distance <= near) {
                angularSpread = fovTopAngle / 2.0;
            } else if (distance >= far) {
                angularSpread = fovBottomAngle / 2.0;
            } else {
                float t = (distance - near) / (far - near);
                angularSpread = (1 - t) * (fovTopAngle / 2.0) + t * (fovBottomAngle / 2.0);
            }

            // Create the mask
            if (distance >= near && distance <= far && std::abs(angle) <= angularSpread) {
                mask[y][x] = true;
            }
        }
    }

    return mask;
}