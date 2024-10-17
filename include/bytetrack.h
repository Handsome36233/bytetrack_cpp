#ifndef BYTETRACK_H
#define BYTETRACK_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include "hungarian.h"
#include "kalman_filter.h"
#include "detector.h"

enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

class STrack {
public:
    static KalmanFilter shared_kalman; 
    std::vector<float> _tlwh;
    KalmanFilter* kalman_filter;
    std::vector<float> mean;
    std::vector<std::vector<float>> covariance;
    TrackState state;
    bool is_activated;
    int track_id;
    int frame_id;
    int start_frame;
    int tracklet_len;
    float score;

    STrack();
    STrack(const std::vector<float>& tlwh, float score);

    std::vector<float> tlwh() const;
    std::vector<float> tlwh_to_xyah(const std::vector<float>& tlwh) const;

    void mark_lost();
    void mark_removed();
    void predict();
    static void multi_predict(std::vector<STrack>& stracks);

    void activate(KalmanFilter* kalman_filter, int frame_id, int track_id);
    void re_activate(const STrack& new_track, int frame_id);
    void update(const STrack& new_track, int frame_id);
};

static KalmanFilter shared_kalman;

class BYTETracker {
public:
    BYTETracker(int frame_rate = 30, float low_conf_thresh = 0.1f, float high_conf_thresh = 0.5f);

    std::vector<STrack> update(const std::vector<Detection>& det_results);

private:
    float det_low_conf_thresh;
    float det_high_conf_thresh;
    int frame_id;
    int track_id;
    int max_time_lost;

    std::vector<STrack> all_tracked_stracks;
    std::vector<STrack> all_lost_stracks;
    std::vector<STrack> all_removed_stracks;

    std::vector<STrack> get_tracked_stracks(const std::vector<STrack>& strack_pool, const std::vector<int>& u_track);
    std::vector<STrack> filter_detections(const std::vector<STrack>& detections, const std::vector<int>& u_detection);
};

#endif
