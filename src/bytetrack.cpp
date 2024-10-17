#include "bytetrack.h"

using namespace std;

std::tuple<std::vector<std::vector<float>>, size_t, size_t> iou_distance(const std::vector<std::vector<float>>& atlbrs,
                                                                        const std::vector<std::vector<float>>& btlbrs) {
    size_t N = atlbrs.size();
    size_t M = btlbrs.size();
    
    std::vector<std::vector<float>> overlaps(N, std::vector<float>(M, 0.0));
    
    if (N == 0 || M == 0) {
        return std::make_tuple(overlaps, N, M);
    }

    std::vector<std::vector<float>> a_lbrs(N, std::vector<float>(4));
    std::vector<std::vector<float>> b_lbrs(M, std::vector<float>(4));

    for (size_t i = 0; i < N; ++i) {
        a_lbrs[i][0] = atlbrs[i][0] - atlbrs[i][2] / 2;
        a_lbrs[i][1] = atlbrs[i][1] - atlbrs[i][3] / 2;
        a_lbrs[i][2] = atlbrs[i][0] + atlbrs[i][2] / 2;
        a_lbrs[i][3] = atlbrs[i][1] + atlbrs[i][3] / 2;
    }

    for (size_t i = 0; i < M; ++i) {
        b_lbrs[i][0] = btlbrs[i][0] - btlbrs[i][2] / 2;
        b_lbrs[i][1] = btlbrs[i][1] - btlbrs[i][3] / 2;
        b_lbrs[i][2] = btlbrs[i][0] + btlbrs[i][2] / 2;
        b_lbrs[i][3] = btlbrs[i][1] + btlbrs[i][3] / 2;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float inter_x1 = std::max(a_lbrs[i][0], b_lbrs[j][0]);
            float inter_y1 = std::max(a_lbrs[i][1], b_lbrs[j][1]);
            float inter_x2 = std::min(a_lbrs[i][2], b_lbrs[j][2]);
            float inter_y2 = std::min(a_lbrs[i][3], b_lbrs[j][3]);

            float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
            float a_area = (a_lbrs[i][2] - a_lbrs[i][0]) * (a_lbrs[i][3] - a_lbrs[i][1]);
            float b_area = (b_lbrs[j][2] - b_lbrs[j][0]) * (b_lbrs[j][3] - b_lbrs[j][1]);
            float union_area = a_area + b_area - inter_area;

            overlaps[i][j] = (union_area > 0) ? inter_area / union_area : 0.0;
        }
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            overlaps[i][j] = 1 - overlaps[i][j];
        }
    }

    return std::make_tuple(overlaps, N, M);
}

STrack::STrack() : score(0.0f), state(TrackState::New), is_activated(false),
                   track_id(-1), frame_id(0), start_frame(0), tracklet_len(0) {
    _tlwh = {0.0f, 0.0f, 0.0f, 0.0f};
}

STrack::STrack(const std::vector<float>& tlwh, float score)
    : _tlwh(tlwh), score(score), state(TrackState::New), is_activated(false),
      track_id(-1), frame_id(0), start_frame(0), tracklet_len(0) {
    assert(tlwh.size() == 4);
}

std::vector<float> STrack::tlwh() const {
    if (mean.empty()) {
        return _tlwh;
    }
    std::vector<float> ret = mean;
    ret[2] *= ret[3]; 
    ret[0] -= ret[2] / 2; 
    ret[1] -= ret[3] / 2; 
    return ret;
}

std::vector<float> STrack::tlwh_to_xyah(const std::vector<float>& tlwh) const {
    std::vector<float> ret = tlwh;
    ret[0] += ret[2] / 2;
    ret[1] += ret[3] / 2;
    ret[2] /= ret[3];
    return ret;
}

void STrack::mark_lost() {
    state = TrackState::Lost;
}

void STrack::mark_removed() {
    state = TrackState::Removed;
}

void STrack::predict() {
    if (mean.empty()) return;
    std::vector<float> mean_state = mean;
    if (state != TrackState::Tracked) {
        mean_state[7] = 0;
    }
    auto [predicted_mean, predicted_covariance] = kalman_filter->predict(mean_state, covariance);
    mean = predicted_mean;
    covariance = predicted_covariance;
}

void STrack::multi_predict(std::vector<STrack>& stracks) {
    if (stracks.empty()) return;

    std::vector<std::vector<float>> multi_mean;
    std::vector<std::vector<std::vector<float>>> multi_covariance;

    for (const auto& st : stracks) {
        multi_mean.push_back(st.mean); 
        multi_covariance.push_back(st.covariance); 
    }

    for (size_t i = 0; i < stracks.size(); ++i) {
        if (stracks[i].state != TrackState::Tracked) {
            multi_mean[i][7] = 0;
        }
    }

    auto [predicted_mean, predicted_covariance] = shared_kalman.multi_predict(multi_mean, multi_covariance);
    for (size_t i = 0; i < stracks.size(); ++i) {
        stracks[i].mean = predicted_mean[i];
        stracks[i].covariance = predicted_covariance[i];
    }
}

void STrack::activate(KalmanFilter* kalman_filter, int frame_id, int track_id) {
    this->kalman_filter = kalman_filter;
    this->track_id = track_id;

    auto [initial_mean, initial_covariance] = kalman_filter->initiate(tlwh_to_xyah(_tlwh));
    this->mean = initial_mean;
    this->covariance = initial_covariance;
    this->tracklet_len = 0;
    this->state = TrackState::Tracked;

    if (frame_id == 1) {
        is_activated = true;
    }

    this->frame_id = frame_id;
    this->start_frame = frame_id;
}

void STrack::re_activate(const STrack& new_track, int frame_id) {
    auto [updated_mean, updated_covariance] = kalman_filter->update(mean, covariance, tlwh_to_xyah(new_track._tlwh));
    mean = updated_mean;
    covariance = updated_covariance;
    tracklet_len = 0;
    state = TrackState::Tracked;
    is_activated = true;
    this->frame_id = frame_id;
    score = new_track.score;
}

void STrack::update(const STrack& new_track, int frame_id) {
    this->frame_id = frame_id;
    tracklet_len += 1;

    std::vector<float> new_tlwh = new_track._tlwh;
    auto [updated_mean, updated_covariance] = kalman_filter->update(mean, covariance, tlwh_to_xyah(new_tlwh));
    mean = updated_mean;
    covariance = updated_covariance;
    state = TrackState::Tracked;
    is_activated = true;

    score = new_track.score;
}

std::vector<std::vector<float>> get_tlwh(const std::vector<STrack>& stracks) {
    std::vector<std::vector<float>> tlwhs;
    for (const auto& strack : stracks) {
        tlwhs.push_back(strack.tlwh());
    }
    return tlwhs;
}

// 合并两个轨迹列表
std::vector<STrack> joint_stracks(const std::vector<STrack>& tlista, const std::vector<STrack>& tlistb) {
    std::unordered_map<int, bool> exists;
    std::vector<STrack> res;

    for (const auto& t : tlista) {
        exists[t.track_id] = true;
        res.push_back(t);
    }

    for (const auto& t : tlistb) {
        if (!exists[t.track_id]) {
            exists[t.track_id] = true;
            res.push_back(t);
        }
    }
    return res;
}

// 从 tlista 中去掉 tlistb 中的轨迹
std::vector<STrack> sub_stracks(const std::vector<STrack>& tlista, const std::vector<STrack>& tlistb) {
    std::unordered_map<int, STrack> stracks;

    for (const auto& t : tlista) {
        stracks[t.track_id] = t;
    }

    for (const auto& t : tlistb) {
        stracks.erase(t.track_id);
    }

    std::vector<STrack> result;
    for (const auto& entry : stracks) {
        result.push_back(entry.second);
    }
    return result;
}

// 移除重复的轨迹
std::pair<std::vector<STrack>, std::vector<STrack>> remove_duplicate_stracks(const std::vector<STrack>& stracksa,
                                                                             const std::vector<STrack>& stracksb) {
    if (stracksa.empty() || stracksb.empty()) {
        return {stracksa, stracksb};
    }
    auto [pdist, N, M] = iou_distance(get_tlwh(stracksa), get_tlwh(stracksb));
    std::vector<int> dupa, dupb;

    for (size_t i = 0; i < pdist.size(); ++i) {
        for (size_t j = 0; j < pdist[i].size(); ++j) {
            if (pdist[i][j] < 0.15) {
                int timep = stracksa[i].frame_id - stracksa[i].start_frame;
                int timeq = stracksb[j].frame_id - stracksb[j].start_frame;
                if (timep > timeq) {
                    dupb.push_back(j);
                } else {
                    dupa.push_back(i);
                }
            }
        }
    }

    std::vector<STrack> resa;
    std::vector<STrack> resb;

    for (size_t i = 0; i < stracksa.size(); ++i) {
        if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }
    for (size_t i = 0; i < stracksb.size(); ++i) {
        if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }

    return {resa, resb};
}

void mark_lost_stracks(std::vector<STrack>& stracks, int track_id) {
    for (int i = 0; i < (int)stracks.size(); i++) {
        if (stracks[i].track_id == track_id) {
            stracks[i].state = TrackState::Lost;
        }
    }
}

void mark_removed_stracks(std::vector<STrack>& stracks, int track_id) {
    for (int i = 0; i < (int)stracks.size(); i++) {
        if (stracks[i].track_id == track_id) {
            stracks[i].state = TrackState::Removed;
        }
    }
}

KalmanFilter STrack::shared_kalman;

BYTETracker::BYTETracker(int frame_rate, float low_conf_thresh, float high_conf_thresh)
    : det_low_conf_thresh(low_conf_thresh), det_high_conf_thresh(high_conf_thresh),
      frame_id(0), track_id(0), max_time_lost(frame_rate) {}

std::vector<STrack> BYTETracker::update(const std::vector<Detection>& det_results) {
    frame_id++;
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> removed_stracks;

    std::vector<int> remain_inds;
    std::vector<int> inds_low;
    for (size_t i = 0; i < det_results.size(); ++i) {
        if (det_results[i].score > det_high_conf_thresh) {
            remain_inds.push_back(i);
        } else if (det_results[i].score > det_low_conf_thresh) {
            inds_low.push_back(i);
        }
    }

    std::vector<STrack> detections;
    for (const auto& i : remain_inds) {
        detections.emplace_back(STrack({det_results[i].left, det_results[i].top, det_results[i].width, det_results[i].height}, det_results[i].score));
    }

    std::vector<STrack> unconfirmed;
    std::vector<STrack> tracked_stracks;
    for (const auto& track : all_tracked_stracks) {
        if (!track.is_activated) {
            unconfirmed.push_back(track);
        } else {
            tracked_stracks.push_back(track);
        }
    }

    std::vector<STrack> strack_pool = joint_stracks(tracked_stracks, all_lost_stracks);
    STrack::multi_predict(strack_pool);
    for (auto & strack: all_lost_stracks){
        bool update_flag = false;
        for (auto & strack2: strack_pool){
            if (strack.track_id == strack2.track_id){
                update_flag = true;
                break;
            }
        }
        if (update_flag){
            strack.predict();
        }
    }

    auto [dists, N, M] = iou_distance(get_tlwh(strack_pool), get_tlwh(detections));
    auto [matches, u_track, u_detection1] = linear_assignment(dists, 0.5f, N, M);
    
    for (const auto& match : matches) {
        int itracked = match.first;
        int idet = match.second;
        auto& track = strack_pool[itracked];
        auto& det = detections[idet];

        if (track.state == TrackState::Tracked) {
            track.update(det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track.re_activate(det, frame_id);
            refind_stracks.push_back(track);
        }
    }

    std::vector<STrack> detections_second;
    for (const auto& i : inds_low) {
        detections_second.emplace_back(STrack({det_results[i].left, det_results[i].top, det_results[i].width, det_results[i].height}, det_results[i].score));
    }

    auto r_tracked_stracks = get_tracked_stracks(strack_pool, u_track);
    tie(dists, N, M) = iou_distance(get_tlwh(r_tracked_stracks), get_tlwh(detections_second));

    auto [matches2, u_track2, u_detection2] = linear_assignment(dists, 0.5f, N, M);

    for (const auto& match : matches2) {
        int itracked = match.first;
        int idet = match.second;
        auto& track = r_tracked_stracks[itracked];
        auto& det = detections_second[idet];

        if (track.state == TrackState::Tracked) {
            track.update(det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track.re_activate(det, frame_id);
            refind_stracks.push_back(track);
        }
    }

    for (const auto& it : u_track2) {
        auto& track = r_tracked_stracks[it];
        if (track.state != TrackState::Lost) {
            track.state = TrackState::Lost;
            for (auto& strack : all_tracked_stracks) {
                if (strack.track_id == track.track_id) {
                    strack.state = TrackState::Lost;
                }
            }
            lost_stracks.push_back(track);
        }
    }

    auto detections3 = filter_detections(detections, u_detection1);
    tie(dists, N, M) = iou_distance(get_tlwh(unconfirmed), get_tlwh(detections3));
    auto [matches3, u_unconfirmed3, u_detection3] = linear_assignment(dists, 0.7f, N, M);

    for (const auto& match : matches3) {
        unconfirmed[match.first].update(detections3[match.second], frame_id);
        activated_stracks.push_back(unconfirmed[match.first]);
    }

    for (const auto& it : u_unconfirmed3) {
        auto& track = unconfirmed[it];
        track.state = TrackState::Removed;
        for (auto& strack : all_tracked_stracks) {
            if (strack.track_id == track.track_id) {
                strack.state = TrackState::Removed;
            }
        }
        for (auto& strack : all_lost_stracks) {
            if (strack.track_id == track.track_id) {
                strack.state = TrackState::Removed;
            }
        }
        removed_stracks.push_back(track);
    }

    for (const auto& i : u_detection3) {
        auto& track = detections3[i];
        if (track.score < det_high_conf_thresh) {
            continue;
        }
        track.activate(new KalmanFilter(), frame_id, track_id);
        track_id++;
        activated_stracks.push_back(track);
    }

    for (auto& track : all_lost_stracks) {
        if (frame_id - track.frame_id > max_time_lost){
            track.state = TrackState::Removed;
            removed_stracks.push_back(track);
        }
    }

    all_tracked_stracks.erase(std::remove_if(all_tracked_stracks.begin(), all_tracked_stracks.end(),
        [](const STrack& track) {
            return track.state != TrackState::Tracked;
        }), all_tracked_stracks.end());

    all_tracked_stracks = joint_stracks(activated_stracks, all_tracked_stracks);
    all_tracked_stracks = joint_stracks(refind_stracks, all_tracked_stracks);
    all_lost_stracks = sub_stracks(all_lost_stracks, all_tracked_stracks);
    all_lost_stracks.insert(all_lost_stracks.end(), lost_stracks.begin(), lost_stracks.end());
    all_lost_stracks = sub_stracks(all_lost_stracks, all_removed_stracks);
    all_removed_stracks.insert(all_removed_stracks.end(), removed_stracks.begin(), removed_stracks.end());

    auto [unique_tracked, unique_lost] = remove_duplicate_stracks(all_tracked_stracks, all_lost_stracks);
    all_tracked_stracks = unique_tracked;
    all_lost_stracks = unique_lost;
    cout<<"all_tracked_stracks: "<<all_tracked_stracks.size()<<" all_lost_stracks: "<<all_lost_stracks.size()<<" all_removed_stracks: "<<all_removed_stracks.size()<<endl;

    std::vector<STrack> output_stracks;
    for (const auto& track : all_tracked_stracks) {
        if (track.is_activated) {
            output_stracks.push_back(track);
        }
    }

    return output_stracks;
}

std::vector<STrack> BYTETracker::get_tracked_stracks(const std::vector<STrack>& strack_pool, const std::vector<int>& u_track) {
    std::vector<STrack> tracked_stracks;

    for (int index : u_track) {
        if (index < 0 || index >= strack_pool.size()) {
            continue;
        }
        const STrack& track = strack_pool[index];
        if (track.state == TrackState::Tracked) {
            tracked_stracks.push_back(track);
        }
    }

    return tracked_stracks;
}

std::vector<STrack> BYTETracker::filter_detections(const std::vector<STrack>& detections, const std::vector<int>& u_detection) {
    std::vector<STrack> filtered_detections;

    for (int index : u_detection) {
        if (index < 0 || index >= detections.size()) {
            continue;
        }
        filtered_detections.push_back(detections[index]);
    }

    return filtered_detections;
}