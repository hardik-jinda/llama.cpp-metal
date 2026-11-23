#pragma once

/* Opt-in step timing trace (LLAMA_SERVER_TRACE_TIMING=1), the same records and /debug endpoints as the vLLM-Metal
   metal_timing plugin, so profile_exact_sweep.py drives both servers unchanged. A step is one synchronized
   llama_process() call of update_slots() plus the sampling after it. Benchmark instrumentation, not for upstream. */

#include "server-common.h"
#include "server-http.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct server_trace_step {
    int64_t t_start_us = 0;
    int64_t t_fwd_end_us = 0;
    std::vector<int> prefill_ids; // tasks with prompt tokens in this step
    std::vector<int> done_ids;    // tasks whose prompt ends in this step
    std::vector<int> decode_ids;  // tasks sampled from a generated token
    std::vector<int> active_ids;  // tasks processing before sampling
};

struct server_trace {
    const bool enabled = [] {
        const char * v = std::getenv("LLAMA_SERVER_TRACE_TIMING");
        return v != nullptr && std::string(v) == "1";
    }();
    std::atomic<int> n_running{0};

    std::mutex mtx;
    int64_t n_steps = 0;
    std::deque<json> steps;
    std::unordered_map<int, double> prefill_ms; // task id -> forward ms summed over its prompt steps
    std::map<int, json> requests;               // task id -> request timing record

    static std::string req_id(int id) { return "task-" + std::to_string(id); }

    static double wall_s() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void add_request_ms(int id, const char * key, double ms) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = requests.find(id);
        if (it == requests.end()) {
            it = requests.emplace(id, json{{"t", "req"}, {"rid", req_id(id)}, {"request_id", req_id(id)},
                                           {"ts", wall_s()}, {"timing", json::object()}}).first;
            while (requests.size() > 100000) {
                requests.erase(requests.begin());
            }
        }
        json & timing = it->second["timing"];
        timing[key] = timing.value(key, 0.0) + ms;
    }

    void record(const server_trace_step & s, int64_t t_end_us, double sampling_ms, const std::vector<int> & finished) {
        const double forward_ms = (s.t_fwd_end_us - s.t_start_us) / 1000.0;
        const double ts_last    = wall_s();
        std::lock_guard<std::mutex> lock(mtx);
        json prefills = json::array();
        for (int id : s.prefill_ids) {
            prefill_ms[id] += forward_ms;
        }
        for (int id : s.done_ids) {
            prefills.push_back({{"req_id", req_id(id)}, {"ms", prefill_ms[id]}, {"sampling_ms", sampling_ms}});
            prefill_ms.erase(id);
        }
        json reqs = json::array();
        for (int id : s.decode_ids) {
            reqs.push_back(req_id(id));
        }
        json done = json::array();
        for (int id : finished) {
            done.push_back(req_id(id));
        }
        const bool decoded = !s.decode_ids.empty();
        const int64_t step_id = ++n_steps;
        steps.push_back({
            {"t", "step"}, {"step_id", step_id}, {"step_id_first", step_id}, {"step_id_last", step_id},
            {"mono_ns_first", s.t_start_us * 1000}, {"mono_ns_last", t_end_us * 1000},
            {"ts_first", ts_last - (t_end_us - s.t_start_us) / 1e6}, {"ts_last", ts_last},
            {"prefills", prefills},
            {"decode", {
                {"reqs", reqs},
                {"tokens", (int) s.decode_ids.size()},
                {"forward_ms_samples", decoded ? json::array({forward_ms}) : json::array()},
                {"sampling_ms_samples", decoded ? json::array({sampling_ms}) : json::array()},
            }},
            {"finished", done},
        });
        while (steps.size() > 100000) {
            steps.pop_front();
        }
    }

    static json page(const std::vector<json> & records, const server_http_req & req, const char * key) {
        const size_t offset = std::stoul(req.get_param("cursor", "0"));
        const size_t limit  = std::max<size_t>(1, std::min<size_t>(std::stoul(req.get_param("limit", "1000")), 1000));
        const size_t end    = std::min(records.size(), offset + limit);
        json out = json::array();
        for (size_t i = std::min(offset, records.size()); i < end; i++) {
            out.push_back(records[i]);
        }
        const bool more = end < records.size();
        return {{key, out}, {"next_cursor", more ? json(std::to_string(end)) : json(nullptr)}, {"truncated", more}};
    }

    json steps_page(const server_http_req & req) {
        const int64_t after   = std::stoll(req.get_param("after_step_id", "-1"));
        const int64_t through = std::stoll(req.get_param("through_step_id", "-1"));
        std::vector<json> records;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto & s : steps) {
                const int64_t id = s.at("step_id").get<int64_t>();
                if (id > after && (through < 0 || id <= through)) {
                    records.push_back(s);
                }
            }
        }
        if (req.get_param("order", "asc") == "desc") {
            std::reverse(records.begin(), records.end());
        }
        return page(records, req, "records");
    }

    json requests_page(const server_http_req & req) {
        const double since = std::stod(req.get_param("since", "-1"));
        const double until = std::stod(req.get_param("until", "1e300"));
        std::vector<json> records;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto & [id, r] : requests) {
                const double ts = r.at("ts").get<double>();
                if (ts >= since && ts <= until) {
                    records.push_back(r);
                }
            }
        }
        // vLLM-Metal pages its records newest first unless order=asc
        if (req.get_param("order", "latest") == "latest") {
            std::reverse(records.begin(), records.end());
        }
        return page(records, req, "requests");
    }

    json metrics() {
        std::lock_guard<std::mutex> lock(mtx);
        return {{"current", {{"running", n_running.load()}, {"waiting", 0}}}, {"totals", {{"total_steps", n_steps}}}};
    }
};

inline server_trace & server_trace_get() {
    static server_trace trace;
    return trace;
}
