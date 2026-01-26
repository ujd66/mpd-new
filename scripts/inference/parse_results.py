import re
import numpy as np


def parse_log(filename):
    with open(filename, "r") as f:
        content = f.read()

    tasks = []
    # Split by PLANNING header
    parts = re.split(r"-+PLANNING \d+/10-+", content)
    # Skip the first part (header stuff)
    for i, part in enumerate(parts[1:]):
        task_data = {}

        # Extract Start/Goal for verification
        start_match = re.search(r"q_pos_start:\s*tensor\(\[(.*?)\]", part, re.DOTALL)
        goal_match = re.search(r"q_pos_goal:\s*tensor\(\[(.*?)\]", part, re.DOTALL)

        if start_match:
            # simple string signature
            task_data["start"] = start_match.group(1).replace("\n", "").replace(" ", "")
        if goal_match:
            task_data["goal"] = goal_match.group(1).replace("\n", "").replace(" ", "")

        # Extract Metrics
        # t_inference_total: 0.748 sec
        time_match = re.search(r"t_inference_total:\s*([\d\.]+)\s*sec", part)
        if time_match:
            task_data["time"] = float(time_match.group(1))

        # Parse Python dict output for metrics
        # 'success': 1,
        success_match = re.search(r"'success':\s*(\d+)", part)
        if success_match:
            task_data["success"] = int(success_match.group(1))

        # 'path_length': array(7.302, dtype=float32),
        pl_match = re.search(r"'path_length':\s*array\(([\d\.]+)", part)
        if pl_match:
            task_data["path_length"] = float(pl_match.group(1))

        # 'smoothness': array(114.304, dtype=float32)
        sm_match = re.search(r"'smoothness':\s*array\(([\d\.]+)", part)
        if sm_match:
            task_data["smoothness"] = float(sm_match.group(1))

        tasks.append(task_data)
    return tasks


print("Parsing MPD Logs...")
mpd_tasks = parse_log("mpd_results.log")
print(f"Found {len(mpd_tasks)} tasks in MPD log")

print("Parsing Hybrid Logs...")
hybrid_tasks = parse_log("hybrid_results.log")
print(f"Found {len(hybrid_tasks)} tasks in Hybrid log")

print("\n" + "=" * 80)
print(
    f"{'Task':<5} | {'Consistency':<12} | {'MPD Time':<10} | {'Hybrid Time':<12} | {'MPD PL':<8} | {'Hybrid PL':<10} | {'MPD Sm':<8} | {'Hybrid Sm':<10}"
)
print("-" * 80)

mpd_stats = {"time": [], "pl": [], "sm": [], "success": []}
hybrid_stats = {"time": [], "pl": [], "sm": [], "success": []}

for i in range(min(len(mpd_tasks), len(hybrid_tasks))):
    m = mpd_tasks[i]
    h = hybrid_tasks[i]

    # Check consistency
    is_consistent = (m.get("start") == h.get("start")) and (m.get("goal") == h.get("goal"))
    consistency_str = "OK" if is_consistent else "DIFF!"

    # Metrics
    m_time = m.get("time", 0)
    h_time = h.get("time", 0)
    m_pl = m.get("path_length", 0)
    h_pl = h.get("path_length", 0)
    m_sm = m.get("smoothness", 0)
    h_sm = h.get("smoothness", 0)

    mpd_stats["time"].append(m_time)
    mpd_stats["pl"].append(m_pl)
    mpd_stats["sm"].append(m_sm)
    mpd_stats["success"].append(m.get("success", 0))

    hybrid_stats["time"].append(h_time)
    hybrid_stats["pl"].append(h_pl)
    hybrid_stats["sm"].append(h_sm)
    hybrid_stats["success"].append(h.get("success", 0))

    print(
        f"{i+1:<5} | {consistency_str:<12} | {m_time:<10.3f} | {h_time:<12.3f} | {m_pl:<8.2f} | {h_pl:<10.2f} | {m_sm:<8.1f} | {h_sm:<10.1f}"
    )

print("=" * 80)
print("\nAverage Stats:")
print(f"MPD Success Rate: {np.mean(mpd_stats['success'])*100}%")
print(f"Hybrid Success Rate: {np.mean(hybrid_stats['success'])*100}%")
print(f"MPD Avg Time: {np.mean(mpd_stats['time']):.3f}s")
print(f"Hybrid Avg Time: {np.mean(hybrid_stats['time']):.3f}s")
print(f"MPD Avg Path Length: {np.mean(mpd_stats['pl']):.3f}")
print(f"Hybrid Avg Path Length: {np.mean(hybrid_stats['pl']):.3f}")
print(f"MPD Avg Smoothness: {np.mean(mpd_stats['sm']):.3f}")
print(f"Hybrid Avg Smoothness: {np.mean(hybrid_stats['sm']):.3f}")
