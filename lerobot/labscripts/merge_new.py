import argparse
import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd

def detect_feature_max_dims(source_folders, features=["observation.state", "action"]):
    max_dims = {feat: 0 for feat in features}
    for folder in source_folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(os.path.join(root, file))
                        for feat in features:
                            if feat in df.columns:
                                for val in df[feat].dropna():
                                    if isinstance(val, (list, np.ndarray)):
                                        max_dims[feat] = max(max_dims[feat], len(val))
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
    print(f"[INFO] Auto-detected feature max dims: {max_dims}")
    return max_dims

def load_jsonl(file_path):
    data = []
    if "episodes_stats.jsonl" in file_path:
        try:
            with open(file_path) as f:
                content = f.read()
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def pad_vector(x, target_dim):
    # Always convert to 1D array
    arr = np.array(x)
    if arr.ndim == 0:
        arr = np.array([arr])
    if len(arr) < target_dim:
        arr = np.pad(arr, (0, target_dim - len(arr)), 'constant')
    elif len(arr) > target_dim:
        arr = arr[:target_dim]
    return arr.tolist()

def merge_stats(stats_list, max_dims):
    merged_stats = {}
    common_features = set(stats_list[0].keys())
    for stats in stats_list[1:]:
        common_features = common_features.intersection(set(stats.keys()))
    for feature in stats_list[0]:
        if feature not in common_features:
            continue
        merged_stats[feature] = {}
        common_stat_types = []
        for stat_type in ["mean", "std", "max", "min"]:
            if all(stat_type in stats[feature] for stats in stats_list):
                common_stat_types.append(stat_type)
        # Always use correct dimension for state/action features
        feat_dim = max_dims[feature] if feature in max_dims else None
        for stat_type in common_stat_types:
            values = [stats[feature][stat_type] for stats in stats_list]
            if feat_dim is not None and isinstance(values[0], list):
                # Pad all vectors
                padded = [pad_vector(val, feat_dim) for val in values]
                arr = np.array(padded)
                if stat_type == "mean":
                    merged_stats[feature][stat_type] = np.mean(arr, axis=0).tolist()
                elif stat_type == "std":
                    merged_stats[feature][stat_type] = np.std(arr, axis=0).tolist()
                elif stat_type == "max":
                    merged_stats[feature][stat_type] = np.max(arr, axis=0).tolist()
                elif stat_type == "min":
                    merged_stats[feature][stat_type] = np.min(arr, axis=0).tolist()
            else:
                try:
                    merged_stats[feature][stat_type] = np.mean(np.array(values), axis=0).tolist()
                except Exception:
                    merged_stats[feature][stat_type] = values[0]
        # Add count if available
        if all("count" in stats[feature] for stats in stats_list):
            try:
                merged_stats[feature]["count"] = [sum(stats[feature]["count"][0] for stats in stats_list)]
            except Exception as e:
                print(f"Warning: Error processing {feature}.count: {e}")
    return merged_stats

def copy_videos(source_folders, output_folder, episode_mapping):
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    video_path_template = info["video_path"]
    video_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)
    print(f"Found video keys: {video_keys}")
    for old_folder, old_index, new_index in episode_mapping:
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]
        for video_key in video_keys:
            source_patterns = [
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk, video_key=video_key, episode_index=old_index
                    ),
                ),
            ]
            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break
            if source_video_path:
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk, video_key=video_key, episode_index=new_index
                    ),
                )
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)
                print(f"Copying video: {source_video_path} -> {dest_video_path}")
                shutil.copy2(source_video_path, dest_video_path)

def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    max_dims,
    fps=None,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    chunks_size=1000,
    default_fps=20,
):
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get("fps", default_fps)
        else:
            fps = default_fps
    print(f"使用FPS={fps} (Using FPS={fps})")
    total_copied = 0
    total_failed = 0
    failed_files = []
    for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
        episode_str = f"episode_{old_index:06d}.parquet"
        source_paths = [
            os.path.join(old_folder, "parquet", episode_str),
            os.path.join(old_folder, "data", episode_str),
        ]
        source_path = None
        for path in source_paths:
            if os.path.exists(path):
                source_path = path
                break
        if source_path:
            try:
                df = pd.read_parquet(source_path)
                # Feature-wise padding
                for feature in max_dims:
                    if feature in df.columns:
                        feat_max_dim = max_dims[feature]
                        df[feature] = df[feature].apply(lambda x: pad_vector(x, feat_max_dim))
                if "episode_index" in df.columns:
                    print(
                        f"更新episode_index从 {df['episode_index'].iloc[0]} 到 {new_index} (Update episode_index from {df['episode_index'].iloc[0]} to {new_index})"
                    )
                    df["episode_index"] = new_index
                if "index" in df.columns:
                    if episode_to_frame_index and new_index in episode_to_frame_index:
                        first_index = episode_to_frame_index[new_index]
                    else:
                        first_index = new_index * len(df)
                    df["index"] = [first_index + i for i in range(len(df))]
                if "task_index" in df.columns and folder_task_mapping and old_folder in folder_task_mapping:
                    current_task_index = df["task_index"].iloc[0]
                    if current_task_index in folder_task_mapping[old_folder]:
                        new_task_index = folder_task_mapping[old_folder][current_task_index]
                        df["task_index"] = new_task_index
                chunk_index = new_index // chunks_size
                chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                os.makedirs(chunk_dir, exist_ok=True)
                dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")
                df.to_parquet(dest_path, index=False)
                total_copied += 1
                print(f"已处理并保存: {dest_path} (Processed and saved: {dest_path})")
            except Exception as e:
                error_msg = f"处理 {source_path} 失败: {e} (Processing {source_path} failed: {e})"
                print(error_msg)
                traceback.print_exc()
                failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                total_failed += 1
    print(f"共复制 {total_copied} 个数据文件，{total_failed} 个失败")
    if failed_files:
        print("\n失败的文件详情 (Details of failed files):")
        for i, failed in enumerate(failed_files):
            print(f"{i + 1}. 文件 (File): {failed['file']}")
            if "folder" in failed:
                print(f"   文件夹 (Folder): {failed['folder']}")
            if "episode" in failed:
                print(f"   Episode索引 (Episode index): {failed['episode']}")
            print(f"   原因 (Reason): {failed['reason']}")
            print("---")
    return total_copied > 0

def merge_datasets(
    source_folders, output_folder, max_dims, validate_ts=False, tolerance_s=1e-4, default_fps=20
):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)
    fps = default_fps
    print(f"使用默认FPS值: {fps}")
    all_episodes = []
    all_episodes_stats = []
    all_tasks = []
    total_frames = 0
    total_episodes = 0
    episode_mapping = []
    all_stats_data = []
    folder_dimensions = {}
    cumulative_frame_count = 0
    episode_to_frame_index = {}
    task_desc_to_new_index = {}
    folder_task_mapping = {}
    all_unique_tasks = []
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    chunks_size = 1000
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)
    total_videos = 0
    for folder in source_folders:
        try:
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(folder_info_path):
                with open(folder_info_path) as f:
                    folder_info = json.load(f)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
            folder_dim = max_dims["observation.state"]
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(os.path.join(root, file))
                            if "observation.state" in df.columns:
                                first_state = df["observation.state"].iloc[0]
                                if isinstance(first_state, (list, np.ndarray)):
                                    folder_dim = len(first_state)
                                    break
                        except Exception as e:
                            print(f"Error checking dimensions in {folder}: {e}")
                        break
                if folder_dim != max_dims["observation.state"]:
                    break
            folder_dimensions[folder] = folder_dim
            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue
            episodes = load_jsonl(episodes_path)
            episodes_stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
            episodes_stats = []
            if os.path.exists(episodes_stats_path):
                episodes_stats = load_jsonl(episodes_stats_path)
            stats_map = {}
            for stat in episodes_stats:
                if "episode_index" in stat:
                    stats_map[stat["episode_index"]] = stat
            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            folder_tasks = []
            if os.path.exists(tasks_path):
                folder_tasks = load_jsonl(tasks_path)
            folder_task_mapping[folder] = {}
            for task in folder_tasks:
                task_desc = task["task"]
                old_index = task["task_index"]
                if task_desc not in task_desc_to_new_index:
                    new_index = len(all_unique_tasks)
                    task_desc_to_new_index[task_desc] = new_index
                    all_unique_tasks.append({"task_index": new_index, "task": task_desc})
                folder_task_mapping[folder][old_index] = task_desc_to_new_index[task_desc]
            for episode in episodes:
                old_index = episode["episode_index"]
                new_index = total_episodes
                episode["episode_index"] = new_index
                all_episodes.append(episode)
                if old_index in stats_map:
                    stats = stats_map[old_index]
                    stats["episode_index"] = new_index
                    # Pad stats for each feature
                    if "stats" in stats:
                        for feature in max_dims:
                            if feature in stats["stats"]:
                                for stat_type in ["mean", "std", "max", "min"]:
                                    if stat_type in stats["stats"][feature]:
                                        stats["stats"][feature][stat_type] = pad_vector(
                                            stats["stats"][feature][stat_type], max_dims[feature]
                                        )
                    all_episodes_stats.append(stats)
                    if "stats" in stats:
                        all_stats_data.append(stats["stats"])
                episode_mapping.append((folder, old_index, new_index))
                total_episodes += 1
                total_frames += episode["length"]
                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]
            all_tasks = all_unique_tasks
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")
    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(all_episodes_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))
    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                stats_list.append(stats)
    if stats_list:
        merged_stats = merge_stats(stats_list, max_dims)
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info["chunks_size"]
    info["splits"] = {"train": f"0:{total_episodes}"}
    if "features" in info:
        for feature_name in max_dims:
            if feature_name in info["features"] and "shape" in info["features"][feature_name]:
                info["features"][feature_name]["shape"] = [max_dims[feature_name]]
                print(f"Updated {feature_name} shape to {max_dims[feature_name]}")
    info["total_videos"] = total_videos
    print(f"更新视频总数为: {total_videos} (Update total videos to: {total_videos})")
    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        max_dims=max_dims,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        chunks_size=chunks_size,
    )
    print(f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets from multiple sources.")
    parser.add_argument("--sources", nargs="+", required=True, help="List of source folder paths")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--max_dim_state", type=int, default=None, help="Maximum dimension for observation.state (auto-detect if not set)")
    parser.add_argument("--max_dim_action", type=int, default=None, help="Maximum dimension for action (auto-detect if not set)")
    parser.add_argument("--fps", type=int, default=20, help="Your datasets FPS (default: 20)")
    args = parser.parse_args()
    features = ["observation.state", "action"]
    detected_max_dims = detect_feature_max_dims(args.sources, features)
    max_dims = {
        "observation.state": args.max_dim_state if args.max_dim_state is not None else detected_max_dims["observation.state"],
        "action": args.max_dim_action if args.max_dim_action is not None else detected_max_dims["action"],
    }
    merge_datasets(args.sources, args.output, max_dims=max_dims, default_fps=args.fps)
