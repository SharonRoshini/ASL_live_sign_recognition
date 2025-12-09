import json
import os
import time
from multiprocessing import Pool
import torch

### ---------------------------
### YOUR PATHS (EDIT IF NEEDED)
### ---------------------------

POSE_ROOT = "C:/Jayasri/PFW/ADL-Project/TGCN-Training/data/pose_per_individual_videos"
FEATURE_ROOT = "C:/Jayasri/PFW/ADL-Project/TGCN-Training/data/features"
INDEX_FILE = "C:/Jayasri/PFW/ADL-Project/TGCN-Training/data/splits/asl100.json"

os.makedirs(FEATURE_ROOT, exist_ok=True)

body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}


### ---------------------------
### SAFE DIFF COMPUTATION
### ---------------------------

def compute_difference(x):
    n = len(x)
    diff = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff[i, j] = x[i] - x[j]
    return diff


### ---------------------------
### PROCESS ONE SPLIT OF ENTRIES
### ---------------------------

def gen(entries):
    for idx, entry in enumerate(entries):

        for inst in entry["instances"]:
            vid = inst["video_id"]
            fstart = inst["frame_start"]
            fend = inst["frame_end"]

            pose_dir = os.path.join(POSE_ROOT, vid)
            if not os.path.exists(pose_dir):
                print(f"Missing pose folder for {vid}")
                continue

            save_dir = os.path.join(FEATURE_ROOT, vid)
            os.makedirs(save_dir, exist_ok=True)

            for frame in range(fstart, fend + 1):
                frame_name = f"image_{str(frame).zfill(5)}"
                json_path = os.path.join(pose_dir, frame_name + "_keypoints.json")
                ft_path = os.path.join(save_dir, frame_name + "_ft.pt")

                if os.path.exists(ft_path):
                    continue

                if not os.path.exists(json_path):
                    continue

                try:
                    people = json.load(open(json_path)).get("people", [])
                    if len(people) == 0:
                        continue
                    kp = people[0]
                except:
                    continue

                # collect all poses
                pts = kp["pose_keypoints_2d"] + kp["hand_left_keypoints_2d"] + kp["hand_right_keypoints_2d"]

                x = [pts[i] for i in range(0, len(pts), 3) if (i // 3) not in body_pose_exclude]
                y = [pts[i] for i in range(1, len(pts), 3) if (i // 3) not in body_pose_exclude]

                x = 2 * ((torch.tensor(x) / 256.0) - 0.5)
                y = 2 * ((torch.tensor(y) / 256.0) - 0.5)

                x_diff = compute_difference(x) / 2
                y_diff = compute_difference(y) / 2

                orient = torch.zeros_like(x_diff)
                mask = x_diff != 0
                orient[mask] = y_diff[mask] / x_diff[mask]

                xy = torch.stack([x, y], dim=1)
                ft = torch.cat([xy, x_diff.unsqueeze(2), y_diff.unsqueeze(2), orient.unsqueeze(2)], dim=2)

                torch.save(ft, ft_path)

        print(f"Finished entry {idx}")


### ---------------------------
### MAIN
### ---------------------------

if __name__ == "__main__":
    content = json.load(open(INDEX_FILE))

    # Split into chunks
    n = len(content)
    splits = [
        content[: n//3],
        content[n//3: 2*n//3],
        content[2*n//3:]
    ]

    print(f"Processing {n} gloss entries using multiprocessing...")

    p = Pool(3)
    p.map(gen, splits)
    p.close()
    p.join()

    print("DONE.")
