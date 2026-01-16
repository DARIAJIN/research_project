import os
import subprocess

CASES = ["s0970", "s1023", "s1101", "s1188"]
N_REPEAT = 5
FM_DOWNSAMPLE = 8

for case_id in CASES:
    for run in range(N_REPEAT):
        out_dir = f"results_dinoreg/case_{case_id}/run_{run}"
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "inference_l2rmrct.py",
            "--case_id", case_id,
            "--output_dir", out_dir,
            "--fm_downsample", str(FM_DOWNSAMPLE),
            "--exp_note", f"run_{run}"
        ]

        subprocess.run(cmd, check=True)
