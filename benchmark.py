import subprocess
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import boto3
import yaml

BENCHMARK_CMD = [
    "python3", "./vllm/benchmarks/benchmark_serving.py",
    "--backend", "openai",
    "--base-url", "http://127.0.0.1:8000",
    "--dataset-name", "random",
    "--model", "/models/qwen3-32b",
    "--max-concurrency", "160",
    "--num-prompts", "1200",
    "--request-rate", "30",
    "--seed", "12345",
    "--random-input-len", "64",
    "--random-output-len", "256",
    "--save-result",
    "--result-dir", "./results",
    "--result-filename", "results.json"
]
S3_BUCKET = "test-jinwook"
S3_KEY_SUMMARY = "test1/gpu_benchmark_summary.png"
CSV_PATH = "nvidia_smi_log.csv"
RESULT_PATH = "./results/results.json"
VLLM_CONFIG_PATH = "./config.yaml"

def run_benchmark_and_nvidia_smi():
    nvsmi_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free",
         "--format=csv", "-l", "1"],
        stdout=open(CSV_PATH, "w")
    )
    time.sleep(2)
    bench_proc = subprocess.Popen(BENCHMARK_CMD)
    bench_proc.wait()
    nvsmi_proc.terminate()
    try:
        nvsmi_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        nvsmi_proc.kill()

def parse_vllm_config_table():
    if not os.path.exists(VLLM_CONFIG_PATH):
        return []
    with open(VLLM_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    keys = [
        "dtype", "kv_cache_dtype", "tensor_parallel_size", "gpu_memory_utilization",
        "swap_space", "rope_scaling", "max_model_len", "max_seq_len_to_capture",
        "enable_auto_tool_choice", "tool_call_parser", "trust_remote_code"
    ]
    rows = [(k, str(config.get(k, ""))) for k in keys if k in config]
    return rows

def parse_bench_table(res):
    opt_rows = [
        ("Concurrency", res.get('max_concurrency', '')),
        ("Request Rate", f"{res.get('request_rate', '')}/s"),
        ("Num Prompts", res.get('num_prompts', '')),
        ("Total Input tokens", res.get('total_input_tokens', '')),
        ("Total Output tokens", res.get('total_output_tokens', '')),
        ("Model", res.get('model_id', ''))
    ]
    sum_rows = [
        ("[Result] TPS", f"{res.get('request_throughput', 0):.2f}"),
        ("[Result] Token/s", f"{res.get('total_token_throughput', 0):.1f}"),
        ("[Result] Mean TTFT", f"{res.get('mean_ttft_ms', 0):.1f} ms")
    ]
    return opt_rows, sum_rows

def analyze_and_plot():
    vllm_config_rows = parse_vllm_config_table()

    with open(RESULT_PATH) as f:
        res = json.load(f)

    opt_rows, sum_rows = parse_bench_table(res)

    gpu_df = pd.read_csv(CSV_PATH)
    gpu_df.columns = gpu_df.columns.str.strip()
    gpu_df['utilization.gpu [%]'] = gpu_df['utilization.gpu [%]'].astype(str).str.replace(' %','').astype(int)
    gpu_df['memory.used [MiB]'] = gpu_df['memory.used [MiB]'].astype(str).str.replace(' MiB','').astype(int)
    gpu_df['memory.total [MiB]'] = gpu_df['memory.total [MiB]'].astype(str).str.replace(' MiB','').astype(int)
    gpu_df['timestamp'] = pd.to_datetime(gpu_df['timestamp'], errors='coerce')
    gpu_df['memory.used.percent'] = (gpu_df['memory.used [MiB]'] / gpu_df['memory.total [MiB]'] * 100).round(1)

    grouped = gpu_df.groupby('timestamp')
    util_mean = grouped['utilization.gpu [%]'].mean()
    util_max = grouped['utilization.gpu [%]'].max()
    util_min = grouped['utilization.gpu [%]'].min()
    mem_mean = grouped['memory.used.percent'].mean()
    mem_max = grouped['memory.used.percent'].max()
    mem_min = grouped['memory.used.percent'].min()

    # --- subplot: 4행 1열, 그래프 높이 20% 감소
    fig, axs = plt.subplots(4, 1, figsize=(15, 16), gridspec_kw={'height_ratios': [1.2, 4, 4, 1.5]})

    # 상단 vLLM config table
    axs[0].axis('off')
    table_top = axs[0].table(
        cellText=[[k, v] for k, v in vllm_config_rows],
        colLabels=["vLLM Option", "Value"],
        loc='center',
        cellLoc='left',
        colColours=['#e2eaff', '#e2eaff'],
        colWidths=[0.40, 0.60]
    )
    table_top.auto_set_font_size(False)
    table_top.set_fontsize(12)
    table_top.scale(1.2, 1.18)
    # row 높이 50% 증가
    for (row, col), cell in table_top.get_celld().items():
        if row == 0:
            cell.set_height(0.20)  # 헤더
        else:
            cell.set_height(0.20)
    axs[0].margins(y=0.13)

    # 2번째: GPU Utilization
    axs[1].plot(util_mean.index, util_mean.values, label='Mean Util (%)')
    axs[1].plot(util_max.index, util_max.values, label='Max Util (%)', linestyle='--')
    axs[1].plot(util_min.index, util_min.values, label='Min Util (%)', linestyle=':')
    axs[1].set_title('GPU Utilization (mean/max/min)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Utilization (%)')
    axs[1].legend(loc='upper right')

    # 3번째: Memory Usage
    axs[2].plot(mem_mean.index, mem_mean.values, label='Mean Mem Usage (%)')
    axs[2].plot(mem_max.index, mem_max.values, label='Max Mem Usage (%)', linestyle='--')
    axs[2].plot(mem_min.index, mem_min.values, label='Min Mem Usage (%)', linestyle=':')
    axs[2].axhline(100, color='red', linestyle='--', label='100% (Max)')
    axs[2].set_title('GPU Memory Usage (mean/max/min, %)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Memory Usage (%)')
    axs[2].legend(loc='upper right')

    # 하단 벤치마크 옵션/summary 테이블
    axs[3].axis('off')
    cell_text_bot = opt_rows + sum_rows
    table_bot = axs[3].table(
        cellText=cell_text_bot,
        colLabels=["Benchmark Option", "Value"],
        loc='center',
        cellLoc='left',
        colColours=['#e2ffe2', '#e2ffe2'],
        colWidths=[0.40, 0.60]
    )
    table_bot.auto_set_font_size(False)
    table_bot.set_fontsize(12)
    table_bot.scale(1.2, 1.18)
    # row 높이 50% 증가
    for (row, col), cell in table_bot.get_celld().items():
        if row == 0:
            cell.set_height(0.20)
        else:
            cell.set_height(0.20)
    axs[3].margins(y=0.13)

    plt.tight_layout(pad=2.5)
    plt.savefig('gpu_benchmark_summary.png', bbox_inches='tight')
    plt.close()

def upload_to_s3():
    s3 = boto3.client('s3')
    s3.upload_file('gpu_benchmark_summary.png', S3_BUCKET, S3_KEY_SUMMARY)
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY_SUMMARY}")

if __name__ == "__main__":
    run_benchmark_and_nvidia_smi()
    analyze_and_plot()
    upload_to_s3()
    print("All done!")
