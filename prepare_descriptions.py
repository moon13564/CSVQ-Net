"""
prepare_descriptions.py
=======================
扫描 data/ 目录下三个数据集(AGIQA-1K, AGIQA-3K, AIGCIQA2023)的所有图像，
调用 Agnes AI (agnes-2.0-flash) 模型为每张图像生成描述性提示词，
并将结果(图像名称 + 描述)以 CSV 表格形式保存到 pre_descriptions/ 文件夹。

用法:
    set AGNES_API_KEY=your_api_key_here
    python prepare_descriptions.py

依赖:
    pip install openai
"""

import os
import csv
import base64
import time
from openai import OpenAI

# ============================================================
# 配置区
# ============================================================

# Agnes AI API 配置
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"

# 根目录（脚本所在目录，即项目根目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集定义: (数据集目录名, 图像子文件夹, 输出CSV文件名)
DATASETS = [
    ("AGIQA-1K",    "images", "AGIQA-1K_descriptions.csv"),
    ("AGIQA-3K",    "images", "AGIQA-3K_descriptions.csv"),
    ("AIGCIQA2023", "images", "AIGCIQA2023_descriptions.csv"),
]

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "pre_descriptions")

# 支持的图像格式（扩展名，小写）
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# API 调用间隔（秒），避免请求过快
API_DELAY = 0.5

# API 调用失败时的最大重试次数
RETRY_TIMES = 5

# ============================================================
# 初始化 OpenAI 风格客户端
# ============================================================

# 如果没有设置 API Key，后续调用会失败
client = OpenAI(
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
)


def read_existing_results(csv_path: str) -> set:
    """
    读取已存在的输出 CSV 文件，返回已处理完成的图像名称集合。
    用于断点续传，避免重复调用 API。
    """
    if not os.path.exists(csv_path):
        return set()
    processed = set()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["image_name"])
    return processed


def append_result(csv_path: str, image_name: str, description: str):
    """
    将单条结果追加写入 CSV 文件。
    若文件不存在，先写入表头。
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_name", "description"])
        writer.writerow([image_name, description])


def scan_images(dataset_dir: str, images_subdir: str) -> list:
    """
    扫描指定数据集的图像文件夹，返回所有支持的图像文件路径列表。
    """
    images_path = os.path.join(dataset_dir, images_subdir)
    if not os.path.isdir(images_path):
        print(f"[警告] 图像目录不存在: {images_path}")
        return []

    all_files = []
    for fname in os.listdir(images_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            all_files.append(os.path.join(images_path, fname))

    # 按文件名排序，保证处理顺序稳定
    all_files.sort()
    return all_files


def encode_image_base64(image_path: str) -> str:
    """
    读取图像文件并返回 Base64 编码的字符串。
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_description(image_path: str) -> str:
    """
    调用 Agnes AI 模型，为指定图像生成描述性提示词。
    失败时自动重试，最多 RETRY_TIMES 次。

    系统提示词要求模型扮演专业图像描述生成器，
    输出约50词的描述，包含详细内容、风格、拍摄角度、色彩氛围和质量。
    用户提示词要求直接以描述开头，不要"该图像"等冗余词汇。
    """
    # 对图像进行 Base64 编码
    base64_str = encode_image_base64(image_path)

    # 获取图像扩展名（不含点），用于构造 Data URL
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    # 修正 jpeg -> jpeg（标准 MIME 类型）
    mime_ext = "jpeg" if ext == "jpg" else ext

    # 构造请求消息体（每次重试复用，避免重复编码）
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional image description generator. "
                "Please generate a description of approximately 50 words. "
                "Including its detailed content, style(anime style, sci-fi style, "
                "realistic style, abstract style, baroque style and so on), "
                "shooting angle as well as color atmosphere and quality. "
                "Put the content at the top, be a little more detailed."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please describe this picture in 50 words. "
                        "Your answer should not have superfluous words "
                        'such as "The image", but rather start directly '
                        "with your description."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime_ext};base64,{base64_str}",
                    },
                },
            ],
        },
    ]

    last_error = ""
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            completion = client.chat.completions.create(
                model=AGNES_MODEL,
                messages=messages,
                max_tokens=300,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)
            if attempt < RETRY_TIMES:
                print(f"  [重试 {attempt}/{RETRY_TIMES}] {os.path.basename(image_path)}: {last_error}")
                time.sleep(API_DELAY * 2)  # 重试前稍长等待
            else:
                print(f"  [API错误(已达最大重试)] {os.path.basename(image_path)}: {last_error}")

    return "Error"


def process_dataset(dataset_name: str, images_subdir: str, output_csv: str):
    """
    处理单个数据集：扫描图像、调用 API、保存结果。
    """
    dataset_dir = os.path.join(ROOT_DIR, "data", dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_csv)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 扫描所有图像文件
    image_files = scan_images(dataset_dir, images_subdir)
    total = len(image_files)
    if total == 0:
        print(f"[{dataset_name}] 未找到任何图像文件，跳过。")
        return

    print(f"[{dataset_name}] 共发现 {total} 张图像")

    # 读取已处理的结果（断点续传）
    processed = read_existing_results(output_path)
    if processed:
        print(f"[{dataset_name}] 已处理 {len(processed)} 张，将继续处理剩余图片")

    # 逐张处理
    for idx, img_path in enumerate(image_files, start=1):
        img_name = os.path.basename(img_path)

        # 跳过已处理的图像
        if img_name in processed:
            print(f"  [{idx}/{total}] 跳过(已处理): {img_name}")
            continue

        print(f"  [{idx}/{total}] 处理中: {img_name} ...")
        description = generate_description(img_path)
        append_result(output_path, img_name, description)
        print(f"    -> 描述: {description[:60]}...")

        # 避免请求过快触发限流
        time.sleep(API_DELAY)

    print(f"[{dataset_name}] 处理完成！结果保存至: {output_path}")


def main():
    """
    主函数：遍历所有数据集并依次处理。
    """
    # 检查 API Key 是否配置
    if not AGNES_API_KEY:
        print("=" * 60)
        print("错误: 未设置 AGNES_API_KEY 环境变量！")
        print("请在运行前设置您的 Agnes AI API Key:")
        print("  Windows: set AGNES_API_KEY=your_key_here")
        print("  Linux:   export AGNES_API_KEY=your_key_here")
        print("=" * 60)
        return

    print(f"Agnes AI API URL: {AGNES_BASE_URL}")
    print(f"模型: {AGNES_MODEL}")
    print(f"数据集数量: {len(DATASETS)}")
    print(f"结果输出目录: {OUTPUT_DIR}")
    print("-" * 60)

    for dataset_name, images_subdir, output_csv in DATASETS:
        process_dataset(dataset_name, images_subdir, output_csv)
        print("-" * 60)

    print("所有数据集处理完成！")


if __name__ == "__main__":
    main()
