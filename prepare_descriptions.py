import os
import csv
import base64
import time
from openai import OpenAI

AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    ("AGIQA-1K",    "images", "AGIQA-1K_descriptions.csv"),
    ("AGIQA-3K",    "images", "AGIQA-3K_descriptions.csv"),
    ("AIGCIQA2023", "images", "AIGCIQA2023_descriptions.csv"),
]

OUTPUT_DIR = os.path.join(ROOT_DIR, "pre_descriptions")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

API_DELAY = 0.5

RETRY_TIMES = 5

client = OpenAI(
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
)


def read_existing_results(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    processed = set()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["image_name"])
    return processed


def append_result(csv_path: str, image_name: str, description: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_name", "description"])
        writer.writerow([image_name, description])


def scan_images(dataset_dir: str, images_subdir: str) -> list:
    images_path = os.path.join(dataset_dir, images_subdir)
    if not os.path.isdir(images_path):
        print(f"[警告] 图像目录不存在: {images_path}")
        return []

    all_files = []
    for fname in os.listdir(images_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            all_files.append(os.path.join(images_path, fname))

    all_files.sort()
    return all_files


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_description(image_path: str) -> str:
    base64_str = encode_image_base64(image_path)

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime_ext = "jpeg" if ext == "jpg" else ext

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
                time.sleep(API_DELAY * 2)
            else:
                print(f"  [API错误(已达最大重试)] {os.path.basename(image_path)}: {last_error}")

    return "Error"


def process_dataset(dataset_name: str, images_subdir: str, output_csv: str):
    dataset_dir = os.path.join(ROOT_DIR, "data", dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_csv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = scan_images(dataset_dir, images_subdir)
    total = len(image_files)
    if total == 0:
        print(f"[{dataset_name}] 未找到任何图像文件，跳过。")
        return

    print(f"[{dataset_name}] 共发现 {total} 张图像")

    processed = read_existing_results(output_path)
    if processed:
        print(f"[{dataset_name}] 已处理 {len(processed)} 张，将继续处理剩余图片")

    for idx, img_path in enumerate(image_files, start=1):
        img_name = os.path.basename(img_path)

        if img_name in processed:
            print(f"  [{idx}/{total}] 跳过(已处理): {img_name}")
            continue

        print(f"  [{idx}/{total}] 处理中: {img_name} ...")
        description = generate_description(img_path)
        append_result(output_path, img_name, description)
        print(f"    -> 描述: {description[:60]}...")

        time.sleep(API_DELAY)

    print(f"[{dataset_name}] 处理完成！结果保存至: {output_path}")


def main():
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