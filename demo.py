import os
import shutil
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from tqdm import tqdm

# 你的图片目录 (请根据实际挂载情况核对路径)
IMAGE_DIR = r"//scrinvme/huilin/bdd/open_source_data/cubit-det/images"
# 隔离坏图的存放目录 (会自动创建)
CORRUPTED_DIR = r"//scrinvme/huilin/bdd/open_source_data/cubit-det/corrupted_images"


def verify_and_move_image(img_name):
    """
    模拟 LLaVA 的读取方式，强制解码像素。如果损坏，返回文件名和错误信息。
    """
    img_path = os.path.join(IMAGE_DIR, img_name)

    # 忽略非图片文件
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return None

    try:
        # 必须使用 convert('RGB') 或 load() 才能触发 truncated 报错
        with Image.open(img_path) as img:
            img.convert("RGB")
        return None  # 正常图片，返回 None

    except Exception as e:
        return (img_name, str(e))


if __name__ == "__main__":
    print(f"开始深度扫描目录: {IMAGE_DIR}")

    if not os.path.exists(CORRUPTED_DIR):
        os.makedirs(CORRUPTED_DIR)

    img_list = os.listdir(IMAGE_DIR)
    bad_images = []

    # 使用多进程加速扫描 (利用你的 NVMe 固态优势)
    # max_workers 可以设置为你的 CPU 核心数，比如 8 或 16
    with ProcessPoolExecutor(max_workers=8) as executor:
        # 使用 tqdm 显示进度条
        results = list(
            tqdm(
                executor.map(verify_and_move_image, img_list),
                total=len(img_list),
                desc="Checking Images",
            )
        )

    # 过滤出有问题的图片
    for res in results:
        if res is not None:
            img_name, error_msg = res
            bad_images.append(img_name)

            # 将坏图移动到隔离区
            src_path = os.path.join(IMAGE_DIR, img_name)
            dst_path = os.path.join(CORRUPTED_DIR, img_name)
            shutil.move(src_path, dst_path)
            print(f"\n[已隔离] {img_name} - 错误: {error_msg}")

    print("\n" + "=" * 50)
    print(f"扫描完成！共检查了 {len(img_list)} 个文件。")
    if len(bad_images) > 0:
        print(f"共发现并隔离了 {len(bad_images)} 张损坏图片。")
        print(f"它们已被移动至: {CORRUPTED_DIR}")
    else:
        print("完美！没有发现任何损坏或截断的图片。")
