import os
import struct
import argparse
import sys
from typing import Tuple, List, Optional, BinaryIO, Iterator
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class Config:
    input_folder: str
    output_folder: str
    bit_depth: int = 24
    batch_size: int = 1024
    verbose: bool = False

class BMPError(Exception):
    """BMP文件处理异常"""
    pass


def validate_bmp_header(file: BinaryIO) -> None:
    """验证BMP文件头"""
    file.seek(0)
    signature = file.read(2)
    if signature != b'BM':
        raise BMPError('不是有效的BMP文件')


def get_bmp_dimensions(file_path: str) -> Tuple[int, int]:
    """获取BMP图像尺寸,不读取像素数据"""
    with open(file_path, 'rb') as f:
        validate_bmp_header(f)
        f.seek(18)  # 宽度和高度位置
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        return width, height


def read_bmp_stream(file_path: str) -> Iterator[List[Tuple[int, int, int]]]:
    """流式读取BMP像素数据"""
    with open(file_path, 'rb') as f:
        validate_bmp_header(f)

        # 跳到位深检查
        f.seek(28)
        bpp = struct.unpack('<H', f.read(2))[0]
        if bpp != 24:
            raise BMPError('仅支持24位BMP文件')

        # 读取尺寸
        f.seek(18)
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]

        # 计算每行字节数（4字节对齐）
        row_padded = (width * 3 + 3) & ~3
        pixel_count = width * 3

        # 跳到像素数据起始位置
        f.seek(54)

        # 从底部开始逐行读取（BMP格式特点）
        for y in range(height-1, -1, -1):
            row_data = f.read(row_padded)
            row_pixels = []
            for x in range(width):
                start_pos = x * 3
                b, g, r = row_data[start_pos:start_pos+3]
                row_pixels.append((r, g, b))
            yield row_pixels


def rgb_to_yuv_batch(rgb_array: np.ndarray) -> np.ndarray:
    """批量RGB转YUV,使用numpy矩阵运算加速"""
    # YUV转换矩阵       
    weights = np.array([
        [0.299, 0.587, 0.114],      # Y分量
        [-0.14713, -0.28886, 0.436], # U分量
        [0.615, -0.51499, -0.10001]  # V分量
    ])

    # 矩阵运算
    yuv = np.dot(rgb_array, weights.T)

    # 边界约束和偏移
    yuv[:, 0] = np.clip(yuv[:, 0], 0, 255)  # Y值
    yuv[:, 1] = np.clip(yuv[:, 1] + 128, 0, 255)  # U值 + 128
    yuv[:, 2] = np.clip(yuv[:, 2] + 128, 0, 255)  # V值 + 128

    return yuv.astype(np.uint8)


def process_bmp_optimized(input_path: str, output_path: str, config: Config) -> None:
    """优化的BMP处理:流式读取 + 批量计算"""
    width, height = get_bmp_dimensions(input_path)
    total_pixels = width * height

    if config.verbose:
        print(f"处理图像: {os.path.basename(input_path)} ({width}x{height})")

    # 预分配YUV输出文件
    with open(output_path, 'wb') as f_out:
        # 批量处理缓冲区
        rgb_buffer = []
        yuv_buffer = []

        for row_idx, row_pixels in enumerate(read_bmp_stream(input_path)):
            rgb_buffer.extend(row_pixels)

            # 当缓冲区达到批次大小时进行处理
            if len(rgb_buffer) >= config.batch_size:
                # 转换为numpy数组
                rgb_array = np.array(rgb_buffer, dtype=np.uint8)

                # 批量转换
                yuv_batch = rgb_to_yuv_batch(rgb_array)

                # 写入YUV数据
                f_out.write(yuv_batch.tobytes())

                # 清空缓冲区
                rgb_buffer = []

        # 处理剩余数据
        if rgb_buffer:
            rgb_array = np.array(rgb_buffer, dtype=np.uint8)
            yuv_batch = rgb_to_yuv_batch(rgb_array)
            f_out.write(yuv_batch.tobytes())

    if config.verbose:
        print(f"完成: {total_pixels} 像素处理")


def process_folder_optimized(input_folder: str, output_folder: str, config: Config) -> None:
    """优化的文件夹批量处理"""
    os.makedirs(output_folder, exist_ok=True)

    # 查找所有BMP文件
    bmp_files = [f for f in os.listdir(input_folder)
                if f.lower().endswith('.bmp')]

    if not bmp_files:
        print(f"在 {input_folder} 中未找到BMP文件")
        return

    print(f"发现 {len(bmp_files)} 个BMP文件")

    # 使用进度条
    with tqdm(total=len(bmp_files), desc="处理进度", unit="文件") as pbar:
        for filename in bmp_files:
            input_path = os.path.join(input_folder, filename)

            try:
                # 输出文件名
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_folder, name + '.yuv')

                # 处理单个文件
                process_bmp_optimized(input_path, output_path, config)

                pbar.update(1)
                pbar.set_postfix({"当前": os.path.basename(filename)})

            except BMPError as e:
                print(f"❌ 文件格式错误 {filename}: {e}")
                pbar.update(1)
            except FileNotFoundError:
                print(f"❌ 文件不存在 {filename}")
                pbar.update(1)
            except PermissionError:
                print(f"❌ 权限不足 {filename}")
                pbar.update(1)
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {e}")
                pbar.update(1)

    print(f"✅ 所有文件处理完成！输出到: {output_folder}")


def parse_args() -> Config:
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description='🐱 优化的BMP转YUV工具 - 内存友好版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python bmp_to_yuv_optimized.py input output
  python bmp_to_yuv_optimized.py input output --batch-size 2048
  python bmp_to_yuv_optimized.py input output --verbose
        '''
    )

    parser.add_argument('input_folder', help='输入文件夹路径')
    parser.add_argument('output_folder', help='输出文件夹路径')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='批处理大小 (默认: 1024)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')

    args = parser.parse_args()

    return Config(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        verbose=args.verbose
    )


def main():
    """主函数"""
    try:
        config = parse_args()

        # 检查输入文件夹
        if not os.path.exists(config.input_folder):
            print(f"❌ 输入文件夹不存在: {config.input_folder}")
            sys.exit(1)

        if not os.path.isdir(config.input_folder):
            print(f"❌ 输入路径不是文件夹: {config.input_folder}")
            sys.exit(1)

        print(f"🐱 浮浮酱的BMP转YUV优化工具启动！")
        print(f"输入文件夹: {config.input_folder}")
        print(f"输出文件夹: {config.output_folder}")
        print(f"批处理大小: {config.batch_size}")
        print("-" * 50)

        process_folder_optimized(config.input_folder, config.output_folder, config)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"💥 程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()