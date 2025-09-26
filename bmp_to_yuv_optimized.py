import os
import struct
import argparse
import sys
from typing import Tuple, List, BinaryIO, Iterator
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# ITU-R BT.601 标准系数, 预先存为常量避免重复创建矩阵
YUV_WEIGHTS = np.array([
    [0.299, 0.587, 0.114],
    [-0.14713, -0.28886, 0.436],
    [0.615, -0.51499, -0.10001],
], dtype=np.float32)


# 运行参数封装, 方便在函数之间传递设置
@dataclass
class Config:
    input_folder: str
    output_folder: str
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


def read_bmp_stream(file_path: str) -> Iterator[np.ndarray]:
    """流式读取BMP像素数据"""
    with open(file_path, 'rb') as f:
        validate_bmp_header(f)

        f.seek(28)
        bpp = struct.unpack('<H', f.read(2))[0]
        if bpp != 24:
            raise BMPError('仅支持24位BMP文件')

        f.seek(18)
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]

        # 每个像素 3 字节(RGB), BMP 行末需要 4 字节对齐
        row_stride = width * 3
        row_padded = (row_stride + 3) & ~3

        # 像素数据从第 54 字节起, 顺序为自底向上存储
        f.seek(54)

        for _ in range(height):
            row_data = f.read(row_padded)
            if len(row_data) < row_padded:
                raise BMPError('像素数据不完整')
            # 丢弃对齐用的填充字节, 并从 BGR 转成 RGB 顺序
            rgb_row = np.frombuffer(row_data[:row_stride], dtype=np.uint8).reshape(width, 3)[:, ::-1]
            yield rgb_row


def rgb_to_yuv_batch(rgb_array: np.ndarray) -> np.ndarray:
    """批量RGB转YUV,使用numpy矩阵运算加速"""
    # 使用矩阵乘法一次性计算 YUV 三个通道
    yuv = rgb_array.astype(np.float32) @ YUV_WEIGHTS.T
    # Y 通道保持 0-255, UV 需平移到无符号范围
    yuv[:, 0] = np.clip(yuv[:, 0], 0, 255)
    yuv[:, 1] = np.clip(yuv[:, 1] + 128, 0, 255)
    yuv[:, 2] = np.clip(yuv[:, 2] + 128, 0, 255)
    return yuv.astype(np.uint8)


def process_bmp_optimized(input_path: str, output_path: str, config: Config) -> None:
    """优化的BMP处理:流式读取 + 批量计算"""
    width, height = get_bmp_dimensions(input_path)

    if config.verbose:
        print(f"处理图像: {os.path.basename(input_path)} ({width}x{height})")

    with open(output_path, 'wb') as f_out:
        pixel_buffer: List[np.ndarray] = []
        buffered_pixels = 0

        def flush_buffer() -> None:
            """把缓冲中的多行像素合并后批量写入"""
            nonlocal buffered_pixels
            if not pixel_buffer:
                return
            rgb_array = np.concatenate(pixel_buffer, axis=0)
            f_out.write(rgb_to_yuv_batch(rgb_array).tobytes())
            pixel_buffer.clear()
            buffered_pixels = 0

        for row_pixels in read_bmp_stream(input_path):
            # 按行追加像素, 保持内存占用可控
            pixel_buffer.append(row_pixels)
            buffered_pixels += row_pixels.shape[0]
            if buffered_pixels >= config.batch_size:
                flush_buffer()

        # 文件读完后可能仍有未写入的数据
        flush_buffer()

    if config.verbose:
        print(f"完成: {width * height} 像素处理")


def process_folder_optimized(input_folder: str, output_folder: str, config: Config) -> None:
    """优化的文件夹批量处理"""
    os.makedirs(output_folder, exist_ok=True)

    # 查找所有BMP文件
    bmp_files = [f for f in os.listdir(input_folder)
                if f.lower().endswith('.bmp')]
    # 按需处理全部 BMP, 保留原顺序

    if not bmp_files:
        print(f"在 {input_folder} 中未找到BMP文件")
        return

    print(f"发现 {len(bmp_files)} 个BMP文件")

    # 使用进度条
    with tqdm(total=len(bmp_files), desc="处理进度", unit="文件") as pbar:
        # 逐个文件转换, 失败时打印提示并继续
        for filename in bmp_files:
            input_path = os.path.join(input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, name + '.yuv')
            pbar.set_postfix({"当前": filename})

            try:
                process_bmp_optimized(input_path, output_path, config)
            except BMPError as err:
                print(f"❌ 文件格式错误 {filename}: {err}")
            except Exception as err:
                print(f"❌ 处理 {filename} 时出错: {err}")
            finally:
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
        # 预先检查输入目录有效性, 早退出给出明确提示
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