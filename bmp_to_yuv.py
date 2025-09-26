import os
import struct
from typing import List, Tuple

def read_bmp(file_path: str) -> Tuple[int, int, List[List[Tuple[int, int, int]]]]:
    """读取24位BMP文件，返回(width, height, 像素二维数组)"""
    with open(file_path, 'rb') as f:
        # 校验BMP头
        if f.read(2) != b'BM':
            raise ValueError('不是有效的BMP文件')
        f.seek(28)
        bpp = struct.unpack('<H', f.read(2))[0]
        if bpp != 24:
            raise ValueError('仅支持24位BMP文件')
        f.seek(18)
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        f.seek(54)
        row_padded = (width * 3 + 3) & ~3
        pixels = [
            [None] * width for _ in range(height)
        ]
        for y in range(height-1, -1, -1):
            row = f.read(row_padded)
            for x in range(width):
                b, g, r = row[x*3:x*3+3]
                pixels[y][x] = (r, g, b)
    return width, height, pixels

def rgb_to_yuv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """RGB转YUV"""
    y = int(0.299 * r + 0.587 * g + 0.114 * b)
    u = int(-0.14713 * r - 0.28886 * g + 0.436 * b + 128)
    v = int(0.615 * r - 0.51499 * g - 0.10001 * b + 128)
    return max(0, min(255, y)), max(0, min(255, u)), max(0, min(255, v))

def save_yuv(pixels: List[List[Tuple[int, int, int]]], output_path: str) -> None:
    """保存YUV文件，逐像素写入YUV"""
    with open(output_path, 'wb') as f:
        for row in pixels:
            for r, g, b in row:
                y, u, v = rgb_to_yuv(r, g, b)
                f.write(struct.pack('BBB', y, u, v))

def process_folder(input_folder: str, output_folder: str) -> None:
    """批量处理文件夹下所有BMP文件"""
    os.makedirs(output_folder, exist_ok=True)
    bmp_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    for filename in bmp_files:
        input_path = os.path.join(input_folder, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_folder, name + '.yuv')
        try:
            width, height, pixels = read_bmp(input_path)
            save_yuv(pixels, output_path)
            print(f"已处理: {filename} ({width}x{height})")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

def main():
    import sys
    if len(sys.argv) != 3:
        print("用法: python bmp_to_yuv.py <输入文件夹> <输出文件夹>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    print(f"处理文件夹: {input_folder}")
    process_folder(input_folder, output_folder)
    print("完成!")

if __name__ == "__main__":
    main()