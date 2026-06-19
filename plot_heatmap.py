import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_heatmap(npy_file, output_file=None, cmap='hot', vmin=None, vmax=None, show=True):
    T = np.load(npy_file)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[-1, 1, -1, 1])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (°C)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Temperature Distribution - {os.path.basename(npy_file)}')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"热力图已保存到: {output_file}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='从.npy文件生成温度分布热力图')
    parser.add_argument('input', type=str, help='输入的.npy文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出图片路径 (默认: 不保存)')
    parser.add_argument('--cmap', type=str, default='hot', help='颜色映射 (默认: hot)')
    parser.add_argument('--vmin', type=float, default=None, help='颜色最小值')
    parser.add_argument('--vmax', type=float, default=None, help='颜色最大值')
    parser.add_argument('--no-show', action='store_true', help='不显示图片窗口')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 - {args.input}")
        return

    plot_heatmap(
        args.input,
        output_file=args.output,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()
