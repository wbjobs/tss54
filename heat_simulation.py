import numpy as np
import multiprocessing as mp
import os
import argparse
from time import time


def initialize_grid(nx, ny):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    T = 100.0 * np.exp(-r ** 2 / 0.1)
    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0
    return T


def compute_chunk(args):
    T_full, start_row, end_row, alpha, dt, dx, dy, steps = args
    nx = T_full.shape[1]

    T = T_full[start_row - 1:end_row + 2, :].copy()
    T_new = np.zeros_like(T)

    for _ in range(steps):
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2 +
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
        )
        T[1:-1, 1:-1] = T_new[1:-1, 1:-1]

    return start_row, end_row, T[1:-1, :]


def run_simulation(nx=50, ny=50, alpha=0.01, dt=0.001, total_time=1.0,
                   num_workers=None, output_dir='output', save_interval=10):
    if num_workers is None:
        num_workers = mp.cpu_count()

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    stability = alpha * dt / dx ** 2
    if stability > 0.25:
        print(f"警告: 稳定性条件不满足 (r = {stability:.4f} > 0.25)")
        print("建议减小 dt 或增大 dx")

    T = initialize_grid(nx, ny)

    num_steps = int(total_time / dt)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f'frame_0000.npy'), T)
    print(f"保存帧 0 / {num_steps}")

    chunk_size = (ny - 2) // num_workers
    chunks = []
    for i in range(num_workers):
        start_row = 1 + i * chunk_size
        if i == num_workers - 1:
            end_row = ny - 2
        else:
            end_row = start_row + chunk_size - 1
        chunks.append((start_row, end_row))

    print(f"网格大小: {nx}x{ny}")
    print(f"时间步数: {num_steps}")
    print(f"进程数: {num_workers}")
    print(f"分块大小: 每个进程约 {chunk_size} 行")
    print(f"稳定性参数 r = {stability:.6f}")
    print("-" * 50)

    start_time = time()

    pool = mp.Pool(processes=num_workers)

    for frame_idx in range(1, num_steps + 1):
        args_list = []
        for start_row, end_row in chunks:
            args_list.append((T, start_row, end_row, alpha, dt, dx, dy, 1))

        results = pool.map(compute_chunk, args_list)

        for start_row, end_row, T_chunk in results:
            T[start_row:end_row + 1, :] = T_chunk

        T[0, :] = 0.0
        T[-1, :] = 0.0
        T[:, 0] = 0.0
        T[:, -1] = 0.0

        if frame_idx % save_interval == 0:
            np.save(os.path.join(output_dir, f'frame_{frame_idx:04d}.npy'), T)
            print(f"保存帧 {frame_idx} / {num_steps}")

    pool.close()
    pool.join()

    elapsed = time() - start_time
    print("-" * 50)
    print(f"模拟完成! 总耗时: {elapsed:.2f} 秒")
    print(f"输出目录: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description='二维金属板热传导模拟（多进程有限差分法）')
    parser.add_argument('--nx', type=int, default=50, help='x方向网格数 (默认: 50)')
    parser.add_argument('--ny', type=int, default=50, help='y方向网格数 (默认: 50)')
    parser.add_argument('--alpha', type=float, default=0.01, help='热扩散系数 (默认: 0.01)')
    parser.add_argument('--dt', type=float, default=0.001, help='时间步长 (默认: 0.001)')
    parser.add_argument('--time', type=float, default=0.5, help='总模拟时间 (默认: 0.5)')
    parser.add_argument('--workers', type=int, default=None, help='进程数 (默认: CPU核心数)')
    parser.add_argument('--output', type=str, default='output', help='输出目录 (默认: output)')
    parser.add_argument('--interval', type=int, default=10, help='保存帧间隔 (默认: 10)')

    args = parser.parse_args()

    run_simulation(
        nx=args.nx,
        ny=args.ny,
        alpha=args.alpha,
        dt=args.dt,
        total_time=args.time,
        num_workers=args.workers,
        output_dir=args.output,
        save_interval=args.interval
    )


if __name__ == '__main__':
    main()
