import numpy as np
import argparse
import os


def compute_jumps(T, chunk_rows):
    print(f"  温度范围: [{T.min():.4f}, {T.max():.4f}]")
    print(f"  接缝处检查 (相邻行的最大温度差):")

    max_jump = 0.0
    max_jump_row = -1
    for row in chunk_rows:
        jump = np.max(np.abs(T[row, :] - T[row + 1, :]))
        mean_jump = np.mean(np.abs(T[row, :] - T[row + 1, :]))
        print(f"    行 {row} <-> {row + 1}: 最大差 = {jump:.2e}, 平均差 = {mean_jump:.2e}")
        if jump > max_jump:
            max_jump = jump
            max_jump_row = row

    interior_max_jump = 0.0
    ny = T.shape[0]
    for row in range(1, ny - 2):
        if row in chunk_rows:
            continue
        jump = np.max(np.abs(T[row, :] - T[row + 1, :]))
        if jump > interior_max_jump:
            interior_max_jump = jump

    print(f"  非接缝内部最大行差: {interior_max_jump:.2e}")
    if max_jump > interior_max_jump * 1.5 and max_jump > 1e-10:
        print(f"  ⚠️  接缝处存在异常突变! (行{max_jump_row})")
    else:
        print(f"  ✅ 接缝正常，与内部行差一致")
    return max_jump, interior_max_jump


def run_single_process(nx, ny, alpha, dt, total_time):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    T = 100.0 * np.exp(-r ** 2 / 0.1)
    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0

    T_new = np.zeros_like(T)
    num_steps = int(total_time / dt)

    for _ in range(num_steps):
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2 +
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
        )
        T[1:-1, 1:-1] = T_new[1:-1, 1:-1]

    return T


def main():
    parser = argparse.ArgumentParser(description='验证多进程接缝正确性')
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time', type=float, default=0.2)
    parser.add_argument('--frame', type=str, default='output/frame_0200.npy')

    args = parser.parse_args()

    interior = args.ny - 2
    chunk = interior // args.workers
    chunk_rows = []
    for i in range(args.workers - 1):
        end_row = (i + 1) * chunk
        chunk_rows.append(end_row)

    print(f"网格 {args.nx}x{args.ny}, {args.workers} 个进程")
    print(f"接缝位于行: {chunk_rows}")
    print()

    if os.path.exists(args.frame):
        T_mp = np.load(args.frame)
        print("【多进程结果】")
        compute_jumps(T_mp, chunk_rows)
        print()

        print("【单进程参考解 (正在计算...)】")
        T_sp = run_single_process(args.nx, args.ny, 0.01, 0.001, args.time)
        compute_jumps(T_sp, chunk_rows)
        print()

        diff = np.abs(T_mp - T_sp)
        print(f"【多进程 vs 单进程 绝对误差】")
        print(f"  最大误差: {diff.max():.2e}")
        print(f"  平均误差: {diff.mean():.2e}")

        print()
        print("接缝处最大误差:")
        for row in chunk_rows:
            local_max = max(diff[row, :].max(), diff[row + 1, :].max())
            print(f"  行 {row}/{row + 1}: {local_max:.2e}")

        non_chunk_error = 0.0
        for row in range(1, args.ny - 1):
            if row not in chunk_rows and (row - 1) not in chunk_rows:
                non_chunk_error = max(non_chunk_error, diff[row, :].max())
        print(f"非接缝区域最大误差: {non_chunk_error:.2e}")

        if diff.max() < 1e-8:
            print("\n✅ 多进程与单进程结果完全一致! 接缝正确。")
        else:
            print(f"\n⚠️  存在误差，需检查。")
    else:
        print(f"找不到文件: {args.frame}")


if __name__ == '__main__':
    main()
