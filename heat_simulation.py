import numpy as np
import multiprocessing as mp
import os
import argparse
from time import time


TAG_COMPUTE = 'compute'
TAG_RESULT = 'result'
TAG_STOP = 'stop'


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


def worker_process(rank, num_workers, start_row, end_row, nx, ny,
                   T_sub_init, alpha, dt, dx, dy,
                   ctrl_pipe, top_bound_pipe, bottom_bound_pipe):
    T = T_sub_init.copy()
    T_new = np.zeros_like(T)
    local_rows = T.shape[0]

    while True:
        msg = ctrl_pipe.recv()

        if msg[0] == TAG_STOP:
            break

        elif msg[0] == TAG_RESULT:
            ctrl_pipe.send((start_row, end_row, T[1:-1, :].copy()))

        elif msg[0] == TAG_COMPUTE:
            num_steps = msg[1]
            for _ in range(num_steps):
                T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
                    (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2 +
                    (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
                )
                T[1:-1, 1:-1] = T_new[1:-1, 1:-1]

                if rank > 0:
                    top_bound_pipe.send(T[1, :].copy())
                    T[0, :] = top_bound_pipe.recv()

                if rank < num_workers - 1:
                    bottom_bound_pipe.send(T[local_rows - 2, :].copy())
                    T[local_rows - 1, :] = bottom_bound_pipe.recv()

                if rank == 0:
                    T[1, 0] = 0.0
                    T[1, -1] = 0.0
                if rank == num_workers - 1:
                    T[local_rows - 2, 0] = 0.0
                    T[local_rows - 2, -1] = 0.0

            ctrl_pipe.send('done')

    ctrl_pipe.close()
    if top_bound_pipe is not None:
        top_bound_pipe.close()
    if bottom_bound_pipe is not None:
        bottom_bound_pipe.close()


def split_grid(ny, num_workers):
    interior_rows = ny - 2
    chunk_size = interior_rows // num_workers
    chunks = []
    for i in range(num_workers):
        start = 1 + i * chunk_size
        if i == num_workers - 1:
            end = ny - 2
        else:
            end = start + chunk_size - 1
        chunks.append((start, end))
    return chunks


def run_simulation(nx=50, ny=50, alpha=0.01, dt=0.001, total_time=1.0,
                   num_workers=None, output_dir='output', save_interval=10):
    if num_workers is None:
        num_workers = mp.cpu_count()

    max_workers = ny - 2
    if num_workers > max_workers:
        num_workers = max_workers
        print(f"提示: 进程数超过内部行数，已调整为 {num_workers}")

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    stability = alpha * dt / dx ** 2
    if stability > 0.25:
        print(f"警告: 稳定性条件不满足 (r = {stability:.4f} > 0.25)")
        print("建议减小 dt 或增大 dx")

    T_full = initialize_grid(nx, ny)
    num_steps = int(total_time / dt)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'frame_0000.npy'), T_full)
    print(f"保存帧 0 / {num_steps}")

    chunks = split_grid(ny, num_workers)

    bound_pipes = []
    for _ in range(num_workers - 1):
        bound_pipes.append(mp.Pipe())

    ctrl_pipes = []
    processes = []
    for rank in range(num_workers):
        start_row, end_row = chunks[rank]
        T_sub = T_full[start_row - 1:end_row + 2, :].copy()

        main_ctrl, worker_ctrl = mp.Pipe()

        top_bound_worker = None
        bottom_bound_worker = None
        if rank > 0:
            top_bound_worker = bound_pipes[rank - 1][1]
        if rank < num_workers - 1:
            bottom_bound_worker = bound_pipes[rank][0]

        proc = mp.Process(
            target=worker_process,
            args=(rank, num_workers, start_row, end_row, nx, ny,
                  T_sub, alpha, dt, dx, dy,
                  worker_ctrl, top_bound_worker, bottom_bound_worker)
        )
        processes.append(proc)
        ctrl_pipes.append(main_ctrl)
        proc.start()

    print(f"网格大小: {nx}x{ny}")
    print(f"时间步数: {num_steps}")
    print(f"进程数: {num_workers}")
    for rank, (s, e) in enumerate(chunks):
        print(f"  Worker {rank}: 行 {s}-{e} ({e - s + 1} 行)")
    print(f"稳定性参数 r = {stability:.6f}")
    print("-" * 50)

    start_time = time()

    steps_done = 0
    while steps_done < num_steps:
        batch_size = min(save_interval, num_steps - steps_done)

        for rank in range(num_workers):
            ctrl_pipes[rank].send((TAG_COMPUTE, batch_size))
        for rank in range(num_workers):
            assert ctrl_pipes[rank].recv() == 'done'

        steps_done += batch_size

        for rank in range(num_workers):
            ctrl_pipes[rank].send((TAG_RESULT,))

        T_full[0, :] = 0.0
        T_full[-1, :] = 0.0
        T_full[:, 0] = 0.0
        T_full[:, -1] = 0.0

        for rank in range(num_workers):
            start_row, end_row, T_chunk = ctrl_pipes[rank].recv()
            T_full[start_row:end_row + 1, :] = T_chunk

        if steps_done % save_interval == 0 or steps_done == num_steps:
            np.save(os.path.join(output_dir, f'frame_{steps_done:04d}.npy'), T_full)
            print(f"保存帧 {steps_done} / {num_steps}")

    for rank in range(num_workers):
        ctrl_pipes[rank].send((TAG_STOP,))

    for proc in processes:
        proc.join()

    elapsed = time() - start_time
    print("-" * 50)
    print(f"模拟完成! 总耗时: {elapsed:.2f} 秒")
    print(f"输出目录: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description='二维金属板热传导模拟（多进程有限差分法，支持边界同步）')
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
