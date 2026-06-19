import numpy as np
import multiprocessing as mp
import os
import argparse
from time import time


TAG_COMPUTE = 'compute'
TAG_RESULT = 'result'
TAG_STOP = 'stop'


def get_heat_source_position(path_type, path_params, t):
    if path_type == 'line':
        x0, y0 = path_params['start']
        x1, y1 = path_params['end']
        total_time = path_params['total_time']
        s = min(t / total_time, 1.0) if total_time > 0 else 0.0
        x = x0 + (x1 - x0) * s
        y = y0 + (y1 - y0) * s
        return x, y

    elif path_type == 'circle':
        cx, cy = path_params['center']
        radius = path_params['radius']
        omega = path_params['angular_speed']
        x = cx + radius * np.cos(omega * t)
        y = cy + radius * np.sin(omega * t)
        return x, y

    elif path_type == 'sine':
        x0, x1 = path_params['x_range']
        amplitude = path_params['amplitude']
        frequency = path_params['frequency']
        total_time = path_params['total_time']
        s = min(t / total_time, 1.0) if total_time > 0 else 0.0
        x = x0 + (x1 - x0) * s
        y = amplitude * np.sin(2 * np.pi * frequency * s)
        return x, y

    elif path_type == 'stationary':
        x, y = path_params['position']
        return x, y

    else:
        raise ValueError(f"未知路径类型: {path_type}")


def add_heat_source(T, x_pos, y_pos, power, radius, dx, dy):
    ny, nx = T.shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    r2 = (X - x_pos) ** 2 + (Y - y_pos) ** 2
    sigma = radius
    q = (power / (2 * np.pi * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))

    T += q * dx * dy


def initialize_grid(nx, ny, init_type='zero'):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    if init_type == 'gaussian':
        r = np.sqrt(X ** 2 + Y ** 2)
        T = 100.0 * np.exp(-r ** 2 / 0.1)
    else:
        T = np.zeros((ny, nx))

    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0
    return T


def worker_process(rank, num_workers, start_row, end_row, nx, ny,
                   T_sub_init, alpha, dt, dx, dy,
                   heat_source_config,
                   ctrl_pipe, top_bound_pipe, bottom_bound_pipe):
    T = T_sub_init.copy()
    T_new = np.zeros_like(T)
    local_rows = T.shape[0]

    y_full = np.linspace(-1, 1, ny)
    y_sub = y_full[start_row - 1:end_row + 2]
    x_full = np.linspace(-1, 1, nx)

    Y_sub, X_sub = np.meshgrid(y_sub, x_full, indexing='ij')

    path_type = heat_source_config['path_type']
    path_params = heat_source_config['path_params']
    power = heat_source_config['power']
    radius = heat_source_config['radius']
    power_scale = heat_source_config.get('power_scale', 1.0)

    effective_power = power * power_scale * dx * dy / (2 * np.pi * radius ** 2)

    step = 0

    while True:
        msg = ctrl_pipe.recv()

        if msg[0] == TAG_STOP:
            break

        elif msg[0] == TAG_RESULT:
            ctrl_pipe.send((start_row, end_row, T[1:-1, :].copy()))

        elif msg[0] == TAG_COMPUTE:
            num_steps = msg[1]
            start_step = msg[2] if len(msg) > 2 else step

            for s in range(num_steps):
                current_step = start_step + s
                t = current_step * dt

                x_pos, y_pos = get_heat_source_position(path_type, path_params, t)

                r2 = (X_sub - x_pos) ** 2 + (Y_sub - y_pos) ** 2
                source = effective_power * np.exp(-r2 / (2 * radius ** 2))

                T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
                    (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2 +
                    (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
                )
                T_new[1:-1, 1:-1] += source[1:-1, 1:-1] * dt
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

            step = start_step + num_steps
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


def parse_path_args(args):
    path_type = args.path_type
    path_params = {}

    if path_type == 'line':
        start = tuple(map(float, args.start.split(',')))
        end = tuple(map(float, args.end.split(',')))
        path_params['start'] = start
        path_params['end'] = end
        path_params['total_time'] = args.time

    elif path_type == 'circle':
        center = tuple(map(float, args.center.split(',')))
        path_params['center'] = center
        path_params['radius'] = args.radius_path
        path_params['angular_speed'] = args.angular_speed

    elif path_type == 'sine':
        x_range = tuple(map(float, args.x_range.split(',')))
        path_params['x_range'] = x_range
        path_params['amplitude'] = args.amplitude
        path_params['frequency'] = args.frequency
        path_params['total_time'] = args.time

    elif path_type == 'stationary':
        position = tuple(map(float, args.position.split(',')))
        path_params['position'] = position

    return path_type, path_params


def run_simulation(nx=50, ny=50, alpha=0.01, dt=0.001, total_time=1.0,
                   num_workers=None, output_dir='output', save_interval=10,
                   init_type='zero',
                   heat_source_config=None):
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

    T_full = initialize_grid(nx, ny, init_type=init_type)
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
                  heat_source_config,
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
    if heat_source_config:
        print(f"热源类型: {heat_source_config['path_type']}")
        print(f"热源功率: {heat_source_config['power']}")
        print(f"热源半径: {heat_source_config['radius']}")
    print("-" * 50)

    start_time = time()

    steps_done = 0
    while steps_done < num_steps:
        batch_size = min(save_interval, num_steps - steps_done)

        for rank in range(num_workers):
            ctrl_pipes[rank].send((TAG_COMPUTE, batch_size, steps_done))
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
    parser = argparse.ArgumentParser(
        description='二维金属板热传导模拟（多进程有限差分法，支持移动热源）'
    )
    parser.add_argument('--nx', type=int, default=100, help='x方向网格数 (默认: 100)')
    parser.add_argument('--ny', type=int, default=100, help='y方向网格数 (默认: 100)')
    parser.add_argument('--alpha', type=float, default=0.005, help='热扩散系数 (默认: 0.005)')
    parser.add_argument('--dt', type=float, default=0.0005, help='时间步长 (默认: 0.0005)')
    parser.add_argument('--time', type=float, default=1.0, help='总模拟时间 (默认: 1.0)')
    parser.add_argument('--workers', type=int, default=None, help='进程数 (默认: CPU核心数)')
    parser.add_argument('--output', type=str, default='output', help='输出目录 (默认: output)')
    parser.add_argument('--interval', type=int, default=20, help='保存帧间隔 (默认: 20)')
    parser.add_argument('--init', type=str, default='zero', choices=['zero', 'gaussian'],
                        help='初始温度场类型 (默认: zero)')

    parser.add_argument('--path-type', type=str, default='line',
                        choices=['line', 'circle', 'sine', 'stationary'],
                        help='热源路径类型 (默认: line)')
    parser.add_argument('--power', type=float, default=500.0, help='热源功率 (默认: 500)')
    parser.add_argument('--source-radius', type=float, default=0.05,
                        help='热源高斯半径 (默认: 0.05)')
    parser.add_argument('--power-scale', type=float, default=1.0,
                        help='功率缩放系数 (默认: 1.0)')

    parser.add_argument('--start', type=str, default='-0.8,0.0',
                        help='直线起点 (x,y)，如 "-0.8,0.0"')
    parser.add_argument('--end', type=str, default='0.8,0.0',
                        help='直线终点 (x,y)，如 "0.8,0.0"')
    parser.add_argument('--center', type=str, default='0.0,0.0',
                        help='圆心 (x,y)，如 "0.0,0.0"')
    parser.add_argument('--radius-path', type=float, default=0.5, help='圆周半径 (默认: 0.5)')
    parser.add_argument('--angular-speed', type=float, default=3.0,
                        help='角速度 rad/s (默认: 3.0)')
    parser.add_argument('--x-range', type=str, default='-0.8,0.8',
                        help='正弦路径 x 范围 (x0,x1)')
    parser.add_argument('--amplitude', type=float, default=0.3,
                        help='正弦路径振幅 (默认: 0.3)')
    parser.add_argument('--frequency', type=float, default=1.5,
                        help='正弦路径频率 (默认: 1.5)')
    parser.add_argument('--position', type=str, default='0.0,0.0',
                        help='固定热源位置 (x,y)')

    args = parser.parse_args()

    path_type, path_params = parse_path_args(args)

    heat_source_config = {
        'path_type': path_type,
        'path_params': path_params,
        'power': args.power,
        'radius': args.source_radius,
        'power_scale': args.power_scale,
    }

    run_simulation(
        nx=args.nx,
        ny=args.ny,
        alpha=args.alpha,
        dt=args.dt,
        total_time=args.time,
        num_workers=args.workers,
        output_dir=args.output,
        save_interval=args.interval,
        init_type=args.init,
        heat_source_config=heat_source_config,
    )


if __name__ == '__main__':
    main()
