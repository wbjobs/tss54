import numpy as np
import argparse
import os


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
    elif path_type == 'stationary':
        x, y = path_params['position']
        return x, y
    else:
        raise ValueError(f"Unknown path type: {path_type}")


def run_single_process(nx, ny, alpha, dt, total_time, path_type, path_params,
                       power, source_radius):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    T = np.zeros((ny, nx))
    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    effective_power = power * dx * dy / (2 * np.pi * source_radius ** 2)

    T_new = np.zeros_like(T)
    num_steps = int(total_time / dt)

    for step in range(num_steps):
        t = step * dt
        sx, sy = get_heat_source_position(path_type, path_params, t)

        r2 = (X - sx) ** 2 + (Y - sy) ** 2
        source = effective_power * np.exp(-r2 / (2 * source_radius ** 2))

        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2 +
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
        )
        T_new[1:-1, 1:-1] += source[1:-1, 1:-1] * dt
        T[1:-1, 1:-1] = T_new[1:-1, 1:-1]

        T[0, :] = 0.0
        T[-1, :] = 0.0
        T[:, 0] = 0.0
        T[:, -1] = 0.0

    return T


def main():
    parser = argparse.ArgumentParser(description='验证移动热源多进程结果与单进程一致')
    parser.add_argument('--nx', type=int, default=80)
    parser.add_argument('--ny', type=int, default=80)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time', type=float, default=0.5)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--power', type=float, default=500.0)
    parser.add_argument('--source-radius', type=float, default=0.05)
    parser.add_argument('--path-type', type=str, default='line')
    parser.add_argument('--frame', type=str, default='output/frame_0500.npy')
    parser.add_argument('--start', type=str, default='-0.6,0.0')
    parser.add_argument('--end', type=str, default='0.6,0.0')
    parser.add_argument('--center', type=str, default='0.0,0.0')
    parser.add_argument('--radius-path', type=float, default=0.4)
    parser.add_argument('--angular-speed', type=float, default=2.0)

    args = parser.parse_args()

    path_params = {}
    if args.path_type == 'line':
        path_params['start'] = tuple(map(float, args.start.split(',')))
        path_params['end'] = tuple(map(float, args.end.split(',')))
        path_params['total_time'] = args.time
    elif args.path_type == 'circle':
        path_params['center'] = tuple(map(float, args.center.split(',')))
        path_params['radius'] = args.radius_path
        path_params['angular_speed'] = args.angular_speed

    interior = args.ny - 2
    chunk = interior // args.workers
    chunk_rows = [ (i + 1) * chunk for i in range(args.workers - 1) ]

    print(f"Grid: {args.nx}x{args.ny}, Workers: {args.workers}, Path: {args.path_type}")
    print(f"Seams at rows: {chunk_rows}")
    print()

    if os.path.exists(args.frame):
        T_mp = np.load(args.frame)
        print("【Single-process reference (computing...)】")
        T_sp = run_single_process(
            args.nx, args.ny, args.alpha, args.dt, args.time,
            args.path_type, path_params, args.power, args.source_radius
        )
        print()

        diff = np.abs(T_mp - T_sp)
        print(f"【Multi-process vs Single-process absolute error】")
        print(f"  Max error:    {diff.max():.2e}")
        print(f"  Mean error:   {diff.mean():.2e}")
        print(f"  Temperature range (MP): [{T_mp.min():.4f}, {T_mp.max():.4f}]")
        print(f"  Temperature range (SP): [{T_sp.min():.4f}, {T_sp.max():.4f}]")
        print()

        print("Max error at seam rows:")
        for row in chunk_rows:
            local_max = max(diff[row, :].max(), diff[row + 1, :].max())
            print(f"  Row {row}/{row + 1}: {local_max:.2e}")

        non_seam_max = 0.0
        for row in range(1, args.ny - 1):
            is_seam = any(abs(row - r) <= 1 for r in chunk_rows)
            if not is_seam:
                non_seam_max = max(non_seam_max, diff[row, :].max())
        print(f"Non-seam region max error: {non_seam_max:.2e}")

        if diff.max() < 1e-8:
            print("\n✅ Multi-process matches single process perfectly! Seams are correct.")
        elif diff.max() < 1e-4:
            print("\n✅ Multi-process is accurate (error < 1e-4).")
        else:
            print("\n⚠️  Significant error detected.")
    else:
        print(f"Frame file not found: {args.frame}")
        print("Run the simulation first, e.g.:")
        print(f"  python heat_simulation.py --nx {args.nx} --ny {args.ny} "
              f"--time {args.time} --dt {args.dt} --alpha {args.alpha} "
              f"--power {args.power} --source-radius {args.source_radius} "
              f"--path-type {args.path_type} --workers {args.workers}")


if __name__ == '__main__':
    main()
