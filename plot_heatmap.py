import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import re


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


def plot_heatmap(npy_file, output_file=None, cmap='hot', vmin=None, vmax=None,
                 show=True, source_pos=None, source_radius=None):
    T = np.load(npy_file)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[-1, 1, -1, 1])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (°C)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Temperature Distribution - {os.path.basename(npy_file)}')

    if source_pos is not None:
        sx, sy = source_pos
        ax.plot(sx, sy, 'w*', markersize=12, markeredgecolor='k', label='Heat Source')
        if source_radius is not None:
            circle = plt.Circle((sx, sy), source_radius, color='white',
                                fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_artist(circle)
        ax.legend()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")

    if show:
        plt.show()

    plt.close()


def animate_frames(output_dir, output_file=None, cmap='hot', vmin=None, vmax=None,
                   fps=10, source_config=None, dt=None):
    npy_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith('frame_') and f.endswith('.npy')
    ], key=lambda f: int(re.search(r'frame_(\d+)\.npy', f).group(1)))

    if not npy_files:
        print(f"No frame files found in {output_dir}")
        return

    first_frame = np.load(os.path.join(output_dir, npy_files[0]))
    ny, nx = first_frame.shape

    if vmin is None:
        vmin = first_frame.min()
    if vmax is None:
        vmax = max(first_frame.max(), 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(first_frame, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[-1, 1, -1, 1])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (°C)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    title = ax.set_title('Frame 0')

    source_marker = None
    source_circle = None
    if source_config is not None:
        source_marker, = ax.plot([], [], 'w*', markersize=12,
                                 markeredgecolor='k', label='Heat Source')
        if source_config.get('radius'):
            source_circle = plt.Circle((0, 0), source_config['radius'],
                                       color='white', fill=False,
                                       linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_artist(source_circle)
        ax.legend()

    def update(frame_idx):
        f = npy_files[frame_idx]
        T = np.load(os.path.join(output_dir, f))
        im.set_array(T)

        frame_num = int(re.search(r'frame_(\d+)\.npy', f).group(1))
        title.set_text(f'Temperature Distribution - Frame {frame_num}')

        if source_config is not None and dt is not None:
            t = frame_num * dt
            sx, sy = get_heat_source_position(
                source_config['path_type'], source_config['path_params'], t
            )
            source_marker.set_data([sx], [sy])
            if source_circle is not None:
                source_circle.center = (sx, sy)

        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(npy_files), interval=1000 / fps, blit=False
    )

    if output_file:
        if output_file.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps)
        else:
            writer = animation.PillowWriter(fps=fps)
        ani.save(output_file, writer=writer)
        print(f"Animation saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='从.npy文件生成温度分布热力图')
    subparsers = parser.add_subparsers(dest='mode', help='模式: single(单帧) 或 animate(动画)')

    single_parser = subparsers.add_parser('single', help='生成单帧热力图')
    single_parser.add_argument('input', type=str, help='输入的.npy文件路径')
    single_parser.add_argument('--output', '-o', type=str, default=None, help='输出图片路径')
    single_parser.add_argument('--cmap', type=str, default='hot', help='颜色映射 (默认: hot)')
    single_parser.add_argument('--vmin', type=float, default=None, help='颜色最小值')
    single_parser.add_argument('--vmax', type=float, default=None, help='颜色最大值')
    single_parser.add_argument('--no-show', action='store_true', help='不显示图片窗口')
    single_parser.add_argument('--source', type=str, default=None,
                               help='热源位置 "x,y"，如 "0.3,0.1"')
    single_parser.add_argument('--source-radius', type=float, default=None,
                               help='热源半径标记')

    anim_parser = subparsers.add_parser('animate', help='生成动画')
    anim_parser.add_argument('input_dir', type=str, help='包含帧.npy文件的目录')
    anim_parser.add_argument('--output', '-o', type=str, default=None,
                             help='输出动画文件 (.gif 或 .mp4)')
    anim_parser.add_argument('--cmap', type=str, default='hot', help='颜色映射')
    anim_parser.add_argument('--vmin', type=float, default=None, help='颜色最小值')
    anim_parser.add_argument('--vmax', type=float, default=None, help='颜色最大值')
    anim_parser.add_argument('--fps', type=int, default=10, help='帧率 (默认: 10)')
    anim_parser.add_argument('--no-show', action='store_true', help='不显示动画窗口')
    anim_parser.add_argument('--path-type', type=str, default=None,
                             choices=['line', 'circle', 'sine', 'stationary'],
                             help='热源路径类型（用于动画中标记热源）')
    anim_parser.add_argument('--source-power', type=float, default=500.0)
    anim_parser.add_argument('--source-radius', type=float, default=0.05)
    anim_parser.add_argument('--start', type=str, default='-0.8,0.0')
    anim_parser.add_argument('--end', type=str, default='0.8,0.0')
    anim_parser.add_argument('--center', type=str, default='0.0,0.0')
    anim_parser.add_argument('--radius-path', type=float, default=0.5)
    anim_parser.add_argument('--angular-speed', type=float, default=3.0)
    anim_parser.add_argument('--x-range', type=str, default='-0.8,0.8')
    anim_parser.add_argument('--amplitude', type=float, default=0.3)
    anim_parser.add_argument('--frequency', type=float, default=1.5)
    anim_parser.add_argument('--position', type=str, default='0.0,0.0')
    anim_parser.add_argument('--dt', type=float, default=0.0005, help='时间步长 (用于计算热源位置)')
    anim_parser.add_argument('--total-time', type=float, default=1.0)

    args = parser.parse_args()

    if args.mode is None or args.mode == 'single':
        if args.mode is None:
            print("默认使用 single 模式")
            input_file = args.input if hasattr(args, 'input') and args.input else args.input_dir
        else:
            input_file = args.input

        source_pos = None
        if args.source:
            sx, sy = map(float, args.source.split(','))
            source_pos = (sx, sy)

        plot_heatmap(
            input_file,
            output_file=args.output,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            show=not args.no_show,
            source_pos=source_pos,
            source_radius=args.source_radius
        )

    elif args.mode == 'animate':
        source_config = None
        if args.path_type:
            path_params = {}
            if args.path_type == 'line':
                path_params['start'] = tuple(map(float, args.start.split(',')))
                path_params['end'] = tuple(map(float, args.end.split(',')))
                path_params['total_time'] = args.total_time
            elif args.path_type == 'circle':
                path_params['center'] = tuple(map(float, args.center.split(',')))
                path_params['radius'] = args.radius_path
                path_params['angular_speed'] = args.angular_speed
            elif args.path_type == 'sine':
                path_params['x_range'] = tuple(map(float, args.x_range.split(',')))
                path_params['amplitude'] = args.amplitude
                path_params['frequency'] = args.frequency
                path_params['total_time'] = args.total_time
            elif args.path_type == 'stationary':
                path_params['position'] = tuple(map(float, args.position.split(',')))

            source_config = {
                'path_type': args.path_type,
                'path_params': path_params,
                'power': args.source_power,
                'radius': args.source_radius,
            }

        animate_frames(
            args.input_dir,
            output_file=args.output,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            fps=args.fps,
            source_config=source_config,
            dt=args.dt
        )


if __name__ == '__main__':
    main()
