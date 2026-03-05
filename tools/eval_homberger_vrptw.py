"""评估 Homberger VRPTW 基准的轻量脚本

用法示例:
    python EARLI/tools/eval_homberger_vrptw.py --input-dir EARLI/homberger/homberger_1000_customer_instances \
        --output-csv results_homberger.csv --method edd

脚本功能：
- 解析 Homberger `.TXT` 文件（使用仓库内解析器）
- 对每个实例运行简单的贪心基线（最近可行 / 最早到期优先）
- 输出每个实例的汇总 CSV（实例名、顾客数、车辆数、总路程、状态）
"""

import os
import glob
import argparse
import csv
import numpy as np

from earli.benchmark_parser import parse_homberger_instance


def compute_distance_matrix(positions):
    pos = np.asarray(positions)
    diff = pos[:, None, :] - pos[None, :, :]
    dm = np.sqrt((diff ** 2).sum(axis=-1))
    return dm


def greedy_vrptw_solver(positions, demand, capacity, time_windows, service_times, method='edd'):
    """Greedy baseline for VRPTW.

    Args:
        positions: (n_nodes,2) array (node 0 = depot)
        demand: (n_nodes,) array (depot demand usually 0)
        capacity: scalar
        time_windows: (n_nodes,2) array [ready, due]
        service_times: (n_nodes,) array
        method: 'edd' (earliest due date) or 'nn' (nearest neighbor)

    Returns:
        dict with keys: total_distance, n_vehicles, served_mask (bool array), routes (list of lists)
    """
    n_nodes = positions.shape[0]
    depot = 0
    customers = list(range(1, n_nodes))
    dm = compute_distance_matrix(positions)

    unserved = set(customers)
    routes = []
    total_distance = 0.0

    while unserved:
        route = [depot]
        cur = depot
        cap_left = capacity
        cur_time = service_times[depot] if depot < len(service_times) else 0.0

        while True:
            # build candidate list of feasible customers
            candidates = []
            for j in list(unserved):
                if demand[j] > cap_left:
                    continue
                travel = dm[cur, j]
                arrive = cur_time + service_times[cur] + travel
                ready, due = time_windows[j]
                arrive_effective = max(arrive, ready)
                if arrive_effective <= due:
                    candidates.append((j, arrive, arrive_effective, travel, due))

            if not candidates:
                # no feasible next -> return to depot
                if cur != depot:
                    total_distance += dm[cur, depot]
                    route.append(depot)
                break

            if method == 'nn':
                # nearest feasible neighbor (by travel from cur)
                candidates.sort(key=lambda x: x[3])
            else:
                # earliest due date
                candidates.sort(key=lambda x: x[4])

            nxt, arrive, arrive_effective, travel, _ = candidates[0]
            # move to nxt
            total_distance += travel
            route.append(nxt)
            cur = nxt
            # update time and capacity
            cur_time = arrive_effective + service_times[nxt]
            cap_left -= demand[nxt]
            unserved.remove(nxt)

        routes.append(route)

    served_mask = np.ones(n_nodes, dtype=bool)
    # depot always served
    return {
        'total_distance': float(total_distance),
        'n_vehicles': len(routes),
        'served_mask': served_mask,
        'routes': routes,
    }


def evaluate_directory(input_dir, output_csv, method='edd', glob_pattern='*.TXT'):
    files = sorted(glob.glob(os.path.join(input_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f'No files in {input_dir} matching {glob_pattern}')

    header = ['instance', 'n_nodes', 'n_customers', 'n_vehicles', 'total_distance', 'status']
    rows = []

    for f in files:
        inst = parse_homberger_instance(f)
        positions = inst['positions']
        demand = inst['demand']
        capacity = inst['capacity']
        time_windows = inst['time_windows']
        service_times = inst['service_times']

        res = greedy_vrptw_solver(positions, demand, capacity, time_windows, service_times, method=method)

        instance_name = inst.get('instance_name') or os.path.splitext(os.path.basename(f))[0]
        n_nodes = positions.shape[0]
        n_customers = n_nodes - 1
        status = 'ok'

        rows.append([instance_name, n_nodes, n_customers, res['n_vehicles'], f"{res['total_distance']:.3f}", status])

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'Wrote summary for {len(files)} instances to {output_csv}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate Homberger VRPTW with simple greedy baselines')
    parser.add_argument('--input-dir', required=True, help='Directory containing Homberger .TXT files')
    parser.add_argument('--output-csv', required=True, help='Output CSV summary path')
    parser.add_argument('--method', choices=['edd', 'nn'], default='edd', help='Greedy policy: edd or nn')
    parser.add_argument('--glob', default='*.TXT', help='Glob pattern for instance files')
    args = parser.parse_args()

    evaluate_directory(args.input_dir, args.output_csv, method=args.method, glob_pattern=args.glob)


if __name__ == '__main__':
    main()
