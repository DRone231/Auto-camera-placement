import math
import sys
import pathlib
import argparse

import matplotlib.colors
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import skimage.io
import skimage.draw


SOLUTION_ANGLE = [
    None,
    45*0,
    45*1,
    45*2,
    45*3,
    45*4,
    45*5,
    45*6,
    45*7,
]

SOLUTION_WEIGHT = np.array([
    4, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=np.float32)

SOLUTION_WEIGHT /= SOLUTION_WEIGHT.sum()

#ген случ распо камер
def sol_random(rng, count):
    return rng.choice(len(SOLUTION_ANGLE), size=count, p=SOLUTION_WEIGHT)


def sol_mutate(rng, sol, times):
    sol = sol.copy()
    for i in range(times):
        n = rng.choice(len(sol))
        sol[n] = rng.choice(len(SOLUTION_ANGLE), p=SOLUTION_WEIGHT)
    return sol


def compute_coverage(sol, point_x, point_y, map_area, args):
    coverage = np.zeros((map_area.shape[0], map_area.shape[1]), dtype=np.int)

    camera_count = 0

    ANGLE_RANGE_DEGREES = args.camera_angle_vision / 2  # Делим диапазон на две части
    MINIMAL_VISION = args.camera_min_distance
    MAXIMAL_VISION = args.camera_max_distance

    xx, yy = np.meshgrid(np.arange(map_area.shape[0]), np.arange(map_area.shape[1]))

    for point_index, point_sol in enumerate(sol):
        point_angle = SOLUTION_ANGLE[point_sol]
        if point_angle is not None:
            camera_count += 1
            xp = point_x[point_index]
            yp = point_y[point_index]

            distance_map = ((xx - xp) ** 2 + (yy - yp) ** 2)
            # arctan2 возвращает значение в диапазоне [-pi, +pi], приводим к [0, 2*pi]
            angle_map = (np.arctan2((yy - yp), (xx - xp)) + (math.pi * 2)) % (math.pi * 2)

            start_angle = np.radians(SOLUTION_ANGLE[sol[point_index]] - ANGLE_RANGE_DEGREES + 360) % (math.pi * 2)
            finish_angle = np.radians(SOLUTION_ANGLE[sol[point_index]] + ANGLE_RANGE_DEGREES)

            if start_angle < finish_angle:
                point_coverage = (distance_map >= MINIMAL_VISION ** 2) & (distance_map <= MAXIMAL_VISION ** 2) & (angle_map >= start_angle) & (angle_map <= finish_angle)
            else:
                point_coverage = (distance_map >= MINIMAL_VISION ** 2) & (distance_map <= MAXIMAL_VISION ** 2) & ((angle_map >= start_angle) | (angle_map <= finish_angle))

            # Проверяем каждую точку на предмет прохода через стены.
            # TODO: подумать, как делать это не циклом, сейчас довольно долго
            xs_covered, ys_covered = np.where(point_coverage)
            walls_map = ~(map_area.transpose())
            for point in zip(xs_covered, ys_covered):
                vision_ray = np.zeros(walls_map.shape, dtype=np.bool)
                line_x, line_y = skimage.draw.line(point[0], point[1], yp, xp)
                vision_ray[line_x, line_y] = True

                if (vision_ray & walls_map).any():
                # Есть пересечение со стеной, удаляем эту точку, мы её не видим на самом деле
                    point_coverage[point[0], point[1]] = False

            # Мы работали в транспонированной матрице из-за различий между skimage и numpy-шным мешгридом.
            # TODO: путаницу с транспонированиями можно привести в порядок
            point_coverage = point_coverage.transpose()
            assert(coverage.shape == point_coverage.shape)
            coverage[np.where(point_coverage)] += 1

    return coverage, camera_count


def sol_loss(sol, point_x, point_y, map_area, map_importance, args):
    assert(map_area.shape == map_importance.shape)

    coverage, camera_count = compute_coverage(sol, point_x, point_y, map_area, args)

    diff = map_importance - coverage

    overcoverage = -diff[diff < 0].sum()
    undercoverage = diff[diff > 0].sum()

    OVECOVERAGE_PENALTY = args.overcoverage_penalty
    UNDERCOVERAGE_PENALTY = args.undercoverage_penalty
    CAMERA_PENALTY = args.camera_penalty

    loss = overcoverage * OVECOVERAGE_PENALTY + undercoverage * UNDERCOVERAGE_PENALTY + camera_count * CAMERA_PENALTY

    return loss


def points_read(points_path):
    data = skimage.io.imread(points_path)

    data = data.sum(axis=2) > 0
    data = data.astype(np.bool)

    point_x, point_y = np.where(data)
    point_count = len(point_y)

    return point_count, point_y, point_x


def importance_read(path):
    data = skimage.io.imread(path)
    data = data.sum(axis=2).astype(np.float32) / 50
    return data


def area_read(path):
    data = skimage.io.imread(path)
    data = data.sum(axis=2) > 0
    return data.astype(np.bool)


def sol_repr(sol, loss):
    camera_count = (sol>0).sum()
    return f"{loss:0.2f} ({camera_count} camera(s))"


def show_coverage(map_area, cur_sol, point_x, point_y, args):
    img = np.full((map_area.shape[0], map_area.shape[1], 3), 255)
    coverage, _ = compute_coverage(cur_sol, point_x, point_y, map_area, args)
    max_coverage = coverage.max()
    coverage = coverage / max_coverage

    img[:, :, 1] = (1 - coverage) * 255
    img[:, :, 2] = (1 - coverage) * 255
    img[np.where(~map_area)] = [0, 0, 0]
    for point_index, point_sol in enumerate(cur_sol):
        point_angle = SOLUTION_ANGLE[point_sol]
        if point_angle is not None:
            xp = point_x[point_index]
            yp = point_y[point_index]
            img[xp, yp] = [0, 0, 255]

    img = img.astype(np.int)

    plt.imshow(img, interpolation=None)
    red_patch = mpatches.Patch(color='red', label='{} покрытий'.format(int(max_coverage)))
    white_patch = mpatches.Patch(color='white', label='0 покрытий')
    plt.legend(handles=[red_patch, white_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(args.output_file)


def main(argv):
    here_path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="store", default=here_path.joinpath('maps', 'area.tif'))
    parser.add_argument("--importance", action="store", default=here_path.joinpath('maps', 'importance.tif'))
    parser.add_argument("--cameras", action="store", default=here_path.joinpath('maps', 'points.tif'))
    parser.add_argument("--seed", type=int, action="store", default=0)

    parser.add_argument("--overcoverage_penalty", type=float, action="store", default=1.0)
    parser.add_argument("--undercoverage_penalty", type=float, action="store", default=3.0)
    parser.add_argument("--camera_penalty", type=float, action="store", default=1000.0)

    parser.add_argument("--camera_angle_vision", type=float, action="store", default=60.0)
    parser.add_argument("--camera_min_distance", type=float, action="store", default=8.0)
    parser.add_argument("--camera_max_distance", type=float, action="store", default=30.0)

    parser.add_argument("--epochs", type=int, action="store", default=100)
    parser.add_argument("--changes_per_mutation", type=int, action="store", default=1)

    parser.add_argument("--output_file", action="store", default="result.png")

    args = parser.parse_args(argv)

    rng = np.random.RandomState(args.seed)

    map_area = area_read(args.map)
    map_importance = importance_read(args.importance)
    point_count, point_y, point_x = points_read(args.cameras)

    cur_sol = sol_random(rng, point_count)
    cur_loss = sol_loss(cur_sol, point_x, point_y, map_area, map_importance, args)

    best_sol = cur_sol.copy()
    best_loss = cur_loss

    EPOCH_MAX = args.epochs
    CHANGES_PER_MUTATION = args.changes_per_mutation

    for epoch_n in range(EPOCH_MAX):
        next_sol = sol_mutate(rng, cur_sol, CHANGES_PER_MUTATION)
        next_loss = sol_loss(next_sol, point_x, point_y, map_area, map_importance, args)

        print(f'Epoch {epoch_n}')
        print(f'    best: {sol_repr(best_sol,best_loss)}')
        print(f'    next: {sol_repr(next_sol,next_loss)}')
        print(f' current: {sol_repr(cur_sol,cur_loss)}')
        print()

        choice_next = False

        if next_loss < cur_loss:
            choice_next = True
        else:
            Q = ((EPOCH_MAX - epoch_n) / EPOCH_MAX) * 100.0
            diff = next_loss - cur_loss
            p = np.exp(-(diff / Q))
            if rng.rand() < p:
                choice_next = True

        if choice_next:
            cur_loss = next_loss
            cur_sol = next_sol

        if best_loss > cur_loss:
            best_loss = cur_loss
            best_sol = cur_sol

    show_coverage(map_area, cur_sol, point_x, point_y, args)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
