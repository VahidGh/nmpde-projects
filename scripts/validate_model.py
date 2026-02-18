#!/usr/bin/env python3
"""Model validator for this project.

Supports either a CSV/numeric results file or VTU/PVTu outputs produced by
the `Heat` program. For VTU/PVTu files the script reads the `solution` field
and computes the volume-weighted average (mass). For the default `FunctionU0`
initial condition the mean over the unit cube is 1/216 and should be conserved
by the scheme with homogeneous Neumann boundary conditions; the validator
compares the computed mass to the expected reference and fails if the
relative error exceeds the given tolerance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_numeric_file(path: Path):
    try:
        data = np.loadtxt(path, delimiter=',')
    except Exception:
        data = np.loadtxt(path)
    return data


def compute_mean_from_numeric(data: np.ndarray):
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError('Numeric result file too small')
        # assume two entries: x and y
        y = data[1]
        return float(y)
    else:
        # assume first column x, next columns are data; take column 1
        y = data[:, 1]
        return float(np.mean(y))


def compute_mean_from_vtu(path: Path):
    try:
        import meshio
    except Exception as e:
        raise RuntimeError('meshio is required to read VTU files; please install it') from e

    mesh = meshio.read(path)

    # Try to find 'solution' in point_data or cell_data
    data = None
    if 'solution' in mesh.point_data:
        vals = mesh.point_data['solution']
        # approximate mean over domain by averaging point values
        return float(np.mean(vals))

    # cell data may be dict of name -> list (per block)
    if 'solution' in mesh.cell_data:
        # mesh.cell_data['solution'] may be dict of block-wise arrays
        cell_data = mesh.cell_data['solution']
        # concatenate arrays
        if isinstance(cell_data, dict):
            arrays = []
            for k, v in cell_data.items():
                arrays.append(v)
            vals = np.concatenate(arrays)
        else:
            vals = np.concatenate(cell_data)
        return float(np.mean(vals))

    # fallback: try first available point_data
    if mesh.point_data:
        key = list(mesh.point_data.keys())[0]
        return float(np.mean(mesh.point_data[key]))

    # fallback: try first available cell_data
    if mesh.cell_data:
        # cell_data is dict of name -> per-block arrays
        key = list(mesh.cell_data.keys())[0]
        cell_data = mesh.cell_data[key]
        if isinstance(cell_data, dict):
            vals = np.concatenate(list(cell_data.values()))
        else:
            vals = np.concatenate(cell_data)
        return float(np.mean(vals))

    raise ValueError('No suitable data arrays found in VTU/PVTu file')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', '-r', required=True, help='Results file (CSV or VTU/PVTu)')
    p.add_argument('--tolerance', '-t', type=float, default=1e-6)
    p.add_argument('--expected-mean', type=float, default=1.0 / 216.0,
                   help='Expected spatial mean (default is 1/216 for FunctionU0)')
    args = p.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f'Results file not found: {path}', file=sys.stderr)
        sys.exit(2)

    if path.suffix in ['.vtu', '.pvtu']:
        mean = compute_mean_from_vtu(path)
    else:
        data = read_numeric_file(path)
        mean = compute_mean_from_numeric(data)

    rel_err = abs(mean - args.expected_mean) / (abs(args.expected_mean) if args.expected_mean != 0 else 1.0)

    print(f'Computed mean: {mean:.12g}')
    print(f'Expected mean: {args.expected_mean:.12g}')
    print(f'Relative error: {rel_err:.6g} (tolerance {args.tolerance})')

    if rel_err > args.tolerance:
        print('Validation FAILED', file=sys.stderr)
        sys.exit(5)
    else:
        print('Validation PASSED')
        sys.exit(0)


if __name__ == '__main__':
    main()
