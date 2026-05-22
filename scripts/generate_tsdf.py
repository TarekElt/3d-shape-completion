

import argparse
import json
import os
import time
from datetime import datetime
import logging

import numpy as np
import trimesh

try:
    # scipy is optional for fastest KDTree; used as a reliable fallback
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_mesh(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Mesh file not found: {path}')
    mesh = trimesh.load(path, process=True)
    if mesh is None:
        raise RuntimeError(f'Failed to load mesh: {path}')
    if isinstance(mesh, trimesh.Scene):
        # pick the largest geometry if present
        if not mesh.geometry:
            raise RuntimeError('Scene contains no geometry')
        # choose the largest by number of vertices
        mesh = max(mesh.geometry.values(), key=lambda g: len(g.vertices))
    return mesh


def compute_cubic_bounds(mesh, padding=0.06):
    
    bmin = mesh.bounds[0].astype(np.float64)
    bmax = mesh.bounds[1].astype(np.float64)
    center = (bmin + bmax) / 2.0
    spans = bmax - bmin
    half = spans.max() / 2.0
    half = half * (1.0 + padding)
    bounds_min = center - half
    bounds_max = center + half
    return bounds_min.astype(np.float32), bounds_max.astype(np.float32)


def make_grid_flat(bounds_min, bounds_max, G):
   
    xs = np.linspace(bounds_min[0], bounds_max[0], G, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], G, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], G, dtype=np.float32)
    # Use broadcasting to avoid making a huge temporary 4D array
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    centers_flat = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T.astype(np.float32)
    return xs, ys, zs, centers_flat


def compute_signed_distance(mesh, centers_flat):
    
    t0 = time.perf_counter()
    N = centers_flat.shape[0]
    logging.info('Signed distance: computing for %d points', N)
    # 1) try trimesh signed_distance
    try:
        sd = trimesh.proximity.signed_distance(mesh, centers_flat)
        logging.info('Used trimesh.proximity.signed_distance (%.3fs)', time.perf_counter() - t0)
        return sd.astype(np.float32)
    except Exception as e:
        logging.warning('trimesh.signed_distance failed: %s', str(e))

    # 2) try trimesh.closest_point then sign with mesh.contains
    try:
        closest, distances, triangle_id = trimesh.proximity.closest_point(mesh, centers_flat)
        dist = np.linalg.norm(centers_flat - closest, axis=1).astype(np.float32)
        inside = mesh.contains(centers_flat)
        dist[inside] *= -1.0
        logging.info('Used trimesh.proximity.closest_point (%.3fs)', time.perf_counter() - t0)
        return dist
    except Exception as e:
        logging.warning('trimesh.closest_point failed: %s', str(e))

    # 3) KDTree on mesh vertices (less accurate but robust)
    if cKDTree is None:
        raise RuntimeError('scipy.spatial.cKDTree is required for KDTree fallback but not available')
    try:
        tree = cKDTree(mesh.vertices.astype(np.float32))
        dists, _ = tree.query(centers_flat)
        dists = dists.astype(np.float32)
        inside = mesh.contains(centers_flat)
        dists[inside] *= -1.0
        logging.info('Used KDTree fallback (%.3fs)', time.perf_counter() - t0)
        return dists
    except Exception as e:
        logging.error('KDTree fallback failed: %s', str(e))
        raise


def truncate_and_normalize(sdf, trunc):
    tsdf = np.clip(sdf, -trunc, trunc) / float(trunc)
    return tsdf.astype(np.float32)


def create_partial(tsdf, occ, partial_type='front', camera_axis='+z', noise=0.0, rng=None):
    
    G = tsdf.shape[0]
    partial = tsdf.copy()
    if rng is None:
        rng = np.random.RandomState()

    if partial_type == 'none':
        return partial

    if partial_type == 'fixed-half':
        # back half in -z is unknown (keep same orientation as classic script)
        partial[:, :, G // 2 :] = 1.0
        mask_observed = np.ones_like(partial, dtype=bool)
        mask_observed[:, :, : G // 2] = True
        mask_observed[:, :, G // 2 :] = False

    elif partial_type == 'random-half':
        # choose random axis and side, mask that half as unknown
        axis = rng.choice([0, 1, 2])
        side = rng.choice([0, 1])
        mask = np.ones((G, G, G), dtype=bool)
        idx = slice(None)
        if axis == 0:
            if side == 0:
                mask[: G // 2, :, :] = False
            else:
                mask[G // 2 :, :, :] = False
        elif axis == 1:
            if side == 0:
                mask[:, : G // 2, :] = False
            else:
                mask[:, G // 2 :, :] = False
        else:
            if side == 0:
                mask[:, :, : G // 2] = False
            else:
                mask[:, :, G // 2 :] = False
        partial[~mask] = 1.0
        mask_observed = mask

    elif partial_type == 'front' or partial_type == 'camera':
        # axis-aligned camera occlusion: find first occupied voxel along camera axis
        # compute reversed occupancy and argmax to find first surface from camera
        if camera_axis == '+z':
            rev = occ[:, :, ::-1]
            has = rev.any(axis=2)
            idx_rev = np.argmax(rev, axis=2)
            first = (G - 1) - idx_rev
            first[~has] = -1
            grid_z = np.arange(G, dtype=np.int32)[None, None, :]
            mask_occluded = grid_z < first[:, :, None]
            # where first == -1 no surface -> mark everything unknown
            mask_occluded[first == -1, :] = True
            partial[mask_occluded] = 1.0
            mask_observed = ~mask_occluded
        elif camera_axis == '-z':
            rev = occ[:, :, :]
            has = rev.any(axis=2)
            idx = np.argmax(rev, axis=2)
            first = idx
            first[~has] = -1
            grid_z = np.arange(G, dtype=np.int32)[None, None, :]
            mask_occluded = grid_z > first[:, :, None]
            mask_occluded[first == -1, :] = True
            partial[mask_occluded] = 1.0
            mask_observed = ~mask_occluded
        elif camera_axis in ('+x', '-x'):
            # handle x-axis symmetric to z
            if camera_axis == '+x':
                rev = occ[::-1, :, :]
                has = rev.any(axis=0)
                idx_rev = np.argmax(rev, axis=0)
                first = (G - 1) - idx_rev
                first[~has] = -1
                grid_x = np.arange(G, dtype=np.int32)[:, None, None]
                mask_occluded = grid_x < first[None, :, :]
                mask_occluded[first == -1, :] = True
                partial[mask_occluded] = 1.0
                mask_observed = ~mask_occluded
            else:
                rev = occ[:, :, :]
                has = rev.any(axis=0)
                idx = np.argmax(rev, axis=0)
                first = idx
                first[~has] = -1
                grid_x = np.arange(G, dtype=np.int32)[:, None, None]
                mask_occluded = grid_x > first[None, :, :]
                mask_occluded[first == -1, :] = True
                partial[mask_occluded] = 1.0
                mask_observed = ~mask_occluded
        elif camera_axis in ('+y', '-y'):
            if camera_axis == '+y':
                rev = occ[:, ::-1, :]
                has = rev.any(axis=1)
                idx_rev = np.argmax(rev, axis=1)
                first = (G - 1) - idx_rev
                first[~has] = -1
                grid_y = np.arange(G, dtype=np.int32)[None, :, None]
                mask_occluded = grid_y < first[:, None, :]
                mask_occluded[first == -1, :] = True
                partial[mask_occluded] = 1.0
                mask_observed = ~mask_occluded
            else:
                rev = occ[:, :, :]
                has = rev.any(axis=1)
                idx = np.argmax(rev, axis=1)
                first = idx
                first[~has] = -1
                grid_y = np.arange(G, dtype=np.int32)[None, :, None]
                mask_occluded = grid_y > first[:, None, :]
                mask_occluded[first == -1, :] = True
                partial[mask_occluded] = 1.0
                mask_observed = ~mask_occluded
        else:
            raise ValueError('Unsupported camera_axis: ' + str(camera_axis))

    elif partial_type == 'spherical':
        # choose a sphere in front of the mesh center and mark voxels outside it unknown
        # radius fraction of bbox diagonal
        diag = np.linalg.norm(np.array([1.0, 1.0, 1.0]))  # placeholder; caller replaces
        # We'll choose center at near-front-of-box
        cx = G // 2 + rng.randint(-G // 8, G // 8)
        cy = G // 2 + rng.randint(-G // 8, G // 8)
        cz = int(G * 0.75)
        R = int(G * 0.25 + rng.randint(0, G // 8))
        gx, gy, gz = np.ogrid[:G, :G, :G]
        mask_inside = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2 <= R * R
        partial[~mask_inside] = 1.0
        mask_observed = mask_inside

    else:
        raise ValueError('Unknown partial_type: ' + str(partial_type))

    # add optional gaussian noise to observed region
    if noise and noise > 0.0:
        if 'mask_observed' not in locals():
            mask_observed = partial != 1.0
        noise_arr = rng.normal(scale=noise, size=partial.shape).astype(np.float32)
        partial[mask_observed] = np.clip(partial[mask_observed] + noise_arr[mask_observed], -1.0, 1.0)

    return partial.astype(np.float32)


def save_npz(out_path, tsdf, partial, occ, bounds_min, bounds_max, meta):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # Save metadata as JSON string for portability
    meta_json = json.dumps(meta)
    np.savez_compressed(
        out_path,
        tsdf=tsdf.astype(np.float32),
        partial=partial.astype(np.float32),
        occ=occ.astype(np.uint8),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        meta=np.array(meta_json),
    )


def generate(file_in, G=32, trunc=None, out=None, padding=0.06, partial_type='front', camera_axis='+z', partial_noise=0.0, seed=None):
    setup_logging()
    t_start = time.perf_counter()
    rng = np.random.RandomState(seed)

    mesh = load_mesh(file_in)
    mesh_name = os.path.basename(file_in)

    bounds_min, bounds_max = compute_cubic_bounds(mesh, padding=padding)
    diag = np.linalg.norm(bounds_max - bounds_min)
    if trunc is None or trunc <= 0.0:
        trunc = float(max(1e-6, 0.1 * diag))
        logging.info('Auto truncation chosen: %.6f (0.1 * diag=%.6f)', trunc, 0.1 * diag)

    if G > 256:
        logging.warning('High grid resolution G=%d may use lots of memory; be careful.', G)

    xs, ys, zs, centers_flat = make_grid_flat(bounds_min, bounds_max, G)

    logging.info('Grid prepared: %d x %d x %d = %d voxels', G, G, G, centers_flat.shape[0])

    t0 = time.perf_counter()
    sdf_flat = compute_signed_distance(mesh, centers_flat)
    logging.info('Signed distance computation took %.3fs', time.perf_counter() - t0)

    sdf = sdf_flat.reshape((G, G, G))
    tsdf = truncate_and_normalize(sdf, trunc)

    occ = (sdf < 0).astype(np.uint8)

    # partial simulation
    t1 = time.perf_counter()
    partial = create_partial(tsdf, occ, partial_type=partial_type, camera_axis=camera_axis, noise=partial_noise, rng=rng)
    logging.info('Partial generation (%s) took %.3fs', partial_type, time.perf_counter() - t1)

    # checks
    zero_crossings = np.any(np.abs(tsdf) <= (1.0 / 255.0))
    if not zero_crossings:
        logging.warning('No zero-crossing detected in TSDF. Mesh may be outside bounds or too small.')

    # build metadata
    meta = {
        'mesh': mesh_name,
        'generated': datetime.utcnow().isoformat() + 'Z',
        'grid': int(G),
        'truncation': float(trunc),
        'padding': float(padding),
        'partial_type': partial_type,
        'camera_axis': camera_axis,
        'partial_noise': float(partial_noise),
    }

    # derive default out name if not provided
    if out is None:
        base = os.path.splitext(os.path.basename(file_in))[0]
        out = os.path.join('outputs', f'{base}_res{G}.npz')

    save_npz(out, tsdf, partial, occ, bounds_min, bounds_max, meta)
    elapsed = time.perf_counter() - t_start

    logging.info('Wrote %s', out)
    logging.info('Summary: TSDF %dx%dx%d, trunc=%.6f, bounds_min=%s, bounds_max=%s, partial=%s',
                 G, G, G, trunc, np.array2string(bounds_min, precision=3), np.array2string(bounds_max, precision=3), partial_type)
    logging.info('Total time: %.3fs', elapsed)


def main():
    parser = argparse.ArgumentParser(description='Generate TSDF (.npz) from mesh')
    parser.add_argument('--file', required=True, help='.off/.ply/.obj input file')
    parser.add_argument('--grid', type=int, default=32, help='grid dimension (G). Supported: 32,64,128,256')
    parser.add_argument('--trunc', type=float, default=None, help='TSDF truncation (world units). If omitted, set to 0.1*bbox_diag')
    parser.add_argument('--out', default=None, help='output .npz file (default: outputs/<mesh>_res<G>.npz)')
    parser.add_argument('--padding', type=float, default=0.06, help='padding fraction for cubic bbox (default 0.06 = 6%%)')
    parser.add_argument('--partial-type', default='front', choices=['front', 'fixed-half', 'random-half', 'spherical', 'none', 'random-half', 'camera'], help='partial observation type (default: front)')
    parser.add_argument('--camera-axis', default='+z', choices=['+x', '-x', '+y', '-y', '+z', '-z'], help='axis-aligned camera direction for view-dependent partials')
    parser.add_argument('--partial-noise', type=float, default=0.0, help='stddev of gaussian noise added to observed TSDF region')
    parser.add_argument('--seed', type=int, default=42, help='random seed for partial generation')
    args = parser.parse_args()

    generate(args.file, G=args.grid, trunc=args.trunc, out=args.out, padding=args.padding,
             partial_type=args.partial_type, camera_axis=args.camera_axis, partial_noise=args.partial_noise, seed=args.seed)


if __name__ == '__main__':
    main()
