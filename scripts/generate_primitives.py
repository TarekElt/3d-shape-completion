"""Generate simple 3D primitive meshes for fast training experiments.

Writes `.off` files into a ModelNet-like layout under `data/ModelNet10/<class>/train` and `.../test`.

Usage:
    python scripts/generate_primitives.py --out-root data/ModelNet10 --class simulated --n 200

This avoids Blender and uses `trimesh` (already in requirements) to create and export meshes.
"""
import os
import argparse
import random
import numpy as np
import trimesh


def make_box(edge=1.0, extents=None):
    """Create an axis-aligned box.

    Accept either a single `edge` length (creates a cube) or an `extents` tuple
    (lengths in x,y,z). This makes the helper flexible for L-shapes which pass
    non-cubic extents.
    """
    if extents is None:
        extents = (edge, edge, edge)
    return trimesh.creation.box(extents=extents)



def make_sphere(radius=0.5):
    return trimesh.creation.icosphere(subdivisions=3, radius=radius)


def make_cylinder(radius=0.4, height=1.0):
    return trimesh.creation.cylinder(radius=radius, height=height, sections=64)


def make_lshape(a=(1.0, 0.3, 0.3), b=(0.3, 1.0, 0.3)):
    # two boxes forming an L; translate second to align
    box1 = make_box(extents=a)
    box2 = make_box(extents=b)
    # move box2 so that it connects at an edge with box1
    # place both boxes so their bottoms are centered at origin, then translate
    t = np.eye(4)
    # translate box1 negative along x/2 to make room
    t1 = np.eye(4)
    t1[0, 3] = - (a[0] / 2.0 - b[0] / 2.0)
    box1.apply_transform(t1)
    t2 = np.eye(4)
    t2[1, 3] = (a[1] / 2.0 - b[1] / 2.0)
    box2.apply_transform(t2)
    merged = trimesh.util.concatenate([box1, box2])
    return merged


PRIMITIVES = ['box', 'sphere', 'cylinder', 'lshape']


def random_transform(scale_range=(0.8, 1.2)):
    # Random uniform scaling
    s = random.uniform(*scale_range)

    # Random rotation around z-axis (safe for cubes & cylinders)
    angle_z = np.deg2rad(random.uniform(0, 360))
    Rz = trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1])

    # VERY SMALL tilt (optional)
    tilt_x = np.deg2rad(random.uniform(-5, 5))
    tilt_y = np.deg2rad(random.uniform(-5, 5))
    Rx = trimesh.transformations.rotation_matrix(tilt_x, [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(tilt_y, [0, 1, 0])

    # Combine transforms
    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = s

    return trimesh.transformations.concatenate_matrices(Rz, Rx, Ry, S)


def create_mesh(prim_type):
    if prim_type == 'box':
        e = np.random.uniform(0.6, 1.2)
        return make_box(extents=(e, e, e))
    if prim_type == 'sphere':
        r = random.uniform(0.4, 0.8)
        return make_sphere(radius=r)
    if prim_type == 'cylinder':
        r = random.uniform(0.3, 0.6)
        h = random.uniform(0.7, 1.3)
        return make_cylinder(radius=r, height=h)
    if prim_type == 'lshape':
        a = (random.uniform(0.4, 1.0), random.uniform(0.2, 0.6), random.uniform(0.2, 0.6))
        b = (random.uniform(0.2, 0.6), random.uniform(0.4, 1.0), random.uniform(0.2, 0.6))
        return make_lshape(a=a, b=b)
    raise ValueError('Unknown primitive: ' + str(prim_type))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default='data/ModelNet10', help='Root folder to create class under')
    parser.add_argument('--class', dest='class_name', default='simulated', help='Class name to use')
    parser.add_argument('--n', type=int, default=200, help='Total number of meshes to create (train+test)')
    parser.add_argument('--train-frac', type=float, default=0.9, help='Fraction to place in train split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = args.out_root
    cls = args.class_name
    n = args.n
    train_n = int(n * args.train_frac)
    test_n = n - train_n

    train_dir = os.path.join(out_root, cls, 'train')
    test_dir = os.path.join(out_root, cls, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f'Generating {n} primitive meshes into {out_root}/{cls} (train={train_n} test={test_n})')

    idx = 0
    for i in range(train_n):
        prim = random.choice(PRIMITIVES)
        mesh = create_mesh(prim)
        tf = random_transform()
        mesh.apply_transform(tf)
        # center and scale to unit bounding box for consistency
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        name = f'{cls}_{idx:04d}.off'
        path = os.path.join(train_dir, name)
        mesh.export(path)
        idx += 1

    for i in range(test_n):
        prim = random.choice(PRIMITIVES)
        mesh = create_mesh(prim)
        tf = random_transform()
        mesh.apply_transform(tf)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        name = f'{cls}_{idx:04d}.off'
        path = os.path.join(test_dir, name)
        mesh.export(path)
        idx += 1

    print('Done')


if __name__ == '__main__':
    main()
