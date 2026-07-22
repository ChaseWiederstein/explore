from pathlib import Path
import sys

import open3d as o3d


ROOT = Path(__file__).resolve().parent


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    mesh_path = ROOT / "exports" / f"tsdf_step_{step}.ply"
    frontier_path = ROOT / "exports" / f"frontiers_step_{step}.ply"
    centroid_path = ROOT / "exports" / f"centroids_step_{step}.ply"
    viewpoint_path = ROOT / "exports" / f"viewpoints_step_{step}.ply"
    heading_path = ROOT / "exports" / f"viewpoint_headings_step_{step}.ply"
    reference_path = ROOT / "exports" / f"reference_step_{step}.ply"
    kino_path = ROOT / "exports" / f"kino_path_step_{step}.ply"

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Empty TSDF mesh: {mesh_path}")
    mesh.compute_vertex_normals()

    geometries = [mesh]
    if frontier_path.exists():
        frontiers = o3d.io.read_point_cloud(str(frontier_path))
        if len(frontiers.points) > 0:
            if not frontiers.has_colors():
                frontiers.paint_uniform_color([1.0, 0.35, 0.0])
            geometries.append(frontiers)

    if centroid_path.exists():
        centroids = o3d.io.read_triangle_mesh(str(centroid_path))
        if len(centroids.vertices) > 0 and len(centroids.triangles) > 0:
            centroids.compute_vertex_normals()
            geometries.append(centroids)

    if viewpoint_path.exists():
        viewpoints = o3d.io.read_point_cloud(str(viewpoint_path))
        if len(viewpoints.points) > 0:
            if not viewpoints.has_colors():
                viewpoints.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(viewpoints)


    if heading_path.exists():
        headings = o3d.io.read_line_set(str(heading_path))
        if len(headings.points) > 0 and len(headings.lines) > 0:
            geometries.append(headings)

    if reference_path.exists():
        reference = o3d.io.read_line_set(str(reference_path))
        if len(reference.points) > 0 and len(reference.lines) > 0:
            if not reference.has_colors():
                reference.paint_uniform_color([1.0, 0.05, 0.0])
            geometries.append(reference)

    if kino_path.exists():
        kino = o3d.io.read_line_set(str(kino_path))
        if len(kino.points) > 0 and len(kino.lines) > 0:
            if not kino.has_colors():
                kino.paint_uniform_color([1.0, 0.55, 0.0])
            geometries.append(kino)

    print(f"Loaded mesh vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
    o3d.visualization.draw_geometries(geometries, window_name=f"TSDF step {step}")


if __name__ == "__main__":
    main()

