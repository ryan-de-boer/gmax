#!/usr/bin/env python3
"""Probe Gmax box fields/transforms and export boxes or Editable Mesh objects."""

from __future__ import annotations

import struct
import re
import sys
import math
from pathlib import Path


CFB_SIGNATURE = bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1")
COMMON_HEADER = bytes.fromhex("DD DB DB DB F5 A4 D1 DB DB DB F1 A5")
COMMON_TAIL = bytes.fromhex("FB DB DB 5B F3 A4 EF E5 E5 E5 F2 A4")
AXIS_PREFIXES = (
    bytes.fromhex("E5 E5 E5 E5") + COMMON_HEADER,  # Width
    bytes.fromhex("E4 E5 E5 E5") + COMMON_HEADER,  # Length
    bytes.fromhex("E7 E5 E5 E5") + COMMON_HEADER,  # Height
)
POSITION_MARKER = bytes.fromhex("F7 E5 E5 E5 F2 81")
ROTATION_MARKER = bytes.fromhex("CD DB DB DB F5 81")
SCALE_MARKER = bytes.fromhex("C7 E5 E5 E5 F4 81")
TEXTURE_PATH_RE = re.compile(
    r"([A-Za-z]:\\[^\x00\r\n<>:\"|?*]{1,512}?\.(?:bmp|dib|jpg|jpeg|png|tif|tiff|tga|dds|gif))",
    re.IGNORECASE,
)


def read_transform_sidecar(path: Path) -> list[dict[str, object]]:
    sidecar = Path(str(path) + ".transforms.tsv")
    if not sidecar.exists():
        return []
    records: list[dict[str, object]] = []
    with sidecar.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split()
        if len(header) != 16:
            print(f"Warning: ignoring malformed transform sidecar: {sidecar}")
            return []
        for line in file:
            row = line.strip().split()
            if len(row) != 16:
                continue
            try:
                records.append({
                    "name": row[0],
                    "dimensions": tuple(float(v) for v in row[1:4]),
                    "matrix": tuple(float(v) for v in row[4:16]),
                })
            except ValueError:
                continue
    return records


def format_matrix(values: tuple[float, ...]) -> str:
    rows = [values[i : i + 3] for i in range(0, 12, 3)]
    return "(matrix3 " + " ".join(
        "[" + ",".join(f"{value:.9g}" for value in row) + "]"
        for row in rows
    ) + ")"


def find_records(data: bytes, marker: bytes, count: int, value_offset: int,
                 value_count: int, xor_mask: int | None = None) -> list[tuple[float, ...]]:
    """Read repeated transform records found by marker in observed Gmax saves."""
    offsets: list[int] = []
    cursor = 0
    while (found := data.find(marker, cursor)) >= 0:
        offsets.append(found + value_offset)
        cursor = found + len(marker)
    if len(offsets) != count:
        return []

    records = []
    for offset in offsets:
        raw = data[offset : offset + 4 * value_count]
        if len(raw) != 4 * value_count:
            return []
        if xor_mask is not None:
            raw = bytes(value ^ xor_mask for value in raw)
        values = struct.unpack("<" + "f" * value_count, raw)
        if not all(abs(value) < 1.0e9 for value in values):
            return []
        records.append(values)
    return records


def raw_matrices(data: bytes, count: int) -> list[tuple[float, ...]]:
    # The position controller stores XOR-E5 encoded float32s. The observed
    # rotation quaternion and scale controller floats are ordinary LE float32.
    positions = find_records(data, POSITION_MARKER, count, 6, 3, 0xE5)
    rotations = find_records(data, ROTATION_MARKER, count, 6, 4)
    scales = find_records(data, SCALE_MARKER, count, 6, 3)
    if not (positions and rotations and scales):
        return []

    matrices: list[tuple[float, ...]] = []
    for (x, y, z), (qx, qy, qz, qw), (sx, sy, sz) in zip(positions, rotations, scales):
        # Gmax's matrix3 basis is row-oriented. This quaternion convention
        # reproduces the serialized matrix basis in the rotated/scaled control.
        rows = (
            (1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)),
            (2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)),
            (2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)),
        )
        basis = tuple(value * scale for row, scale in zip(rows, (sx, sy, sz)) for value in row)
        matrices.append(basis + (x, y, z))
    return matrices


def texture_paths(data: bytes) -> list[str]:
    """Find absolute bitmap paths stored as UTF-16LE in the CFB payload."""
    found: list[tuple[int, str]] = []
    for alignment in (0, 1):
        text = data[alignment:].decode("utf-16le", errors="ignore")
        for match in TEXTURE_PATH_RE.finditer(text):
            raw_offset = alignment + match.start(1) * 2
            found.append((raw_offset, match.group(1)))
    # Alignment scans can find the same path twice; preserve on-disk order.
    paths: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in sorted(found):
        key = (item[0], item[1].casefold())
        if key not in seen:
            seen.add(key)
            paths.append(item)
    return [value for _, value in paths]


def ordered_textures(group_count: int, mesh_count: int, maps: list[str]) -> list[str]:
    """Pair discovered bitmap paths with exporter order (boxes, then meshes).

    In mixed saves, bitmap and transform records are serialized mesh-first,
    then box-first-to-last. Box dimension records are stored separately. In
    box-only saves, the observed bitmap order is the reverse of box records.
    """
    total = group_count + mesh_count
    if len(maps) != total:
        return [""] * total
    if mesh_count:
        return maps[mesh_count:] + maps[:mesh_count]
    return list(reversed(maps))


def mesh_material_maps(material_ids: list[int], paths: list[str]) -> dict[int, dict[str, str]]:
    """Infer diffuse/opacity bitmaps by material slot and serialized path order.

    The observed Gmax tree stores the MultiMaterial's diffuse maps in slot
    order and each opacity map immediately after that slot's diffuse bitmap.
    Opacity filenames are identified by common alpha/mask naming.
    """
    ids = sorted(set(material_ids))
    result = {material_id: {} for material_id in ids}
    if not ids or not paths:
        return result

    def is_opacity(path: str) -> bool:
        stem = Path(path.replace("\\", "/")).stem.casefold()
        return any(token in stem for token in ("opacity", "alpha", "mask"))

    diffuse_paths = [(index, path) for index, path in enumerate(paths) if not is_opacity(path)]
    path_material: dict[int, int] = {}
    for material_id, (path_index, path) in zip(ids, diffuse_paths):
        result[material_id]["map_Kd"] = path
        path_material[path_index] = material_id

    for path_index, path in enumerate(paths):
        if not is_opacity(path):
            continue
        preceding = [entry for entry in path_material.items() if entry[0] < path_index]
        following = [entry for entry in path_material.items() if entry[0] > path_index]
        candidate = max(preceding, default=None) or min(following, default=None)
        if candidate is not None:
            result[candidate[1]]["map_d"] = path
    return result


def opacity_png_with_alpha(source: str, obj_path: Path, material_name: str) -> str:
    """Return an opacity map path with an alpha channel, creating a PNG if needed.

    Blender reads the opacity channel from alpha when importing the OBJ MTL.
    For RGB-only opacity images, preserve the color image and place its
    grayscale intensity in alpha so both common MTL readers can use it.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Warning: Pillow is unavailable; cannot add alpha to opacity map "
              f"{source}. Install Pillow to convert it.", file=sys.stderr)
        return source

    source_path = Path(source)
    try:
        with Image.open(source_path) as image:
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            if has_alpha:
                return source
            color = image.convert("RGB")
            alpha = image.convert("L")
            output = obj_path.with_name(
                f"{obj_path.stem}_{material_name}_opacity.png"
            )
            rgba = color.convert("RGBA")
            rgba.putalpha(alpha)
            rgba.save(output, format="PNG")
            return output.name
    except Exception as exc:
        print(f"Warning: could not create alpha PNG for opacity map {source}: {exc}",
              file=sys.stderr)
        return source


def cfb_streams(data: bytes) -> dict[str, bytes]:
    """Return streams from a standard 512-byte-sector CFB/Gmax file."""
    if not data.startswith(CFB_SIGNATURE) or len(data) < 512:
        return {}
    sector_size = 1 << struct.unpack_from("<H", data, 30)[0]
    mini_size = 1 << struct.unpack_from("<H", data, 32)[0]

    def sector(index: int) -> bytes:
        start = 512 + index * sector_size
        return data[start : start + sector_size]

    free, end = 0xFFFFFFFF, 0xFFFFFFFE
    fat_ids = [value for value in struct.unpack_from("<109I", data, 76) if value not in (free, end)]
    difat_start, difat_count = struct.unpack_from("<II", data, 68)
    current = difat_start
    for _ in range(difat_count):
        values = struct.unpack("<" + "I" * (sector_size // 4), sector(current))
        fat_ids.extend(value for value in values[:-1] if value not in (free, end))
        current = values[-1]
    fat: list[int] = []
    for index in fat_ids:
        fat.extend(struct.unpack("<" + "I" * (sector_size // 4), sector(index)))

    def chain(start: int, table: list[int] = fat) -> list[int]:
        ids, seen = [], set()
        while start not in (free, end) and start < len(table) and start not in seen:
            seen.add(start)
            ids.append(start)
            start = table[start]
        return ids

    def big(start: int, size: int) -> bytes:
        return b"".join(sector(index) for index in chain(start))[:size]

    directory_start = struct.unpack_from("<I", data, 48)[0]
    directory = big(directory_start, 1_000_000)
    entries = []
    for offset in range(0, len(directory) - 127, 128):
        item = directory[offset : offset + 128]
        name_length, kind = struct.unpack_from("<H", item, 64)[0], item[66]
        if kind and name_length >= 2:
            name = item[: name_length - 2].decode("utf-16le", "replace")
            start = struct.unpack_from("<I", item, 116)[0]
            size = struct.unpack_from("<Q", item, 120)[0]
            entries.append((name, kind, start, size))
    root = next((item for item in entries if item[1] == 5), None)
    if root is None:
        return {}
    root_data = big(root[2], root[3])
    mini_fat_start, mini_fat_count = struct.unpack_from("<II", data, 60)
    mini_fat_data = big(mini_fat_start, mini_fat_count * sector_size)
    mini_fat = list(struct.unpack("<" + "I" * (len(mini_fat_data) // 4), mini_fat_data)) if mini_fat_data else []

    def mini(start: int, size: int) -> bytes:
        return b"".join(root_data[index * mini_size : (index + 1) * mini_size] for index in chain(start, mini_fat))[:size]

    cutoff = struct.unpack_from("<I", data, 56)[0]
    return {
        name: (mini(start, size) if size < cutoff else big(start, size))
        for name, kind, start, size in entries if kind == 2
    }


def editable_meshes(scene: bytes) -> list[dict[str, object]]:
    """Decode Gmax Editable Mesh vertex (0xADE5) and face (0xADE3) arrays.

    These two-byte IDs and record layouts are observed in the supplied Gmax
    save controls. Vertices are float32 XYZ; faces are five uint32 values,
    of which the first three are zero-based vertex indices.
    """
    vertex_blocks: list[tuple[int, list[tuple[float, float, float]]]] = []
    face_blocks: list[tuple[int, list[tuple[int, int, int]], list[int]]] = []
    for marker, is_vertex in ((bytes.fromhex("E5 AD"), True), (bytes.fromhex("E3 AD"), False)):
        cursor = 0
        while (offset := scene.find(marker, cursor)) >= 0:
            cursor = offset + 2
            if offset + 6 > len(scene):
                continue
            count = struct.unpack_from("<I", scene, offset + 2)[0]
            stride = 12 if is_vertex else 20
            if count == 0 or count > 2_000_000 or offset + 6 + count * stride > len(scene):
                continue
            if is_vertex:
                points = [struct.unpack_from("<fff", scene, offset + 6 + i * 12) for i in range(count)]
                if not all(all(abs(value) < 1.0e7 for value in point) for point in points):
                    continue
                vertex_blocks.append((offset, points))
            else:
                faces = []
                material_ids = []
                for i in range(count):
                    a, b, c, _smooth, flags = struct.unpack_from("<IIIII", scene, offset + 6 + i * 20)
                    faces.append((a, b, c))
                    # In observed Gmax editable-mesh saves the material slot
                    # is stored in the upper 16 bits of the fifth face word,
                    # with slot 1 encoded as zero and slot 2 as 0x00010000.
                    material_ids.append(((flags >> 16) & 0xFFFF) + 1)
                face_blocks.append((offset, faces, material_ids))

    meshes = []
    for vertex_offset, vertices in vertex_blocks:
        candidates = [(offset, faces, material_ids) for offset, faces, material_ids in face_blocks if offset > vertex_offset]
        if not candidates:
            continue
        face_offset, faces, material_ids = min(candidates, key=lambda item: item[0])
        valid_indices = [i for i, face in enumerate(faces) if max(face) < len(vertices)]
        valid = [faces[i] for i in valid_indices]
        valid_material_ids = [material_ids[i] for i in valid_indices]
        if valid:
            mesh: dict[str, object] = {"vertices": vertices, "faces": valid, "material_ids": valid_material_ids, "vertex_offset": vertex_offset, "face_offset": face_offset}
            # Editable Mesh map channel 1 follows the geometry block after a
            # small version-dependent header. Search only that nearby header
            # for a UV count, then validate a matching map-face count directly
            # after the UVW array. Avoid treating arbitrary E7 bytes elsewhere
            # in a large Scene stream as UV records.
            geometry_end = face_offset + 6 + len(faces) * 20
            uv_header_limit = min(len(scene) - 4, geometry_end + 4096)
            for uv_count_offset in range(geometry_end, uv_header_limit + 1):
                uv_count = struct.unpack_from("<I", scene, uv_count_offset)[0]
                if not 1 <= uv_count <= 2_000_000:
                    continue
                uv_start = uv_count_offset + 4
                uv_end = uv_start + uv_count * 12
                map_face_limit = min(len(scene), uv_end + 256)
                map_face_offset = scene.find(struct.pack("<I", len(faces)), uv_end, map_face_limit)
                if map_face_offset < 0 or map_face_offset + 4 + len(faces) * 12 > len(scene):
                    continue
                uv_sample_indices = (0, uv_count // 2, uv_count - 1)
                if not all(
                    all(math.isfinite(value) and abs(value) < 1.0e7
                        for value in struct.unpack_from("<fff", scene, uv_start + i * 12))
                    for i in uv_sample_indices
                ):
                    continue
                uvs = [struct.unpack_from("<fff", scene, uv_start + i * 12) for i in range(uv_count)]
                if not all(all(math.isfinite(value) and abs(value) < 1.0e7 for value in uv) for uv in uvs):
                    continue
                map_faces = [struct.unpack_from("<III", scene, map_face_offset + 4 + i * 12) for i in range(len(faces))]
                if not all(max(face) < uv_count for face in map_faces):
                    continue
                mesh.update({"uvs": uvs, "uv_faces": [map_faces[i] for i in valid_indices], "uv_offset": uv_count_offset, "uv_face_offset": map_face_offset})
                break
            meshes.append(mesh)
    return meshes


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale_vec(v: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def rotate_box_vertical_90(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Trial quarter-turn of box mesh geometry around its local vertical Z."""
    x, y, z = point
    return (-y, x, z)


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
    if length == 0:
        return (0.0, 0.0, 0.0)
    return scale_vec(v, 1.0 / length)


class BufferedLineWriter:
    """Write generated OBJ lines incrementally instead of retaining the whole file."""

    def __init__(self, path: Path, buffer_size: int = 16384) -> None:
        self.file = path.open("w", encoding="utf-8", newline="\n")
        self.buffer_size = buffer_size
        self.buffer: list[str] = []

    def append(self, line: str) -> None:
        self.buffer.append(line)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def extend(self, lines) -> None:
        for line in lines:
            self.append(line)

    def flush(self) -> None:
        if self.buffer:
            self.file.write("\n".join(self.buffer) + "\n")
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.file.close()


def transform_point(point: tuple[float, float, float], matrix: tuple[float, ...]) -> tuple[float, float, float]:
    # MAXScript matrix3 stores the transformed local X/Y/Z axes as its first
    # three rows and translation as its fourth row.
    x, y, z = point
    transformed = tuple(
        x * matrix[axis] + y * matrix[3 + axis] + z * matrix[6 + axis] + matrix[9 + axis]
        for axis in range(3)
    )
    # Apply the same Gmax-to-OBJ basis conversion to boxes and editable meshes:
    # swap Y/Z, then negate the new Z (the original Gmax Y coordinate).
    return (transformed[0], transformed[2], -transformed[1])


def transform_direction(direction: tuple[float, float, float], matrix: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z = direction
    transformed = tuple(
        x * matrix[axis] + y * matrix[3 + axis] + z * matrix[6 + axis]
        for axis in range(3)
    )
    return (transformed[0], transformed[2], -transformed[1])


def box_faces(width: float, length: float, height: float,
              width_segs: int, length_segs: int, height_segs: int):
    """Yield segmented outward-facing quads with normalized face UVs.

    Gmax/3ds Max box width runs along local X, length along local Y, and
    height along local Z. The pivot is at the center of the bottom face, so
    local Z spans 0..height. Separate vertices preserve hard normals/seams.
    """
    hx, hy = width / 2, length / 2
    faces = (
        # origin, edge-U, edge-V, U segments, V segments
        (( hx, -hy, 0), (0, length, 0), (0, 0, height), length_segs, height_segs),
        ((-hx,  hy, 0), (0, -length, 0), (0, 0, height), length_segs, height_segs),
        (( hx,  hy, 0), (-width, 0, 0), (0, 0, height), width_segs, height_segs),
        ((-hx, -hy, 0), (width, 0, 0), (0, 0, height), width_segs, height_segs),
        ((-hx, -hy, height), (width, 0, 0), (0, length, 0), width_segs, length_segs),
        ((-hx,  hy, 0), (width, 0, 0), (0, -length, 0), width_segs, length_segs),
    )
    for origin, edge_u, edge_v, u_count, v_count in faces:
        for v_index in range(v_count):
            v0, v1 = v_index / v_count, (v_index + 1) / v_count
            for u_index in range(u_count):
                u0, u1 = u_index / u_count, (u_index + 1) / u_count
                points = tuple(
                    add(add(origin, scale_vec(edge_u, u)), scale_vec(edge_v, v))
                    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
                )
                yield points, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))


def export_obj(path: Path, groups: list[dict[str, object]], matrices: list[tuple[float, ...]],
               maps: list[str], meshes: list[dict[str, object]] | None = None,
               mesh_matrices: list[tuple[float, ...]] | None = None) -> None:
    """Write generated box primitives and decoded Editable Mesh geometry."""
    meshes = meshes or []
    mesh_matrices = mesh_matrices or []
    if groups and len(matrices) != len(groups):
        raise ValueError("Cannot export boxes: raw transforms were not recognized for every box")
    if meshes and len(mesh_matrices) != len(meshes):
        raise ValueError("Cannot export editable meshes: raw transforms were not recognized for every mesh")
    textures = ordered_textures(len(groups), len(meshes), maps)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    obj_lines = BufferedLineWriter(path)
    obj_lines.extend(("# Geometry exported from a Gmax scene", f"mtllib {mtl_path.name}", ""))
    mtl_lines: list[str] = []
    vertex_index = texture_index = normal_index = 1

    for box_index, (group, matrix, texture) in enumerate(zip(groups, matrices, textures), start=1):
        width, length, height = group["dims"]
        segs = group["segs"] or {"width": 1, "length": 1, "height": 1}
        object_name = f"Box{box_index:02d}"
        material_name = f"{object_name}_Material"
        obj_lines.extend((f"o {object_name}", f"usemtl {material_name}"))
        mtl_lines.extend((f"newmtl {material_name}", "Kd 1.0 1.0 1.0"))
        if texture:
            mtl_lines.append(f"map_Kd {texture}")
        mtl_lines.append("")

        for local_points, uvs in box_faces(
            width, length, height,
            int(segs["width"]), int(segs["length"]), int(segs["height"]),
        ):
            # Trial a quarter-turn about vertical on the generated editable
            # box mesh around its bottom pivot, then apply the node transform
            # and OBJ axis conversion.
            turned_points = [rotate_box_vertical_90(point) for point in local_points]
            points = [transform_point(point, matrix) for point in turned_points]
            # Transform the normal with the cross product of transformed face
            # edges, which handles non-uniform scale and mirrored transforms.
            world_normal = normalize(cross(
                transform_direction((turned_points[1][0] - turned_points[0][0], turned_points[1][1] - turned_points[0][1], turned_points[1][2] - turned_points[0][2]), matrix),
                transform_direction((turned_points[3][0] - turned_points[0][0], turned_points[3][1] - turned_points[0][1], turned_points[3][2] - turned_points[0][2]), matrix),
            ))
            for point in points:
                obj_lines.append("v " + " ".join(f"{value:.9g}" for value in point))
            for uv in uvs:
                obj_lines.append(f"vt {uv[0]:.9g} {uv[1]:.9g}")
            obj_lines.append("vn " + " ".join(f"{value:.9g}" for value in world_normal))
            refs = [
                f"{vertex_index + corner}/{texture_index + corner}/{normal_index}"
                for corner in range(4)
            ]
            obj_lines.append("f " + " ".join(refs))
            vertex_index += 4
            texture_index += 4
            normal_index += 1
        obj_lines.append("")

    for mesh_index, (mesh, matrix) in enumerate(zip(meshes, mesh_matrices), start=1):
        base_name = f"EditableMesh{mesh_index:02d}"
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        uvs = mesh.get("uvs", [])
        uv_faces = mesh.get("uv_faces", [])
        face_material_ids = mesh.get("material_ids", [1] * len(faces))
        material_ids = sorted(set(face_material_ids)) or [1]
        world_points = [transform_point(point, matrix) for point in vertices]

        if len(maps) == len(groups) + len(meshes) and len(material_ids) == 1:
            texture = textures[len(groups) + mesh_index - 1]
            material_maps = {material_ids[0]: ({"map_Kd": texture} if texture else {})}
        elif not groups and len(meshes) == 1:
            material_maps = mesh_material_maps(face_material_ids, maps)
        else:
            material_maps = {material_id: {} for material_id in material_ids}

        for material_id in material_ids:
            face_numbers = [i for i, value in enumerate(face_material_ids) if value == material_id]
            if not face_numbers:
                continue
            suffix = f"_Mat{material_id:02d}" if len(material_ids) > 1 else ""
            object_name = base_name + suffix
            material_name = f"{base_name}_Material{material_id:02d}"
            obj_lines.extend((f"o {object_name}", f"usemtl {material_name}"))
            mtl_lines.extend((f"newmtl {material_name}", "Kd 1.0 1.0 1.0"))
            info = material_maps.get(material_id, {})
            if info.get("map_Kd"):
                mtl_lines.append(f"map_Kd {info['map_Kd']}")
            if info.get("map_d"):
                opacity_map = opacity_png_with_alpha(
                    info["map_d"], path, material_name
                )
                mtl_lines.append(f"map_d {opacity_map}")
            mtl_lines.append("")

            vertex_refs: dict[int, int] = {}
            uv_refs: dict[int, int] = {}
            for emitted_count, face_number in enumerate(face_numbers, start=1):
                if emitted_count % 100000 == 0:
                    print(f"  wrote {emitted_count:,} of {len(face_numbers):,} faces for {object_name}...", flush=True)
                face = faces[face_number]
                world_face = [world_points[index] for index in face]
                normal = normalize(cross(
                    tuple(world_face[1][i] - world_face[0][i] for i in range(3)),
                    tuple(world_face[2][i] - world_face[0][i] for i in range(3)),
                ))
                normal_ref = normal_index
                obj_lines.append("vn " + " ".join(f"{value:.9g}" for value in normal))
                normal_index += 1
                map_face = uv_faces[face_number] if face_number < len(uv_faces) else None
                refs = []
                for corner, vertex_id in enumerate(face):
                    if vertex_id not in vertex_refs:
                        vertex_refs[vertex_id] = vertex_index
                        obj_lines.append("v " + " ".join(f"{value:.9g}" for value in world_points[vertex_id]))
                        vertex_index += 1
                    vertex_ref = vertex_refs[vertex_id]
                    if map_face is not None and corner < len(map_face):
                        uv_id = map_face[corner]
                        if uv_id not in uv_refs:
                            uv_refs[uv_id] = texture_index
                            uv = uvs[uv_id]
                            obj_lines.append(f"vt {uv[0]:.9g} {uv[1]:.9g}")
                            texture_index += 1
                        refs.append(f"{vertex_ref}/{uv_refs[uv_id]}/{normal_ref}")
                    else:
                        refs.append(f"{vertex_ref}//{normal_ref}")
                obj_lines.append("f " + " ".join(refs))
            obj_lines.append("")

    obj_lines.close()
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")


def probe(path: Path, obj_out: Path | None = None) -> None:
    data = path.read_bytes()
    print(f"File: {path}", flush=True)
    print(f"Bytes: {len(data)}")
    print(f"Container: {'OLE Compound File (CFB signature)' if data.startswith(CFB_SIGNATURE) else 'unknown'}")

    groups: list[dict[str, object]] = []
    cursor = 0
    while (start := data.find(AXIS_PREFIXES[0], cursor)) >= 0:
        cursor = start + 1
        if any(data[start + i * 32 : start + i * 32 + 16] != prefix
               for i, prefix in enumerate(AXIS_PREFIXES)):
            continue
        if any(data[start + i * 32 + 20 : start + i * 32 + 32] != COMMON_TAIL
               for i in range(3)):
            continue

        dims = [struct.unpack_from("<f", data, start + 16 + i * 32)[0]
                for i in range(3)]
        # Gmax stores dimensions and segment counts in Width, Length, Height
        # order, matching the Box primitive's local X, Y, Z axes.
        raw_segs = [struct.unpack_from("<i", data, start + 112 + i * 32)[0]
                    for i in range(3)]
        if not all(0.0 < value < 1.0e7 for value in dims):
            continue
        segs = {"width": raw_segs[0], "length": raw_segs[1], "height": raw_segs[2]}
        if not all(1 <= value <= 10000 for value in segs.values()):
            segs = None

        offsets = [start + 16 + i * 32 for i in range(3)]
        groups.append({"dims": tuple(dims), "segs": segs, "offsets": offsets})

    print("Reading scene streams and editable meshes...", flush=True)
    mesh_records = editable_meshes(cfb_streams(data).get("Scene", b""))
    print(f"  found {len(mesh_records)} editable mesh(es)", flush=True)
    if not groups and not mesh_records:
        print("No matching box dimensions or supported Editable Mesh records found.")
        return

    all_matrices = raw_matrices(data, len(groups) + len(mesh_records)) if groups or mesh_records else []
    # In the observed mixed scene, transform-controller records follow Gmax's
    # serialized node order (editable meshes first, then parametric boxes),
    # while the box dimension records are encountered before mesh reporting.
    # Splitting by object type in the opposite order assigned each box another
    # node's translation. Keep this ordering explicit until node names can be
    # decoded from the scene stream.
    mesh_matrices = all_matrices[:len(mesh_records)] if mesh_records else []
    matrices = all_matrices[len(mesh_records):len(mesh_records) + len(groups)] if groups else []
    maps = texture_paths(data)
    transforms = read_transform_sidecar(path) if not matrices else []
    export_matrices = matrices or [record["matrix"] for record in transforms]

    unused = set(range(len(transforms)))
    for index, group in enumerate(groups, start=1):
        dims = group["dims"]
        offsets = group["offsets"]
        segs = group["segs"]
        print(f"\nCandidate box field group {index}:")
        print("  dimension offsets: " + ", ".join(f"0x{x:X}" for x in offsets))
        print(f"  width={dims[0]:.9g}, length={dims[1]:.9g}, height={dims[2]:.9g}")
        if segs is not None:
            print(f"  segments: length={segs['length']}, width={segs['width']}, height={segs['height']}")
        else:
            print("  segment fields: not recognized")

        total_objects = len(groups) + len(mesh_records)
        if len(maps) == total_objects:
            print(f"  diffuse texture: {ordered_textures(len(groups), len(mesh_records), maps)[index - 1]}")
        else:
            print("  diffuse texture: unavailable (bitmap-to-box mapping is ambiguous)")

        if matrices:
            print(f"  transform: {format_matrix(matrices[index - 1])}")
            continue

        match = None
        for record_index in list(unused):
            candidate_dims = transforms[record_index]["dimensions"]
            if all(abs(a - b) < 1e-4 for a, b in zip(dims, candidate_dims)):
                match = record_index
                break
        if match is None and len(transforms) == len(groups) and index - 1 in unused:
            # Equal counts allow a useful fallback when Gmax rounds its
            # MAXScript dimensions differently from the serialized floats.
            match = index - 1
        if match is None:
            print("  transform: unavailable (raw transform markers not recognized)")
            continue

        unused.remove(match)
        record = transforms[match]
        print(f"  object: {record['name']}")
        print(f"  transform: {format_matrix(record['matrix'])}")

    if not matrices and not transforms and groups:
        print("\nNo raw transform set or transform sidecar recognized for this save layout.")
    can_pair_mesh_materials = not groups and len(mesh_records) == 1 and bool(maps)
    if maps and len(maps) != len(groups) + len(mesh_records) and not can_pair_mesh_materials:
        print("\nBitmap paths found in file (not paired to objects):")
        for texture in maps:
            print(f"  {texture}")

    if obj_out is not None:
        if groups and not matrices and transforms:
            export_matrices = [record["matrix"] for record in transforms]
        if mesh_records and not mesh_matrices:
            print("\nWarning: editable mesh transform markers were not found; exporting meshes in local coordinates.")
            mesh_matrices = [tuple((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))] * len(mesh_records)
        print("Writing OBJ geometry...", flush=True)
        export_obj(obj_out, groups, export_matrices, maps, mesh_records, mesh_matrices)
        print(f"\nOBJ exported: {obj_out}")
        print(f"MTL exported: {obj_out.with_suffix('.mtl')}")

    if mesh_records:
        print(f"\nEditable meshes found: {len(mesh_records)}")
        for index, mesh in enumerate(mesh_records, start=1):
            print(f"  EditableMesh{index:02d}: {len(mesh['vertices'])} vertices, {len(mesh['faces'])} triangular faces")
            if mesh.get("uvs"):
                print(f"    texture coordinates: {len(mesh['uvs'])} vertices, {len(mesh['uv_faces'])} mapped faces")
            else:
                print("    texture coordinates: unavailable (map channel not recognized)")
            if len(maps) == len(groups) + len(mesh_records):
                print(f"    diffuse texture: {ordered_textures(len(groups), len(mesh_records), maps)[len(groups) + index - 1]}")
            elif can_pair_mesh_materials:
                assignments = mesh_material_maps(mesh.get("material_ids", []), maps)
                for material_id, info in sorted(assignments.items()):
                    print(f"    material slot {material_id}:")
                    if info.get("map_Kd"):
                        print(f"      diffuse texture: {info['map_Kd']}")
                    if info.get("map_d"):
                        print(f"      opacity texture: {info['map_d']}")
            if len(mesh_matrices) == len(mesh_records):
                print(f"    transform: {format_matrix(mesh_matrices[index - 1])}")
            else:
                print("    transform: unavailable (raw transform markers not recognized)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--export-obj"):
        raise SystemExit(
            f"Usage: {Path(sys.argv[0]).name} path/to/file.gmax [--export-obj path/to/output.obj]"
        )
    output = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    probe(Path(sys.argv[1]), output)
