#!/usr/bin/env python3
"""Convert a Wavefront OBJ into a native Gmax Editable Mesh scene.

Gmax's binary scene container is not written directly. This tool parses OBJ,
generates a temporary MAXScript that constructs editable meshes in Gmax, and
asks the installed Gmax executable to save the result as a .gmax file.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ObjFace:
    vertices: list[int]
    uvs: list[int | None]
    material: str


@dataclass
class ObjObject:
    name: str
    faces: list[ObjFace] = field(default_factory=list)


def parse_obj(path: Path):
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    objects: list[ObjObject] = []
    materials: dict[str, dict[str, str]] = {}
    material_libraries: list[Path] = []
    current_object = ObjObject("Object01")
    objects.append(current_object)
    current_material = "Material01"

    def resolve_index(value: str, count: int) -> int:
        index = int(value)
        result = index - 1 if index > 0 else count + index
        if result < 0 or result >= count:
            raise ValueError(f"OBJ index {index} is out of range")
        return result

    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            keyword, _, body = line.partition(" ")
            body = body.strip()
            try:
                if keyword == "v":
                    values = [float(value) for value in body.split()[:3]]
                    if len(values) != 3 or not all(math.isfinite(value) for value in values):
                        raise ValueError("expected three finite coordinates")
                    # The companion exporter maps Gmax (X,Y,Z) to OBJ
                    # (X,Z,-Y); apply the inverse so round-trips restore the
                    # original Gmax scene coordinates.
                    x, y, z = values
                    positions.append((x, -z, y))
                elif keyword == "vt":
                    values = [float(value) for value in body.split()]
                    if len(values) < 2 or not all(math.isfinite(value) for value in values[:2]):
                        raise ValueError("expected at least two finite UV coordinates")
                    texcoords.append((values[0], values[1]))
                elif keyword == "o" or keyword == "g":
                    name = body or f"Object{len(objects) + 1:02d}"
                    if current_object.faces:
                        current_object = ObjObject(name)
                        objects.append(current_object)
                    else:
                        current_object.name = name
                elif keyword == "usemtl":
                    current_material = body or "Material01"
                elif keyword == "mtllib":
                    material_libraries.extend(path.parent / value for value in body.split())
                elif keyword == "f":
                    corners = [token.split("/") for token in body.split()]
                    if len(corners) < 3:
                        raise ValueError("face has fewer than three corners")
                    v_indices, vt_indices = [], []
                    for corner in corners:
                        if not corner[0]:
                            raise ValueError("face corner is missing a vertex index")
                        v_indices.append(resolve_index(corner[0], len(positions)))
                        vt_indices.append(
                            resolve_index(corner[1], len(texcoords))
                            if len(corner) > 1 and corner[1] else None
                        )
                    # Fan triangulation preserves per-corner UV references.
                    for i in range(1, len(corners) - 1):
                        current_object.faces.append(ObjFace(
                            [v_indices[0], v_indices[i], v_indices[i + 1]],
                            [vt_indices[0], vt_indices[i], vt_indices[i + 1]],
                            current_material,
                        ))
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

    objects = [obj for obj in objects if obj.faces]
    if not objects:
        raise ValueError(f"No faces were found in {path}")
    if not positions:
        raise ValueError(f"No vertices were found in {path}")

    for library in material_libraries:
        if not library.exists():
            continue
        current = None
        with library.open("r", encoding="utf-8-sig", errors="replace") as source:
            for raw_line in source:
                line = raw_line.partition("#")[0].strip()
                if not line:
                    continue
                keyword, _, body = line.partition(" ")
                body = body.strip()
                if keyword == "newmtl":
                    current = body
                    materials.setdefault(current, {})
                elif current and keyword in ("map_Kd", "map_d"):
                    # In MTL, the filename is the remainder of the line and
                    # may contain spaces. Preserve plain Windows/relative paths
                    # whole; only peel off options when the line starts with
                    # an option token.
                    texture = body.strip()
                    quoted = re.match(r'^"([^"\r\n]+)"(?:\s+.*)?$', texture)
                    if quoted:
                        texture = quoted.group(1)
                    elif texture.startswith("-"):
                        # Common MTL options precede the filename. Prefer a
                        # drive-qualified path (which may itself contain spaces).
                        drive_path = re.search(r"(?<!\S)([A-Za-z]:[\\/].*)$", texture)
                        if drive_path:
                            texture = drive_path.group(1).strip().strip('"')
                        else:
                            match = re.search(
                                r"(?:^|\s)([^\s]+\.(?:bmp|dib|jpe?g|png|tiff?|tga|dds|gif))\s*$",
                                texture, re.I,
                            )
                            texture = match.group(1) if match else texture
                    texture_path = Path(texture)
                    if not texture_path.is_absolute():
                        texture_path = library.parent / texture_path
                    materials[current][keyword] = str(texture_path)
                elif current and keyword == "Kd":
                    values = body.split()
                    if len(values) >= 3:
                        materials[current]["Kd"] = " ".join(values[:3])
    return positions, texcoords, objects, materials


def ms_string(value: str | Path) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def ms_float(value: float) -> str:
    return format(value, ".9g")


def maxscript_source(obj_path: Path, output_path: Path, positions, texcoords,
                     objects: list[ObjObject], materials: dict[str, dict[str, str]],
                     status_path: Path | None = None,
                     data_dir: Path | None = None) -> str:
    lines = [
        "resetMaxFile #noPrompt",
    ]
    if data_dir is not None:
        lines.extend((
            "fn tgReadPoint3 inputFile = (",
            "    local fields = filterString (readLine inputFile) \",\"",
            "    [fields[1] as float, fields[2] as float, fields[3] as float]",
            ")",
        ))

    all_material_names = list(dict.fromkeys(face.material for obj in objects for face in obj.faces))
    safe_material_names = {name: f"M{index + 1}" for index, name in enumerate(all_material_names)}
    for name, safe_name in safe_material_names.items():
        info = materials.get(name, {})
        lines.append(f"{safe_name} = standardMaterial name:{ms_string(name)}")
        if "Kd" in info:
            vals = info["Kd"].split()
            lines.append(f"{safe_name}.diffuse = color {int(float(vals[0])*255)} {int(float(vals[1])*255)} {int(float(vals[2])*255)}")
        if "map_Kd" in info:
            lines.append(f"{safe_name}Map = bitmapTexture filename:{ms_string(info['map_Kd'])}")
            lines.append(f"{safe_name}.diffuseMap = {safe_name}Map")
            # Standard materials keep map slots disabled by default; setting
            # diffuseMap alone leaves only the diffuse color visible.
            lines.append(f"{safe_name}.diffuseMapEnable = true")
        if "map_d" in info:
            lines.append(f"{safe_name}OpacityMap = bitmapTexture filename:{ms_string(info['map_d'])}")
            lines.append(f"{safe_name}.opacityMap = {safe_name}OpacityMap")
            lines.append(f"{safe_name}.opacityMapEnable = true")

    for object_index, obj in enumerate(objects, start=1):
        name = re.sub(r"[^A-Za-z0-9_]", "_", obj.name) or f"Object{object_index:02d}"
        name = name[:60]
        has_complete_uvs = bool(texcoords) and all(
            all(index is not None for index in face.uvs) for face in obj.faces
        )
        local_vertices = []
        local_uvs = []
        faces = []
        vertex_map = {}
        for face in obj.faces:
            converted_face = []
            for corner, vertex_index in enumerate(face.vertices):
                uv_index = face.uvs[corner] if has_complete_uvs else None
                # A Gmax mesh constructor pairs each texture vertex with the
                # mesh vertex at the same index. Split shared OBJ vertices at
                # UV seams so constructor topology can preserve every corner UV.
                key = (vertex_index, uv_index)
                if key not in vertex_map:
                    vertex_map[key] = len(local_vertices) + 1
                    local_vertices.append(positions[vertex_index])
                    if has_complete_uvs:
                        local_uvs.append(texcoords[uv_index])
                converted_face.append(vertex_map[key])
            faces.append(converted_face)
        object_materials = list(dict.fromkeys(face.material for face in obj.faces))
        if data_dir is None:
            lines.append(f"meshVerts{object_index} = #(")
            lines.extend(
                f"    [{ms_float(x)},{ms_float(y)},{ms_float(z)}]" + ("," if i + 1 < len(local_vertices) else "")
                for i, (x, y, z) in enumerate(local_vertices)
            )
            lines.append(")")
            lines.append(f"meshFaces{object_index} = #(")
            lines.extend(
                f"    [{a},{b},{c}]" + ("," if i + 1 < len(faces) else "")
                for i, (a, b, c) in enumerate(faces)
            )
            lines.append(")")
            if local_uvs:
                lines.append(f"meshTVerts{object_index} = #(")
                lines.extend(
                    f"    [{ms_float(u)},{ms_float(v)},0]" + ("," if i + 1 < len(local_uvs) else "")
                    for i, (u, v) in enumerate(local_uvs)
                )
                lines.append(")")
                mesh_args = (
                    f"vertices:meshVerts{object_index} "
                    f"tverts:meshTVerts{object_index} faces:meshFaces{object_index}"
                )
            else:
                mesh_args = f"vertices:meshVerts{object_index} faces:meshFaces{object_index}"
            lines.append(f"node{object_index} = mesh name:{ms_string(name)} {mesh_args}")
            if local_uvs:
                # tverts stores UV coordinates; TVFaces associate each geometry
                # face corner with its UV vertex. Gmax's mesh constructor does not
                # reliably create this association from tverts alone.
                lines.append(f"buildTVFaces node{object_index}.mesh")
                lines.append(
                    f"for faceIndex = 1 to meshFaces{object_index}.count do "
                    f"setTVFace node{object_index}.mesh faceIndex meshFaces{object_index}[faceIndex]"
                )
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            vertex_path = data_dir / f"mesh_{object_index}_vertices.csv"
            face_path = data_dir / f"mesh_{object_index}_faces.csv"
            uv_path = data_dir / f"mesh_{object_index}_uvs.csv"
            with vertex_path.open("w", encoding="ascii", newline="\n") as data_file:
                for point in local_vertices:
                    data_file.write(",".join(ms_float(value) for value in point) + "\n")
            with face_path.open("w", encoding="ascii", newline="\n") as data_file:
                for face, obj_face in zip(faces, obj.faces):
                    material_id = object_materials.index(obj_face.material) + 1
                    data_file.write(",".join(str(value) for value in (*face, material_id)) + "\n")
            if local_uvs:
                with uv_path.open("w", encoding="ascii", newline="\n") as data_file:
                    for u, v in local_uvs:
                        data_file.write(f"{ms_float(u)},{ms_float(v)},0\n")

            lines.append(f"node{object_index} = mesh name:{ms_string(name)} numverts:{len(local_vertices)} numfaces:{len(faces)}")
            lines.append(f"vertexFile{object_index} = openFile {ms_string(vertex_path)}")
            lines.append(f"for i = 1 to {len(local_vertices)} do setVert node{object_index} i (tgReadPoint3 vertexFile{object_index})")
            lines.append(f"close vertexFile{object_index}")
            lines.append(f"faceFile{object_index} = openFile {ms_string(face_path)}")
            lines.append(f"for i = 1 to {len(faces)} do (")
            lines.append(f"    faceFields{object_index} = filterString (readLine faceFile{object_index}) \",\"")
            lines.append(f"    setFace node{object_index} i [faceFields{object_index}[1] as integer, faceFields{object_index}[2] as integer, faceFields{object_index}[3] as integer]")
            if len(object_materials) > 1:
                lines.append(f"    setFaceMatID node{object_index}.mesh i (faceFields{object_index}[4] as integer)")
            lines.append(")")
            lines.append(f"close faceFile{object_index}")
            if local_uvs:
                lines.append(f"setNumTVerts node{object_index}.mesh {len(local_uvs)}")
                lines.append(f"uvFile{object_index} = openFile {ms_string(uv_path)}")
                lines.append(f"for i = 1 to {len(local_uvs)} do setTVert node{object_index}.mesh i (tgReadPoint3 uvFile{object_index})")
                lines.append(f"close uvFile{object_index}")
                lines.append(f"buildTVFaces node{object_index}.mesh")
                lines.append(f"for i = 1 to {len(faces)} do setTVFace node{object_index}.mesh i (getFace node{object_index}.mesh i)")
            lines.append(f"update node{object_index}")
        if len(object_materials) == 1:
            # Most exported OBJ objects have one `usemtl`. Assign that
            # Standard material directly so viewport bitmap display does not
            # depend on a one-slot MultiMaterial wrapper.
            material_name = object_materials[0]
            material_var = safe_material_names[material_name]
            lines.append(f"node{object_index}.material = {material_var}")
            if any(slot in materials.get(material_name, {}) for slot in ("map_Kd", "map_d")):
                # Material-level display shows supported diffuse and opacity
                # maps together. OpenGL avoids the crash seen with the other
                # viewport driver on opacity-mapped materials.
                lines.append(f"showTextureMap {material_var} on")
        elif object_materials:
            lines.append(f"subMats{object_index} = #({', '.join(safe_material_names[n] for n in object_materials)})")
            lines.append(f"multiMat{object_index} = multiMaterial numsubs:subMats{object_index}.count")
            lines.append(f"for i = 1 to subMats{object_index}.count do multiMat{object_index}[i] = subMats{object_index}[i]")
            lines.append(f"node{object_index}.material = multiMat{object_index}")
            if data_dir is None:
                for face_index, face in enumerate(obj.faces, start=1):
                    material_id = object_materials.index(face.material) + 1
                    lines.append(f"setFaceMatID node{object_index}.mesh {face_index} {material_id}")
            for material_index, material_name in enumerate(object_materials, start=1):
                if any(slot in materials.get(material_name, {}) for slot in ("map_Kd", "map_d")):
                    lines.append(f"showTextureMap subMats{object_index}[{material_index}] on")
        # `mesh` constructs an Editable Mesh directly in Gmax; older Gmax
        # builds do not expose the later `convertToMesh` helper.

    # Gmax exposes the one-argument saveMaxFile form. Newer 3ds Max keywords
    # (useNewFile/quiet) and MAXScript file streams (createFile) are missing.
    lines.append("redrawViews()")
    lines.append(f"saveMaxFile {ms_string(output_path)}")
    return "\n".join(lines) + "\n"


def find_gmax(explicit: Path | None) -> Path:
    candidates = [explicit, Path(r"C:\gmax\gmax.exe"), Path(os.environ.get("GMAX_EXE", ""))]
    for candidate in candidates:
        if candidate and str(candidate) not in ("", ".") and candidate.is_file():
            return candidate
    raise FileNotFoundError("Gmax executable not found; use --gmax-exe or set GMAX_EXE")


def stop_gmax_process(process: subprocess.Popen) -> None:
    """Close the spawned Gmax process before its temporary script is removed."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        if process.poll() is not None:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def convert(obj_path: Path, output_path: Path, gmax_exe: Path | None,
            timeout: int = 180) -> None:
    obj_path = obj_path.resolve()
    output_path = output_path.resolve()
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file does not exist: {obj_path}")
    if output_path.suffix.lower() != ".gmax":
        raise ValueError("Output file must have a .gmax extension")
    positions, texcoords, objects, materials = parse_obj(obj_path)
    executable = find_gmax(gmax_exe)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="to_gmax_") as temp_dir:
        temp = Path(temp_dir)
        script_path = temp / "convert.ms"
        script_path.write_text(
            maxscript_source(obj_path, output_path, positions, texcoords, objects, materials, data_dir=temp),
            encoding="utf-8",
        )
        process = subprocess.Popen([str(executable), "-U", "MAXScript", str(script_path)])
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if output_path.is_file() and output_path.stat().st_size > 0:
                    break
                if process.poll() is not None and not output_path.is_file():
                    raise RuntimeError(f"Gmax exited with code {process.returncode} before saving the scene")
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Gmax did not finish within {timeout} seconds")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("Gmax did not create a non-empty output file")
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    stop_gmax_process(process)
            print(f"Created {output_path} ({output_path.stat().st_size} bytes)")
            print(f"Editable mesh objects: {len(objects)}; vertices: {len(positions)}; UVs: {len(texcoords)}")
            print(f"Texture materials: {sum('map_Kd' in materials.get(name, {}) for name in {face.material for obj in objects for face in obj.faces})}")
            print(f"Opacity texture materials: {sum('map_d' in materials.get(name, {}) for name in {face.material for obj in objects for face in obj.faces})}")
        finally:
            # TemporaryDirectory cleanup on Windows fails while Gmax still has
            # convert.ms or its CSV sidecars open, even if saveMaxFile succeeded.
            stop_gmax_process(process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path, help="Input Wavefront OBJ file")
    parser.add_argument("output", type=Path, help="Output .gmax file")
    parser.add_argument("--gmax-exe", type=Path, help="Path to gmax.exe (default: C:\\gmax\\gmax.exe or GMAX_EXE)")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for Gmax to save (default: 180)")
    args = parser.parse_args()
    try:
        convert(args.obj, args.output, args.gmax_exe, args.timeout)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
