"""Generate frontend/china-map.js from the downloaded China GeoJSON."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = ROOT / "china_provinces.geojson"
OUTPUT_PATH = ROOT / "frontend" / "china-map.js"

LAT_MIN = 17.5
LAT_MAX = 53.563269
LON_MIN = 73.502355
LON_MAX = 135.09567

VIEW_W = 1000.0
VIEW_H = 720.0
PAD_X = 30.0
PAD_Y = 24.0


def build_projection():
    mid_lat = (LAT_MIN + LAT_MAX) / 2
    k = math.cos(math.radians(mid_lat))
    x_range = (LON_MAX - LON_MIN) * k
    y_range = LAT_MAX - LAT_MIN
    scale = min((VIEW_W - 2 * PAD_X) / x_range, (VIEW_H - 2 * PAD_Y) / y_range)
    pad_y = (VIEW_H - (y_range * scale)) / 2

    def project(coord):
        x = PAD_X + (coord[0] - LON_MIN) * k * scale
        y = pad_y + (LAT_MAX - coord[1]) * scale
        return (x, y)

    return project


def rdp(points, eps):
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = math.hypot(dx, dy)
    max_dist = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        p = points[i]
        if denom == 0:
            dist = math.hypot(p[0] - start[0], p[1] - start[1])
        else:
            dist = abs(dy * p[0] - dx * p[1] + end[0] * start[1] - end[1] * start[0]) / denom
        if dist > max_dist:
            max_dist = dist
            index = i
    if max_dist > eps:
        left = rdp(points[: index + 1], eps)
        right = rdp(points[index:], eps)
        return left[:-1] + right
    return [start, end]


def simplify_ring(points, eps):
    if len(points) < 4:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diag < 4:
        return points
    return rdp(points, min(0.5, diag * 0.025))


def round_coord(value):
    value = round(value, 1)
    return str(int(value)) if value.is_integer() else str(value)


def path_from_coords(coords):
    if len(coords) < 3:
        return ""
    x, y = coords[0]
    parts = [f"M{round_coord(x)} {round_coord(y)}"]
    for x, y in coords[1:]:
        parts.append(f"L{round_coord(x)} {round_coord(y)}")
    parts.append("Z")
    return "".join(parts)


def make_feature_data(project):
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    featured = {
        "北京市",
        "浙江省",
        "四川省",
        "陕西省",
        "重庆市",
        "广东省",
        "江苏省",
        "湖南省",
    }
    records = []
    for feature in geojson["features"]:
        name = (feature["properties"].get("name") or "").strip()
        if not name:
            continue
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        subpaths = []
        for polygon in polygons:
            for ring in polygon:
                if not ring:
                    continue
                lats = [coord[1] for coord in ring]
                if max(lats) < LAT_MIN or min(lats) < LAT_MIN:
                    continue
                projected = [project(coord) for coord in ring]
                simplified = simplify_ring(projected, 0.5)
                subpath = path_from_coords(simplified)
                if subpath:
                    subpaths.append(subpath)
        if not subpaths:
            continue
        records.append(
            {
                "name": name,
                "short": short_name(name),
                "path": " ".join(subpaths),
                "covered": name in featured,
            }
        )
    return records


def short_name(name):
    replacements = {
        "北京市": "北京",
        "天津市": "天津",
        "上海市": "上海",
        "重庆市": "重庆",
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "西藏自治区": "西藏",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
        "香港特别行政区": "香港",
        "澳门特别行政区": "澳门",
    }
    if name in replacements:
        return replacements[name]
    return name.rstrip("省")


def main():
    project = build_projection()
    records = make_feature_data(project)
    records.sort(key=lambda item: item["name"])
    escaped = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    output = (
        "/* Auto-generated from china_provinces.geojson; keep in sync with tools/generate_china_map.py */\n"
        f"window.CHINA_PROVINCES = {escaped};\n"
    )
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(records)} provinces")


if __name__ == "__main__":
    main()
