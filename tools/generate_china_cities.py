"""Generate frontend/china-cities.js from DataV city-level GeoJSON.

Downloads one `{adcode}_full.json` per province-level feature and keeps only
`level == "city"` polygons. Municipalities and special administrative regions
have no prefecture city children, so they are represented by their province
boundary instead.
"""

import json
import math
import ssl
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVINCE_GEOJSON = ROOT / "china_provinces.geojson"
OUTPUT_PATH = ROOT / "frontend" / "china-cities.js"
BASE_URL = "https://geo.datav.aliyun.com/areas_v3/bound"

LAT_MIN = 17.5
LAT_MAX = 53.563269
LON_MIN = 73.502355
LON_MAX = 135.09567

VIEW_W = 1000.0
VIEW_H = 720.0
PAD_X = 30.0
PAD_Y = 24.0

ctx = ssl.create_default_context()


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
            dist = (
                abs(dy * p[0] - dx * p[1] + end[0] * start[1] - end[1] * start[0])
                / denom
            )
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
    return rdp(points, min(0.35, diag * 0.02))


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


def geometry_path(geometry, project):
    if geometry is None:
        return ""
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    subpaths = []
    for polygon in polygons:
        for ring in polygon:
            if not ring:
                continue
            projected = [project(coord) for coord in ring]
            simplified = simplify_ring(projected, 0.35)
            subpath = path_from_coords(simplified)
            if subpath:
                subpaths.append(subpath)
    return " ".join(subpaths)


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
    for suffix in ("特别行政区", "维吾尔自治区", "壮族自治区", "回族自治区", "自治区"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.rstrip("省")


def city_short(name):
    # A forgiving short name for matching user-entered city text. Handles
    # autonomous prefectures by dropping the ethnic qualifier and ending at the
    # leading locality token, e.g. "凉山彝族自治州" -> "凉山".
    if name.endswith("自治州"):
        ethnic_starts = set("维回藏彝苗侗布壮傣景傈哈土朝蒙柯白")
        for i, ch in enumerate(name):
            if ch in ethnic_starts:
                return name[:i]
    for suffix in ("特别行政区", "自治区", "地区", "盟", "市", "州"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def download_json(adcode):
    url = f"{BASE_URL}/{adcode}_full.json"
    try:
        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(f"  skip download {adcode}: {exc}")
        return None


def center_point(props, project):
    coord = props.get("center") or props.get("centroid")
    if not coord:
        geometry = props.get("_geometry")
        if geometry:
            rings = []
            polygons = (
                [geometry["coordinates"]]
                if geometry["type"] == "Polygon"
                else geometry["coordinates"]
            )
            for polygon in polygons:
                for ring in polygon:
                    rings.extend(ring)
            if rings:
                lon = sum(c[0] for c in rings) / len(rings)
                lat = sum(c[1] for c in rings) / len(rings)
                coord = [lon, lat]
    if not coord:
        return (VIEW_W / 2, VIEW_H / 2)
    x, y = project(coord)
    return (round(x, 1), round(y, 1))


def main():
    project = build_projection()
    provinces = json.loads(PROVINCE_GEOJSON.read_text(encoding="utf-8"))

    province_meta = []
    cities = []

    for feature in provinces["features"]:
        props = feature["properties"]
        name = (props.get("name") or "").strip()
        if not name:
            continue
        adcode = str(props.get("adcode") or "")
        province = {
            "name": name,
            "short": short_name(name),
            "adcode": adcode,
        }
        cx, cy = center_point(props, project)
        province["cx"] = cx
        province["cy"] = cy

        children = download_json(adcode) if adcode else None
        city_features = [
            child
            for child in (children or {}).get("features", [])
            if child.get("properties", {}).get("level") == "city"
        ]

        if city_features:
            province["hasCityRegions"] = True
            for child in city_features:
                cprops = child["properties"]
                city_name = (cprops.get("name") or "").strip()
                if not city_name:
                    continue
                geometry = child.get("geometry")
                path = geometry_path(geometry, project)
                if not path:
                    continue
                ccx, ccy = center_point(cprops, project)
                cities.append(
                    {
                        "name": city_name,
                        "short": city_short(city_name),
                        "province": name,
                        "path": path,
                        "cx": ccx,
                        "cy": ccy,
                        "adcode": str(cprops.get("adcode") or ""),
                    }
                )
        else:
            # Municipalities / SARs are both a province and a single city.
            province["hasCityRegions"] = False
            path = geometry_path(feature.get("geometry"), project)
            if path:
                cities.append(
                    {
                        "name": name,
                        "short": short_name(name),
                        "province": name,
                        "path": path,
                        "cx": cx,
                        "cy": cy,
                        "adcode": adcode,
                    }
                )

        province_meta.append(province)
        print(f"{name}: {len(city_features)} cities")

    province_meta.sort(key=lambda item: item["name"])
    cities.sort(key=lambda item: (item["province"], item["name"]))

    output = (
        "/* Auto-generated from DataV city GeoJSON; keep in sync with "
        "tools/generate_china_cities.py */\n"
        f"window.CHINA_PROVINCE_META = {json.dumps(province_meta, ensure_ascii=False, separators=(',', ':'))};\n"
        f"window.CHINA_CITIES = {json.dumps(cities, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(cities)} cities")


if __name__ == "__main__":
    main()
