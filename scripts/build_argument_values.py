#!/usr/bin/env python
"""
build_argument_values.py
─────────────────────────
Parse the raw argument value tables (KPI codes, location codes, etc.)
and write data/processed/argument_values.json.

This file maps parameter semantic types → list of {code, label, group} dicts.
The retriever uses this catalog to surface relevant values per query.

Usage:
    python scripts/build_argument_values.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_PATH = Path("data/processed/argument_values.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── KPI codes ─────────────────────────────────────────────────────────────────
KPI_CODES = [
    {
        "code": "call_setup_success_rate",
        "label": "Tỷ lệ thiết lập cuộc gọi thành công",
        "group": "voice",
    },
    {"code": "call_drop_rate", "label": "Tỷ lệ rớt cuộc gọi", "group": "voice"},
    {"code": "data_traffic_volume", "label": "Lưu lượng dữ liệu", "group": "data"},
    {
        "code": "network_latency_avg_ms",
        "label": "Độ trễ mạng trung bình (ms)",
        "group": "quality",
    },
    {
        "code": "service_cancellation_rate",
        "label": "Tỷ lệ hủy dịch vụ",
        "group": "service",
    },
    {
        "code": "voice_call_duration_avg",
        "label": "Thời lượng cuộc gọi thoại trung bình (phút)",
        "group": "voice",
    },
    {
        "code": "data_usage_per_user_4g",
        "label": "Mức sử dụng dữ liệu trung bình trên mỗi thuê bao 4G",
        "group": "data",
    },
    {
        "code": "active_user_count_4g",
        "label": "Số lượng người dùng 4G hoạt động",
        "group": "data",
    },
    {
        "code": "interruption_incident_count",
        "label": "Số lượng sự cố gián đoạn dịch vụ",
        "group": "quality",
    },
    {
        "code": "subscriber_growth_rate",
        "label": "Tỷ lệ tăng trưởng thuê bao mới",
        "group": "subscriber",
    },
    {
        "code": "coverage_percentage_4g",
        "label": "Tỷ lệ phủ sóng 4G (%)",
        "group": "coverage",
    },
    {
        "code": "user_experience_score",
        "label": "Điểm trải nghiệm người dùng",
        "group": "quality",
    },
]

# ── Location codes ────────────────────────────────────────────────────────────
LOCATION_CODES = [
    # National
    {
        "code": "VNM",
        "label": "Việt Nam",
        "alt_label": "toàn quốc",
        "group": "Toàn quốc",
    },
    # Regions
    {
        "code": "KV1",
        "label": "Khu vực 1",
        "alt_label": "Khu vực miền Bắc",
        "group": "Khu vực",
    },
    {
        "code": "KV2",
        "label": "Khu vực 2",
        "alt_label": "Khu vực miền Trung",
        "group": "Khu vực",
    },
    {
        "code": "KV3",
        "label": "Khu vực 3",
        "alt_label": "Khu vực miền Nam",
        "group": "Khu vực",
    },
    # Provinces
    {"code": "HNI", "label": "Hà Nội", "group": "Tỉnh/Thành phố"},
    {"code": "HUE", "label": "Huế", "group": "Tỉnh/Thành phố"},
    {"code": "LCU", "label": "Lai Châu", "group": "Tỉnh/Thành phố"},
    {"code": "DBN", "label": "Điện Biên", "group": "Tỉnh/Thành phố"},
    {"code": "SLA", "label": "Sơn La", "group": "Tỉnh/Thành phố"},
    {"code": "LSN", "label": "Lạng Sơn", "group": "Tỉnh/Thành phố"},
    {"code": "QNH", "label": "Quảng Ninh", "group": "Tỉnh/Thành phố"},
    {"code": "THA", "label": "Thanh Hóa", "group": "Tỉnh/Thành phố"},
    {"code": "NAN", "label": "Nghệ An", "group": "Tỉnh/Thành phố"},
    {"code": "HTH", "label": "Hà Tĩnh", "group": "Tỉnh/Thành phố"},
    {"code": "CBG", "label": "Cao Bằng", "group": "Tỉnh/Thành phố"},
    {"code": "TQG", "label": "Tuyên Quang", "group": "Tỉnh/Thành phố"},
    {"code": "LCI", "label": "Lào Cai", "group": "Tỉnh/Thành phố"},
    {"code": "TNN", "label": "Thái Nguyên", "group": "Tỉnh/Thành phố"},
    {"code": "PTO", "label": "Phú Thọ", "group": "Tỉnh/Thành phố"},
    {"code": "BNH", "label": "Bắc Ninh", "group": "Tỉnh/Thành phố"},
    {"code": "HYN", "label": "Hưng Yên", "group": "Tỉnh/Thành phố"},
    {"code": "HPG", "label": "Hải Phòng", "group": "Tỉnh/Thành phố"},
    {"code": "NBH", "label": "Ninh Bình", "group": "Tỉnh/Thành phố"},
    {"code": "QTI", "label": "Quảng Trị", "group": "Tỉnh/Thành phố"},
    {"code": "DNG", "label": "Đà Nẵng", "group": "Tỉnh/Thành phố"},
    {"code": "QNI", "label": "Quảng Ngãi", "group": "Tỉnh/Thành phố"},
    {"code": "GLI", "label": "Gia Lai", "group": "Tỉnh/Thành phố"},
    {"code": "KHA", "label": "Khánh Hòa", "group": "Tỉnh/Thành phố"},
    {"code": "LDG", "label": "Lâm Đồng", "group": "Tỉnh/Thành phố"},
    {"code": "DLK", "label": "Đắk Lắk", "group": "Tỉnh/Thành phố"},
    {"code": "HCM", "label": "Thành phố Hồ Chí Minh", "group": "Tỉnh/Thành phố"},
    {"code": "DNI", "label": "Đồng Nai", "group": "Tỉnh/Thành phố"},
    {"code": "TNH", "label": "Tây Ninh", "group": "Tỉnh/Thành phố"},
    {"code": "CTO", "label": "Cần Thơ", "group": "Tỉnh/Thành phố"},
    {"code": "VLG", "label": "Vĩnh Long", "group": "Tỉnh/Thành phố"},
    {"code": "DTP", "label": "Đồng Tháp", "group": "Tỉnh/Thành phố"},
    {"code": "CMU", "label": "Cà Mau", "group": "Tỉnh/Thành phố"},
    {"code": "AGG", "label": "An Giang", "group": "Tỉnh/Thành phố"},
    # International
    {"code": "KHM", "label": "Cambodia", "group": "Quốc tế"},
    {"code": "LAO", "label": "Laos", "group": "Quốc tế"},
    {"code": "PER", "label": "Peru", "group": "Quốc tế"},
    {"code": "MMR", "label": "Myanmar", "group": "Quốc tế"},
    {"code": "TLS", "label": "TimorLeste", "group": "Quốc tế"},
    {"code": "HTI", "label": "Haiti", "group": "Quốc tế"},
    {"code": "MOZ", "label": "Mozambique", "group": "Quốc tế"},
    {"code": "TZA", "label": "Tanzania", "group": "Quốc tế"},
    {"code": "CMR", "label": "Cameroon", "group": "Quốc tế"},
    {"code": "BDI", "label": "Burundi", "group": "Quốc tế"},
]

# ── Technology types ──────────────────────────────────────────────────────────
TECH_TYPES = [
    {"code": "2G", "label": "2G GSM", "group": "technology"},
    {"code": "3G", "label": "3G WCDMA/HSPA", "group": "technology"},
    {"code": "4G", "label": "4G LTE", "group": "technology"},
    {"code": "5G", "label": "5G NR", "group": "technology"},
]

# ── Network providers ─────────────────────────────────────────────────────────
NETWORK_PROVIDERS = [
    {"code": "viettel", "label": "Viettel", "group": "provider"},
    {"code": "vinaphone", "label": "VinaPhone (VNPT)", "group": "provider"},
    {"code": "mobifone", "label": "MobiFone", "group": "provider"},
    {"code": "reddi", "label": "Reddi (GTel)", "group": "provider"},
    {"code": "gmobile", "label": "Gmobile", "group": "provider"},
]

# ── Data aggregation levels ───────────────────────────────────────────────────
DATA_LEVELS = [
    {"code": "day", "label": "Ngày (daily)", "group": "aggregation"},
    {"code": "week", "label": "Tuần (weekly)", "group": "aggregation"},
    {"code": "month", "label": "Tháng (monthly)", "group": "aggregation"},
    {"code": "year", "label": "Năm (yearly)", "group": "aggregation"},
]

# ── Speedtest providers ───────────────────────────────────────────────────────
SPEEDTEST_PROVIDERS = [
    {"code": "ookla", "label": "Ookla Speedtest", "group": "speedtest"},
    {"code": "nperf", "label": "nPerf", "group": "speedtest"},
    {"code": "internal", "label": "Internal probe", "group": "speedtest"},
]

# ── Location types (for regional station queries) ─────────────────────────────
LOCATION_TYPES = [
    {"code": "NTMN", "label": "Nông thôn miền núi", "group": "location_type"},
    {"code": "TTLL", "label": "Thị trấn lớn lẻ", "group": "location_type"},
    {"code": "TTLN", "label": "Thị trấn lớn nhỏ", "group": "location_type"},
    {"code": "TPTW", "label": "Thành phố Trung ương", "group": "location_type"},
    {"code": "TPTT", "label": "Thành phố tỉnh", "group": "location_type"},
]

# ── Query types (for regional station queries) ────────────────────────────────
QUERY_TYPES = [
    {
        "code": "station_distance",
        "label": "Khoảng cách trung bình giữa các trạm",
        "group": "query_type",
    },
    {"code": "coverage", "label": "Vùng phủ sóng", "group": "query_type"},
    {"code": "station_count", "label": "Số lượng trạm", "group": "query_type"},
    {"code": "capacity", "label": "Dung lượng mạng", "group": "query_type"},
]

# ── Alarm types ───────────────────────────────────────────────────────────────
ALARM_TYPES = [
    {"code": "CRITICAL", "label": "Nghiêm trọng (Critical)", "group": "alarm_severity"},
    {"code": "MAJOR", "label": "Chính (Major)", "group": "alarm_severity"},
    {"code": "MINOR", "label": "Phụ (Minor)", "group": "alarm_severity"},
    {"code": "WARNING", "label": "Cảnh báo (Warning)", "group": "alarm_severity"},
]


# ── Master catalog ─────────────────────────────────────────────────────────────
# Maps parameter_semantic_type → list of value entries
# The retriever maps function parameter names → semantic types using this.
ARGUMENT_VALUES: dict = {
    # Location parameters
    "location_code": LOCATION_CODES,
    "province_code": LOCATION_CODES,
    "region_code": LOCATION_CODES,
    # KPI parameters
    "kpi_code": KPI_CODES,
    "kpi_name": KPI_CODES,
    "metric": KPI_CODES,
    # Technology parameters
    "tech_type": TECH_TYPES,
    "technology": TECH_TYPES,
    "network_type": TECH_TYPES,
    # Provider parameters
    "network_provider": NETWORK_PROVIDERS,
    "provider": NETWORK_PROVIDERS,
    "operator": NETWORK_PROVIDERS,
    # Aggregation parameters
    "data_level": DATA_LEVELS,
    "aggregation_level": DATA_LEVELS,
    "time_granularity": DATA_LEVELS,
    # Speedtest
    "speedtest_provider": SPEEDTEST_PROVIDERS,
    # Location type
    "location_type": LOCATION_TYPES,
    "area_type": LOCATION_TYPES,
    # Query type
    "query_type": QUERY_TYPES,
    # Alarm
    "alarm_type": ALARM_TYPES,
    "severity": ALARM_TYPES,
}


def main():
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(ARGUMENT_VALUES, fh, indent=2, ensure_ascii=False)
    print(f"✓ Argument values catalog saved → {OUTPUT_PATH}")
    print(f"  Parameter types : {len(ARGUMENT_VALUES)}")
    total_values = sum(len(v) for v in ARGUMENT_VALUES.values())
    print(f"  Total values    : {total_values}")

    # Print summary
    for param_type, values in ARGUMENT_VALUES.items():
        print(f"  {param_type:25s}: {len(values):3d} values")


if __name__ == "__main__":
    main()
