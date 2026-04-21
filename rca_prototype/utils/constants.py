RANDOM_SEED = 42

SCENARIO_TITLE = "Mounter의 pickup 불안정이 생산 손실과 AOI 하류 품질 영향으로 이어지는 시나리오"

PROCESS_FLOW = [
    ("LD", "Loader / Input"),
    ("PR", "Printer"),
    ("SPI", "SPI"),
    ("MNT", "Mounter"),
    ("RFL", "Reflow"),
    ("AOI", "AOI"),
    ("ULD", "Unloader / Output"),
]

ISSUE_WINDOW = {
    "date": "2026-03-24",
    "hours": [14, 15, 16],
    "machine_id": "M05",
    "feeder_id": "FDR-05",
    "nozzle_id": "NZ-03",
    "part_number": "PN-004",
    "affected_lots": ["LOT-002", "LOT-003"],
}

CAUSE_ORDER = [
    "자재 / 피더 / 노즐 관련",
    "설비 관련",
    "공정 흐름 관련",
    "제품 / LOT 관련",
]
