"""Generate report-ready diagrams for the DDoS ML/SDN graduation project.

Outputs PNG and SVG files into docs/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

COLORS = {
    "bg": "#f8fafc",
    "text": "#0f172a",
    "muted": "#475569",
    "green": "#dcfce7",
    "green_edge": "#16a34a",
    "blue": "#dbeafe",
    "blue_edge": "#2563eb",
    "cyan": "#cffafe",
    "cyan_edge": "#0891b2",
    "amber": "#fef3c7",
    "amber_edge": "#d97706",
    "red": "#fee2e2",
    "red_edge": "#dc2626",
    "violet": "#ede9fe",
    "violet_edge": "#7c3aed",
    "slate": "#e2e8f0",
    "slate_edge": "#64748b",
}


def setup_figure(width: float = 16, height: float = 9):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    fc: str = "slate",
    ec: str | None = None,
    title_size: int = 11,
    body_size: int = 9,
):
    edge = COLORS[ec] if ec else COLORS[f"{fc}_edge"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=COLORS[fc],
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 2.4,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 1,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            family="DejaVu Sans",
            linespacing=1.35,
        )
    return patch


def arrow(ax, start, end, color="#334155", rad: float = 0.0, label: str | None = None):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 2,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_system_architecture():
    fig, ax = setup_figure()
    ax.text(
        50,
        96,
        "Kiến trúc hệ thống phát hiện DDoS sử dụng học máy trong mạng SDN",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )

    box(ax, 4, 67, 19, 16, "Host h1", "Máy khách hợp lệ\n10.0.0.1", "green")
    box(ax, 4, 42, 19, 16, "Host h3", "Nguồn tấn công\nSYN/UDP flood\n10.0.0.3", "red")
    box(ax, 4, 17, 19, 16, "Host h2", "Máy chủ nạn nhân\nHTTP service\n10.0.0.2", "blue")

    box(ax, 31, 39, 16, 24, "Open vSwitch", "Switch SDN s1\nOpenFlow 1.3\nChuyển tiếp / drop flow", "slate")
    box(ax, 56, 58, 20, 22, "Ryu Controller", "Learning switch\nFlow stats collector\nFeature extractor", "amber")
    box(ax, 56, 28, 20, 18, "ML Inference", "selected_model.pkl\nDự đoán Benign/DDoS\nConfidence threshold", "violet")
    box(ax, 82, 63, 14, 13, "Dashboard", "Streamlit\nLive Monitor\nBiểu đồ cảnh báo", "cyan")
    box(ax, 82, 41, 14, 13, "Event Log", "results/\nlive_events.csv", "slate")
    box(ax, 82, 19, 14, 13, "Mitigation", "OpenFlow\nDrop rule\nkhi bật IPS", "red")

    arrow(ax, (23, 75), (31, 55), label="traffic hợp lệ")
    arrow(ax, (23, 50), (31, 51), color=COLORS["red_edge"], label="traffic flood")
    arrow(ax, (47, 49), (56, 68), label="PacketIn / stats")
    arrow(ax, (66, 58), (66, 46), label="feature vector")
    arrow(ax, (76, 69), (82, 69), label="events")
    arrow(ax, (76, 37), (82, 47), label="ghi log")
    arrow(ax, (76, 33), (82, 26), color=COLORS["red_edge"], label="quyết định")
    arrow(ax, (82, 23), (47, 43), color=COLORS["red_edge"], rad=-0.24)
    arrow(ax, (31, 43), (23, 25), label="forward/drop")

    ax.text(
        50,
        7,
        "Mặc định hệ thống chạy ở chế độ quan sát (OBSERVE_ONLY). Khi bật IPS, controller cài rule drop trên Open vSwitch.",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
        family="DejaVu Sans",
    )
    save(fig, "kien_truc_he_thong_sdn_ml_ids_ips")


def draw_training_to_demo_workflow():
    fig, ax = setup_figure()
    ax.text(
        50,
        96,
        "Workflow từ dữ liệu CICDDoS2019 đến demo IDS/IPS",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )

    y_top = 69
    boxes = [
        (4, y_top, 14, 16, "Dataset", "CICDDoS2019\nCSV/Parquet\nFlow records", "blue"),
        (22, y_top, 15, 16, "Tiền xử lý", "Làm sạch NaN/Inf\nChuẩn hóa nhãn\nLoại leakage", "cyan"),
        (41, y_top, 15, 16, "Pipeline", "Imputer\nScaler\nEncoder\nfit trên train", "slate"),
        (60, y_top, 15, 16, "Huấn luyện", "RF / ExtraTrees\nXGBoost / KNN / MLP\nCV + tuning", "violet"),
        (79, y_top, 15, 16, "Model", "selected_model.pkl\nLưu để triển khai", "amber"),
    ]
    for spec in boxes:
        box(ax, *spec)
    for x in (18, 37, 56, 75):
        arrow(ax, (x, y_top + 8), (x + 4, y_top + 8))

    box(ax, 8, 33, 18, 17, "Nguồn traffic demo", "Mininet\nh1 benign\nh3 attacker\nh2 victim", "green")
    box(ax, 33, 33, 18, 17, "Ryu Runtime", "OpenFlow stats\nPacketIn evidence\nFeature extraction", "amber")
    box(ax, 58, 33, 18, 17, "Phân loại", "Model dự đoán\nBenign / DDoS\nConfidence", "violet")
    box(ax, 80, 33, 16, 17, "Đầu ra", "Log cảnh báo\nDashboard\nDrop rule tùy chọn", "red")

    arrow(ax, (87, y_top), (87, 50), color=COLORS["amber_edge"], label="triển khai model")
    arrow(ax, (26, 41), (33, 41), label="traffic")
    arrow(ax, (51, 41), (58, 41), label="features")
    arrow(ax, (76, 41), (80, 41), label="decision")

    box(ax, 22, 9, 56, 12, "Nguyên tắc chống overfitting/leakage", "Chia train/test trước khi fit preprocessing; dùng stratified CV; so sánh train-validation-test; chọn model theo F1 và độ ổn định tổng quát hóa.", "slate", title_size=11, body_size=9)
    arrow(ax, (50, 33), (50, 21), color=COLORS["slate_edge"], label="đánh giá")

    save(fig, "workflow_du_lieu_huan_luyen_va_demo")


def draw_sdn_detection_sequence():
    fig, ax = setup_figure()
    ax.text(
        50,
        96,
        "Luồng phát hiện và giảm thiểu DDoS trong SDN",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )

    participants = [
        (10, "h3\nAttacker", "red"),
        (28, "s1\nOpen vSwitch", "slate"),
        (46, "Ryu\nController", "amber"),
        (64, "ML\nModel", "violet"),
        (82, "Dashboard\n/ Log", "cyan"),
    ]
    for x, label, color in participants:
        box(ax, x - 6, 78, 12, 10, label, "", color, title_size=10)
        ax.plot([x, x], [12, 78], color="#cbd5e1", linewidth=1.2, linestyle="--")

    steps = [
        (71, 10, 28, "1. Gửi SYN/UDP flood"),
        (62, 28, 46, "2. PacketIn / flow stats"),
        (53, 46, 64, "3. Trích xuất đặc trưng"),
        (44, 64, 46, "4. Dự đoán nhãn + confidence"),
        (35, 46, 82, "5. Ghi live_events.csv"),
        (26, 46, 28, "6. Cài drop rule nếu IPS bật"),
        (17, 28, 10, "7. Luồng tấn công bị chặn"),
    ]
    for y, x1, x2, label in steps:
        color = COLORS["red_edge"] if "flood" in label or "drop" in label or "chặn" in label else "#334155"
        arrow(ax, (x1, y), (x2, y), color=color, label=label)

    box(ax, 14, 3, 72, 7, "Kết quả quan sát", "OBSERVE_ONLY=1: chỉ ghi cảnh báo.  OBSERVE_ONLY=0: ghi cảnh báo và cài OpenFlow rule để chặn flow tấn công.", "slate", title_size=10, body_size=9)

    save(fig, "luong_phat_hien_va_giam_thieu_ddos_sdn")


def main():
    draw_system_architecture()
    draw_training_to_demo_workflow()
    draw_sdn_detection_sequence()
    print(f"Generated diagrams in: {OUT_DIR}")


if __name__ == "__main__":
    main()
