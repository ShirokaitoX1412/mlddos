"""Sinh các sơ đồ PNG dùng trong báo cáo đồ án DDoS ML/IDS/IPS."""

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


def box(ax, x, y, w, h, heading, body="", color="slate", title_size=11, body_size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=1.8,
        edgecolor=COLORS[f"{color}_edge"],
        facecolor=COLORS[color],
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 2.3,
        heading,
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
            y + h / 2 - 0.8,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            family="DejaVu Sans",
            linespacing=1.3,
        )


def arrow(ax, start, end, color="#334155", label=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
        )
    )
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
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.86),
        )


def title(ax, text):
    ax.text(
        50,
        96,
        text,
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_system_architecture():
    fig, ax = setup_figure()
    title(ax, "Kiến trúc hệ thống phát hiện DDoS dùng học máy")

    box(ax, 4, 66, 20, 16, "Khối dữ liệu", "CICDDoS2019\nBenign và các lớp DDoS", "blue")
    box(ax, 30, 66, 20, 16, "Khối huấn luyện", "làm sạch dữ liệu\nchia train/test\nhuấn luyện mô hình", "cyan")
    box(ax, 56, 66, 18, 16, "Khối đánh giá", "Accuracy\nBalanced Accuracy\nF1-score\nROC-AUC", "green")
    box(ax, 80, 66, 16, 16, "Mô hình tốt nhất", "selected_model.pkl\npipeline đã huấn luyện", "violet")
    arrow(ax, (24, 74), (30, 74), label="dữ liệu")
    arrow(ax, (50, 74), (56, 74), label="kết quả")
    arrow(ax, (74, 74), (80, 74), label="lựa chọn")

    box(ax, 6, 35, 20, 17, "Máy ảo Attacker", "sinh traffic kiểm thử\nBenign, SYN, UDP,\nLDAP, MSSQL, NetBIOS", "red")
    box(ax, 40, 35, 20, 17, "Mạng nội bộ", "Host-only/Internal Network\nphục vụ lab an toàn", "slate")
    box(ax, 74, 35, 20, 17, "Máy ảo Victim", "dịch vụ đích\nbắt gói và chạy IDS/IPS", "blue")
    arrow(ax, (26, 43), (40, 43), color=COLORS["red_edge"], label="traffic kiểm thử")
    arrow(ax, (60, 43), (74, 43), label="traffic đến Victim")

    box(ax, 8, 10, 18, 14, "Bắt gói", "tcpdump/Scapy\nquan sát lưu lượng", "amber")
    box(ax, 32, 10, 18, 14, "Trích xuất flow", "gom gói tin thành\nđặc trưng số", "cyan")
    box(ax, 56, 10, 18, 14, "Dự đoán", "Benign hoặc\nloại tấn công DDoS", "violet")
    box(ax, 80, 10, 16, 14, "Dashboard", "cảnh báo\nlog và biểu đồ", "green")
    arrow(ax, (82, 35), (17, 24), label="lưu lượng quan sát")
    arrow(ax, (26, 17), (32, 17))
    arrow(ax, (50, 17), (56, 17))
    arrow(ax, (74, 17), (80, 17))

    save(fig, "kien_truc_he_thong_hai_may_ao_ml_ids_ips")


def draw_data_preprocessing():
    fig, ax = setup_figure()
    title(ax, "Quy trình tiền xử lý dữ liệu luồng mạng")
    steps = [
        (4, 60, "Dữ liệu gốc", "CICDDoS2019\ncác tệp flow", "blue"),
        (21, 60, "Đọc dữ liệu", "gộp dữ liệu\nhuấn luyện và kiểm thử", "cyan"),
        (38, 60, "Làm sạch", "loại bản ghi trùng\nthiếu và vô hạn", "slate"),
        (55, 60, "Chuẩn hóa nhãn", "đưa tên lớp về\nhệ thống thống nhất", "amber"),
        (72, 60, "Loại rò rỉ", "bỏ IP, cổng\nthời gian, mã flow", "red"),
    ]
    for item in steps:
        box(ax, *item)
    for x in (18, 35, 52, 69):
        arrow(ax, (x, 68), (x + 3, 68))

    box(ax, 16, 28, 18, 17, "Mã hóa và chuẩn hóa", "imputer, scaler\nfit trên train set", "violet")
    box(ax, 41, 28, 18, 17, "Chọn đặc trưng", "loại cột hằng\nloại tương quan cao", "green")
    box(ax, 66, 28, 18, 17, "Dữ liệu sẵn sàng", "đầu vào cho mô hình\nkhông rò rỉ nhãn", "blue")
    arrow(ax, (79, 60), (75, 45), label="sau khi chia tập")
    arrow(ax, (34, 36), (41, 36), label="đặc trưng")
    arrow(ax, (59, 36), (66, 36), label="dữ liệu đã chuẩn bị")
    save(fig, "quy_trinh_tien_xu_ly_du_lieu")


def draw_training_evaluation():
    fig, ax = setup_figure()
    title(ax, "Quy trình huấn luyện, đánh giá và lựa chọn mô hình")
    box(ax, 5, 65, 16, 15, "Tập huấn luyện", "dữ liệu đã làm sạch\nchia có phân tầng", "blue")
    box(ax, 28, 65, 18, 15, "Kiểm định chéo", "nhiều lượt đánh giá\nmean và std", "cyan")
    box(ax, 53, 65, 18, 15, "Tối ưu tham số", "giới hạn độ sâu\nregularization\nclass weight", "amber")
    box(ax, 78, 65, 16, 15, "Mô hình ứng viên", "RF, ET, KNN\nMLP, XGBoost", "violet")
    arrow(ax, (21, 72), (28, 72))
    arrow(ax, (46, 72), (53, 72))
    arrow(ax, (71, 72), (78, 72))

    box(ax, 15, 34, 18, 16, "Đánh giá kiểm thử", "Accuracy\nBalanced Accuracy\nMacro F1\nROC-AUC", "green")
    box(ax, 41, 34, 18, 16, "Phân tích quá khớp", "so sánh train/test\nvà cảnh báo gap", "red")
    box(ax, 67, 34, 18, 16, "Lựa chọn cuối", "ưu tiên Macro F1\nvà khả năng tổng quát", "blue")
    arrow(ax, (86, 65), (76, 50), label="đưa ra kiểm thử")
    arrow(ax, (33, 42), (41, 42))
    arrow(ax, (59, 42), (67, 42))
    save(fig, "quy_trinh_huan_luyen_danh_gia_mo_hinh")


def draw_two_vm_demo_topology():
    fig, ax = setup_figure()
    title(ax, "Sơ đồ mạng demo hai máy ảo Attacker - Victim")
    box(ax, 8, 55, 22, 18, "Máy ảo Attacker", "sinh lưu lượng kiểm thử\nTCP SYN, UDP, LDAP,\nMSSQL, NetBIOS, Benign", "red")
    box(ax, 39, 55, 22, 18, "Mạng nội bộ lab", "Host-only/Internal Network\nkhông phát tán ra Internet", "slate")
    box(ax, 70, 55, 22, 18, "Máy ảo Victim", "dịch vụ web/DNS mẫu\nIDS/IPS engine\nDashboard", "blue")
    arrow(ax, (30, 64), (39, 64), color=COLORS["red_edge"], label="gói kiểm thử")
    arrow(ax, (61, 64), (70, 64), label="traffic đến dịch vụ")
    save(fig, "so_do_mang_demo_hai_may_ao")


def draw_realtime_monitoring():
    fig, ax = setup_figure()
    title(ax, "Luồng giám sát thời gian thực và ghi nhận cảnh báo")
    nodes = [
        (5, 60, "Luồng mạng", "traffic từ Attacker\nđến Victim", "blue"),
        (24, 60, "Bộ tổng hợp flow", "gom theo IP, cổng\ngiao thức và thời gian", "cyan"),
        (43, 60, "Trích xuất đặc trưng", "số gói, số byte\nthời lượng, tốc độ", "amber"),
        (62, 60, "Mô hình học máy", "phân loại\nBenign/DDoS", "violet"),
        (81, 60, "Cảnh báo", "nhãn dự đoán\nđộ tin cậy", "red"),
    ]
    for item in nodes:
        box(ax, *item)
    for x in (19, 38, 57, 76):
        arrow(ax, (x, 69), (x + 5, 69))
    box(ax, 20, 28, 19, 15, "Nhật ký", "live_events.csv\nlưu sự kiện", "slate")
    box(ax, 50, 28, 19, 15, "Dashboard", "hiển thị trực quan\nlog và biểu đồ", "green")
    box(ax, 73, 28, 19, 15, "Bằng chứng demo", "ảnh chụp màn hình\nvà số liệu cảnh báo", "blue")
    arrow(ax, (88, 60), (33, 43), label="ghi sự kiện")
    arrow(ax, (39, 36), (50, 36), label="đọc dữ liệu")
    arrow(ax, (69, 36), (73, 36), label="trình bày")
    save(fig, "luong_giam_sat_thoi_gian_thuc")


def main():
    draw_system_architecture()
    draw_data_preprocessing()
    draw_training_evaluation()
    draw_two_vm_demo_topology()
    draw_realtime_monitoring()
    print(f"Generated PNG diagrams in: {OUT_DIR}")


if __name__ == "__main__":
    main()
