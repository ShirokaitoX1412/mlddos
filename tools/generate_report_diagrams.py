"""Sinh các sơ đồ PNG dùng trong báo cáo đồ án DDoS ML/SDN."""

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
        y + h - 2.3,
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
            y + h / 2 - 0.8,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            family="DejaVu Sans",
            linespacing=1.3,
        )
    return patch


def arrow(ax, start, end, color="#334155", rad: float = 0.0, label: str | None = None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
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


def title(ax, text: str):
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


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_system_architecture():
    fig, ax = setup_figure()
    title(ax, "Kiến trúc hệ thống phát hiện tấn công từ chối dịch vụ phân tán")

    box(ax, 4, 67, 19, 16, "Máy h1", "Máy khách hợp lệ\n10.0.0.1", "green")
    box(ax, 4, 42, 19, 16, "Máy h3", "Nguồn tấn công\nngập lụt gói tin\n10.0.0.3", "red")
    box(ax, 4, 17, 19, 16, "Máy h2", "Máy chủ nạn nhân\ndịch vụ máy chủ\n10.0.0.2", "blue")

    box(ax, 31, 39, 16, 24, "Bộ chuyển mạch ảo", "Nút s1 trong mạng\ngiao thức điều khiển\nchuyển tiếp hoặc chặn", "slate")
    box(ax, 56, 58, 20, 22, "Bộ điều khiển mạng", "Học địa chỉ thiết bị\nthu thập thống kê luồng\ntrích xuất đặc trưng", "amber")
    box(ax, 56, 28, 20, 18, "Khối suy luận", "Tệp mô hình đã huấn luyện\ndự đoán bình thường/tấn công\nngưỡng tin cậy", "violet")
    box(ax, 82, 63, 14, 13, "Bảng giám sát", "Màn hình trực tiếp\nbiểu đồ cảnh báo", "cyan")
    box(ax, 82, 41, 14, 13, "Nhật ký sự kiện", "Tệp kết quả\nnhật ký cảnh báo", "slate")
    box(ax, 82, 19, 14, 13, "Giảm thiểu", "Luật điều khiển\nchặn luồng\nkhi bật chế độ chặn", "red")

    arrow(ax, (23, 75), (31, 55), label="lưu lượng hợp lệ")
    arrow(ax, (23, 50), (31, 51), color=COLORS["red_edge"], label="lưu lượng tấn công")
    arrow(ax, (47, 49), (56, 68), label="gói tin và thống kê")
    arrow(ax, (66, 58), (66, 46), label="đặc trưng luồng")
    arrow(ax, (76, 69), (82, 69), label="sự kiện")
    arrow(ax, (76, 37), (82, 47), label="ghi nhật ký")
    arrow(ax, (76, 33), (82, 26), color=COLORS["red_edge"], label="quyết định")
    arrow(ax, (82, 23), (47, 43), color=COLORS["red_edge"], rad=-0.24)
    arrow(ax, (31, 43), (23, 25), label="chuyển tiếp/chặn")

    ax.text(
        50,
        7,
        "Mặc định hệ thống chạy ở chế độ quan sát. Khi bật chế độ chặn, bộ điều khiển cài luật loại bỏ luồng tấn công.",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
        family="DejaVu Sans",
    )
    save(fig, "kien_truc_he_thong_sdn_ml_ids_ips")


def draw_data_preprocessing():
    fig, ax = setup_figure()
    title(ax, "Quy trình tiền xử lý dữ liệu luồng mạng")

    steps = [
        (4, 60, "Dữ liệu gốc", "CICDDoS2019\ncác tệp luồng mạng", "blue"),
        (21, 60, "Đọc dữ liệu", "gộp tập huấn luyện\nvà tập kiểm thử", "cyan"),
        (38, 60, "Làm sạch", "loại giá trị thiếu\nvà giá trị vô hạn", "slate"),
        (55, 60, "Chuẩn hóa nhãn", "đưa các kiểu tấn công\nvề nhóm thống nhất", "amber"),
        (72, 60, "Loại rò rỉ", "bỏ địa chỉ IP\nthời gian và mã phiên", "red"),
    ]
    for x, y, heading, body, color in steps:
        box(ax, x, y, 14, 17, heading, body, color)
    for x in (18, 35, 52, 69):
        arrow(ax, (x, 68), (x + 3, 68))

    box(ax, 16, 28, 18, 17, "Mã hóa và chuẩn hóa", "bộ mã hóa biến phân loại\nbộ co giãn biến số", "violet")
    box(ax, 41, 28, 18, 17, "Chọn đặc trưng", "loại cột hằng\nloại tương quan cao", "green")
    box(ax, 66, 28, 18, 17, "Tập dữ liệu sạch", "đầu vào cho mô hình\nkhông xử lý trước khi chia", "blue")
    arrow(ax, (79, 60), (75, 45), label="sau khi chia tập")
    arrow(ax, (34, 36), (41, 36), label="đặc trưng")
    arrow(ax, (59, 36), (66, 36), label="dữ liệu đã chuẩn bị")

    box(ax, 18, 8, 64, 10, "Nguyên tắc quan trọng", "Bộ tiền xử lý chỉ được học từ tập huấn luyện, sau đó mới áp dụng cho tập kiểm thử để tránh rò rỉ dữ liệu.", "slate", title_size=11, body_size=9)
    save(fig, "quy_trinh_tien_xu_ly_du_lieu")


def draw_training_evaluation():
    fig, ax = setup_figure()
    title(ax, "Quy trình huấn luyện, đánh giá và lựa chọn mô hình")

    box(ax, 5, 65, 16, 15, "Tập huấn luyện", "dữ liệu đã làm sạch\nchia có phân tầng", "blue")
    box(ax, 28, 65, 18, 15, "Kiểm định chéo", "nhiều lượt đánh giá\ntrung bình và độ lệch chuẩn", "cyan")
    box(ax, 53, 65, 18, 15, "Tối ưu tham số", "tìm kiếm ngẫu nhiên\ncho các mô hình cây", "amber")
    box(ax, 78, 65, 16, 15, "Mô hình ứng viên", "Rừng ngẫu nhiên\nCây cực ngẫu nhiên\nXGBoost", "violet")

    arrow(ax, (21, 72), (28, 72))
    arrow(ax, (46, 72), (53, 72))
    arrow(ax, (71, 72), (78, 72))

    box(ax, 15, 34, 18, 16, "Đánh giá kiểm thử", "Accuracy\nPrecision\nRecall\nF1-score\nROC-AUC", "green")
    box(ax, 41, 34, 18, 16, "Phân tích quá khớp", "so sánh huấn luyện\nkiểm định và kiểm thử", "red")
    box(ax, 67, 34, 18, 16, "Lựa chọn cuối", "ưu tiên F1 cân bằng\nvà khả năng tổng quát", "blue")

    arrow(ax, (86, 65), (76, 50), label="đưa ra kiểm thử")
    arrow(ax, (33, 42), (41, 42))
    arrow(ax, (59, 42), (67, 42))

    box(ax, 26, 8, 48, 12, "Đầu ra", "Tệp selected_model.pkl, bảng kết quả, ma trận nhầm lẫn, đường cong ROC và biểu đồ tầm quan trọng đặc trưng.", "slate", title_size=11, body_size=9)
    arrow(ax, (76, 34), (59, 20), label="lưu mô hình")
    save(fig, "quy_trinh_huan_luyen_danh_gia_mo_hinh")


def draw_hyperparameter_optimization():
    fig, ax = setup_figure()
    title(ax, "Sơ đồ tối ưu tham số trong quá trình huấn luyện mô hình")

    box(ax, 4, 64, 16, 16, "Dữ liệu huấn luyện", "Tập dữ liệu đã tiền xử lý\nvà giữ nguyên nhãn gốc", "blue")
    box(ax, 25, 64, 17, 16, "Không gian tham số", "Số cây\nđộ sâu cây\ntốc độ học\ntrọng số lớp", "amber")
    box(ax, 47, 64, 17, 16, "Lấy mẫu cấu hình", "Chọn ngẫu nhiên\nnhiều tổ hợp tham số", "cyan")
    box(ax, 69, 64, 17, 16, "Kiểm định chéo", "Chia nhiều lượt\nhuấn luyện và đánh giá", "violet")

    arrow(ax, (20, 72), (25, 72), label="đưa vào")
    arrow(ax, (42, 72), (47, 72), label="lấy mẫu")
    arrow(ax, (64, 72), (69, 72), label="đánh giá")

    box(ax, 8, 34, 18, 16, "Tính điểm", "F1-score\nPrecision\nRecall\nđộ lệch chuẩn", "green")
    box(ax, 32, 34, 18, 16, "So sánh cấu hình", "Ưu tiên hiệu quả ổn định\nkhông chọn theo accuracy đơn lẻ", "slate")
    box(ax, 56, 34, 18, 16, "Cấu hình tốt nhất", "Bộ tham số tối ưu\ncho từng mô hình ứng viên", "blue")
    box(ax, 78, 34, 16, 16, "Huấn luyện lại", "Fit trên toàn bộ\ntập huấn luyện", "amber")

    arrow(ax, (78, 64), (17, 50), label="kết quả từng lượt")
    arrow(ax, (26, 42), (32, 42), label="tổng hợp")
    arrow(ax, (50, 42), (56, 42), label="chọn")
    arrow(ax, (74, 42), (78, 42), label="fit lại")

    box(ax, 24, 8, 20, 14, "Đánh giá kiểm thử", "Chỉ dùng một lần\nsau khi chọn tham số", "red")
    box(ax, 56, 8, 20, 14, "Mô hình triển khai", "Lưu thành tệp mô hình\nphục vụ demo IDS/IPS", "violet")
    arrow(ax, (86, 34), (44, 22), label="kiểm tra tổng quát")
    arrow(ax, (44, 15), (56, 15), label="đạt yêu cầu")

    ax.text(
        50,
        3,
        "Mục tiêu của tối ưu tham số là tăng khả năng tổng quát hóa, không làm tăng giả tạo điểm trên tập kiểm định.",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
        family="DejaVu Sans",
    )
    save(fig, "so_do_toi_uu_tham_so_mo_hinh")


def draw_demo_topology():
    fig, ax = setup_figure()
    title(ax, "Sơ đồ mạng mô phỏng trong môi trường SDN")

    box(ax, 8, 63, 20, 14, "Máy h1", "máy khách hợp lệ\n10.0.0.1", "green")
    box(ax, 8, 23, 20, 14, "Máy h3", "máy phát sinh tấn công\n10.0.0.3", "red")
    box(ax, 41, 43, 18, 18, "Bộ chuyển mạch s1", "OVS\nOpenFlow 1.3", "slate")
    box(ax, 72, 63, 20, 14, "Máy h2", "máy chủ nạn nhân\n10.0.0.2", "blue")
    box(ax, 72, 23, 20, 14, "Bộ điều khiển", "Ryu\nứng dụng IDS/IPS", "amber")

    arrow(ax, (28, 70), (41, 55), label="truy cập web")
    arrow(ax, (28, 30), (41, 49), color=COLORS["red_edge"], label="ngập lụt gói tin")
    arrow(ax, (59, 52), (72, 70), label="chuyển tiếp")
    arrow(ax, (59, 46), (72, 30), label="kênh điều khiển")
    arrow(ax, (72, 25), (59, 45), color=COLORS["red_edge"], label="luật chặn")

    box(ax, 23, 8, 54, 10, "Mục đích mô phỏng", "Kiểm chứng việc bộ điều khiển SDN quan sát luồng, phân loại bằng mô hình học máy và có thể chặn nguồn tấn công.", "slate", title_size=11, body_size=9)
    save(fig, "so_do_mang_mo_phong_sdn")


def draw_realtime_monitoring():
    fig, ax = setup_figure()
    title(ax, "Luồng giám sát thời gian thực và ghi nhận cảnh báo")

    nodes = [
        (5, 60, "Luồng mạng", "gói tin trong SDN\nhoặc dữ liệu phát lại", "blue"),
        (24, 60, "Bộ tổng hợp luồng", "gom theo địa chỉ\ncổng và giao thức", "cyan"),
        (43, 60, "Trích xuất đặc trưng", "thời lượng\nsố gói\nsố byte\ntốc độ gói", "amber"),
        (62, 60, "Mô hình học máy", "phân loại\nbình thường/DDoS", "violet"),
        (81, 60, "Cảnh báo", "mức tin cậy\ntrạng thái chặn", "red"),
    ]
    for x, y, heading, body, color in nodes:
        box(ax, x, y, 14, 18, heading, body, color)
    for x in (19, 38, 57, 76):
        arrow(ax, (x, 69), (x + 5, 69))

    box(ax, 20, 28, 19, 15, "Nhật ký", "live_events.csv\nlưu mọi sự kiện", "slate")
    box(ax, 50, 28, 19, 15, "Bảng giám sát", "hiển thị trực quan\nbiểu đồ và bảng log", "green")
    box(ax, 73, 28, 19, 15, "Bằng chứng demo", "ảnh chụp màn hình\nvà kết quả cảnh báo", "blue")

    arrow(ax, (88, 60), (33, 43), label="ghi sự kiện")
    arrow(ax, (39, 36), (50, 36), label="đọc dữ liệu")
    arrow(ax, (69, 36), (73, 36), label="trình bày")
    save(fig, "luong_giam_sat_thoi_gian_thuc")


def draw_sdn_detection_sequence():
    fig, ax = setup_figure()
    title(ax, "Luồng phát hiện và giảm thiểu DDoS trong SDN")

    participants = [
        (10, "Máy tấn công\nh3", "red"),
        (28, "Bộ chuyển mạch\ns1", "slate"),
        (46, "Bộ điều khiển\nRyu", "amber"),
        (64, "Mô hình\nhọc máy", "violet"),
        (82, "Bảng giám sát\nvà nhật ký", "cyan"),
    ]
    for x, label, color in participants:
        box(ax, x - 6, 78, 12, 10, label, "", color, title_size=10)
        ax.plot([x, x], [12, 78], color="#cbd5e1", linewidth=1.2, linestyle="--")

    steps = [
        (71, 10, 28, "1. Gửi lưu lượng tấn công"),
        (62, 28, 46, "2. Gửi gói tin và thống kê"),
        (53, 46, 64, "3. Trích xuất đặc trưng"),
        (44, 64, 46, "4. Trả về nhãn và độ tin cậy"),
        (35, 46, 82, "5. Ghi nhật ký cảnh báo"),
        (26, 46, 28, "6. Cài luật chặn khi bật IPS"),
        (17, 28, 10, "7. Luồng tấn công bị loại bỏ"),
    ]
    for y, x1, x2, label in steps:
        color = COLORS["red_edge"] if "tấn công" in label or "chặn" in label else "#334155"
        arrow(ax, (x1, y), (x2, y), color=color, label=label)

    box(ax, 14, 3, 72, 7, "Kết quả quan sát", "Chế độ quan sát chỉ ghi cảnh báo. Khi bật IPS, hệ thống vừa ghi cảnh báo vừa cài luật chặn trên OVS.", "slate", title_size=10, body_size=9)
    save(fig, "luong_phat_hien_va_giam_thieu_ddos_sdn")


def main():
    draw_system_architecture()
    draw_data_preprocessing()
    draw_training_evaluation()
    draw_hyperparameter_optimization()
    draw_demo_topology()
    draw_realtime_monitoring()
    draw_sdn_detection_sequence()
    print(f"Generated PNG diagrams in: {OUT_DIR}")


if __name__ == "__main__":
    main()
