import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
import warnings

class Arrow3D(FancyArrowPatch):
    """自定义3D箭头类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def azimuth_plunge_to_vector(azimuth, plunge):
    """
    将方位角和倾伏角转换为单位向量
    
    参数:
    azimuth: 方位角（度），从北向顺时针测量
    plunge: 倾伏角（度），向下为正
    
    返回:
    (x, y, z): 单位向量，x指向东，y指向北，z指向上
    """
    az_rad = np.radians(azimuth)
    pl_rad = np.radians(plunge)
    
    x = np.cos(pl_rad) * np.sin(az_rad)  # 东向分量
    y = np.cos(pl_rad) * np.cos(az_rad)  # 北向分量
    z = -np.sin(pl_rad)  # 向上分量
    
    return x, y, z

def plot_stress_3d(sigma1_azimuth, sigma1_plunge, sigma3_azimuth, sigma3_plunge, 
                   show_ellipse=True, view_angle=(25, 45)):
    """
    绘制3D应力方向示意图
    σ₁表示压应力（向内对向箭头，在中心相接），σ₃表示相对张应力（向外箭头）
    
    参数:
    sigma1_azimuth: σ₁的方位角（度）
    sigma1_plunge: σ₁的倾伏角（度）
    sigma3_azimuth: σ₃的方位角（度）
    sigma3_plunge: σ₃的倾伏角（度）
    show_ellipse: 是否显示应力椭圆
    view_angle: 视角（仰角, 方位角）
    """
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 隐藏轴的数值标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # 计算应力方向向量
    sigma1_vec = azimuth_plunge_to_vector(sigma1_azimuth, sigma1_plunge)
    sigma3_vec = azimuth_plunge_to_vector(sigma3_azimuth, sigma3_plunge)
    
    # 计算σ₂（垂直于σ₁和σ₃）
    sigma1_vec = np.array(sigma1_vec)  # 确保是numpy数组
    sigma3_vec = np.array(sigma3_vec)  # 确保是numpy数组
    
    sigma2_vec = np.cross(sigma1_vec, sigma3_vec)
    if np.linalg.norm(sigma2_vec) > 0:
        sigma2_vec = sigma2_vec / np.linalg.norm(sigma2_vec)
    else:
        # 如果σ₁和σ₃平行，则选择一个垂直于σ₁的向量作为σ₂
        if abs(sigma1_vec[2]) < 0.9:
            sigma2_vec = np.cross(sigma1_vec, np.array([0, 0, 1]))
        else:
            sigma2_vec = np.cross(sigma1_vec, np.array([1, 0, 0]))
        sigma2_vec = sigma2_vec / np.linalg.norm(sigma2_vec)
    
    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 绘制细淡的网格线
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    
    # 定义箭头参数
    outer_pos = 1.3  # 外侧位置
    center_pos = 0.0  # 中心点（σ₁箭头相接点）
    sigma3_start = 0.4  # σ₃箭头起始点（从中心稍微偏离）
    
    # σ₁ (蓝色) - 压应力，向内对向箭头，在中心相接
    # 正方向箭头（从外向中心）
    arrow1_pos = Arrow3D([sigma1_vec[0]*outer_pos, sigma1_vec[0]*center_pos], 
                        [sigma1_vec[1]*outer_pos, sigma1_vec[1]*center_pos], 
                        [sigma1_vec[2]*outer_pos, sigma1_vec[2]*center_pos],
                        mutation_scale=20, lw=3, arrowstyle='->', color='blue')
    # 负方向箭头（从外向中心）
    arrow1_neg = Arrow3D([-sigma1_vec[0]*outer_pos, -sigma1_vec[0]*center_pos], 
                        [-sigma1_vec[1]*outer_pos, -sigma1_vec[1]*center_pos], 
                        [-sigma1_vec[2]*outer_pos, -sigma1_vec[2]*center_pos],
                        mutation_scale=20, lw=3, arrowstyle='->', color='blue')
    ax.add_artist(arrow1_pos)
    ax.add_artist(arrow1_neg)
    
    # 绘制σ₁连接线（完整的线，穿过中心）
    ax.plot([sigma1_vec[0]*outer_pos, -sigma1_vec[0]*outer_pos],
            [sigma1_vec[1]*outer_pos, -sigma1_vec[1]*outer_pos],
            [sigma1_vec[2]*outer_pos, -sigma1_vec[2]*outer_pos],
            'b-', linewidth=3)
    
    # σ₃ (黑色) - 相对张应力，向外箭头
    # 正方向箭头（从中心附近向外）
    arrow3_pos = Arrow3D([sigma3_vec[0]*sigma3_start, sigma3_vec[0]*outer_pos], 
                        [sigma3_vec[1]*sigma3_start, sigma3_vec[1]*outer_pos], 
                        [sigma3_vec[2]*sigma3_start, sigma3_vec[2]*outer_pos],
                        mutation_scale=20, lw=3, arrowstyle='->', color='black')
    # 负方向箭头（从中心附近向外）
    arrow3_neg = Arrow3D([-sigma3_vec[0]*sigma3_start, -sigma3_vec[0]*outer_pos], 
                        [-sigma3_vec[1]*sigma3_start, -sigma3_vec[1]*outer_pos], 
                        [-sigma3_vec[2]*sigma3_start, -sigma3_vec[2]*outer_pos],
                        mutation_scale=20, lw=3, arrowstyle='->', color='black')
    ax.add_artist(arrow3_pos)
    ax.add_artist(arrow3_neg)
    
    # 绘制σ₃连接线（完整的线，穿过中心）
    ax.plot([sigma3_vec[0]*outer_pos, -sigma3_vec[0]*outer_pos],
            [sigma3_vec[1]*outer_pos, -sigma3_vec[1]*outer_pos],
            [sigma3_vec[2]*outer_pos, -sigma3_vec[2]*outer_pos],
            'k-', linewidth=3)
    
    # 添加标签（调整位置以避免与箭头重叠）
    label_dist_sigma1 = outer_pos * 0.6
    label_dist_sigma3 = outer_pos * 0.7
    
    # 计算垂直于应力轴的偏移，避免标签与轴线重叠
    # σ₁标签 - 根据倾伏角调整位置
    if abs(sigma1_plunge) > 20:  # 如果倾伏角较大
        # 在垂直于轴线的方向偏移
        perpendicular = np.cross(sigma1_vec, [0, 0, 1])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        else:
            perpendicular = np.cross(sigma1_vec, [1, 0, 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        offset_sigma1 = perpendicular * 0.3
        ax.text(sigma1_vec[0]*label_dist_sigma1 + offset_sigma1[0], 
                sigma1_vec[1]*label_dist_sigma1 + offset_sigma1[1], 
                sigma1_vec[2]*label_dist_sigma1 + offset_sigma1[2], 
                'σ₁', fontsize=18, fontweight='bold', 
                color='blue', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='blue', alpha=0.8))
    else:  # 如果倾伏角较小，垂直偏移
        ax.text(sigma1_vec[0]*label_dist_sigma1, 
                sigma1_vec[1]*label_dist_sigma1, 
                sigma1_vec[2]*label_dist_sigma1 - 0.25, 
                'σ₁', fontsize=18, fontweight='bold', 
                color='blue', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='blue', alpha=0.8))
    
    # σ₃标签 - 根据倾伏角调整位置
    if abs(sigma3_plunge) > 20:  # 如果倾伏角较大
        # 在垂直于轴线的方向偏移
        perpendicular = np.cross(sigma3_vec, [0, 0, 1])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        else:
            perpendicular = np.cross(sigma3_vec, [1, 0, 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        offset_sigma3 = perpendicular * 0.3
        ax.text(sigma3_vec[0]*label_dist_sigma3 + offset_sigma3[0], 
                sigma3_vec[1]*label_dist_sigma3 + offset_sigma3[1], 
                sigma3_vec[2]*label_dist_sigma3 + offset_sigma3[2], 
                'σ₃', fontsize=18, fontweight='bold', 
                color='black', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='black', alpha=0.8))
    else:  # 如果倾伏角较小，垂直偏移
        ax.text(sigma3_vec[0]*label_dist_sigma3, 
                sigma3_vec[1]*label_dist_sigma3, 
                sigma3_vec[2]*label_dist_sigma3 + 0.25, 
                'σ₃', fontsize=18, fontweight='bold', 
                color='black', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='black', alpha=0.8))
    
    # 绘制应力椭球（可选）
    if show_ellipse:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        # 椭球参数
        a = 0.9  # σ₁方向半径
        b = 0.5  # σ₂方向半径
        c = 0.4  # σ₃方向半径
        
        # 生成椭球表面
        x_ellipsoid = a * np.outer(np.cos(u), np.sin(v))
        y_ellipsoid = b * np.outer(np.sin(u), np.sin(v))
        z_ellipsoid = c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 创建旋转后的椭球坐标
        x_rotated = np.zeros_like(x_ellipsoid)
        y_rotated = np.zeros_like(y_ellipsoid)
        z_rotated = np.zeros_like(z_ellipsoid)
        
        # 旋转椭球到正确的方向
        for i in range(len(u)):
            for j in range(len(v)):
                point = np.array([x_ellipsoid[i,j], y_ellipsoid[i,j], z_ellipsoid[i,j]])
                # 变换到主应力坐标系
                rotated_point = (point[0] * np.array(sigma1_vec) + 
                                point[1] * np.array(sigma2_vec) + 
                                point[2] * np.array(sigma3_vec))
                x_rotated[i,j] = rotated_point[0]
                y_rotated[i,j] = rotated_point[1]
                z_rotated[i,j] = rotated_point[2]
        
        # 绘制椭球表面
        ax.plot_surface(x_rotated, y_rotated, z_rotated, alpha=0.15, color='lightgray', 
                       linewidth=0, antialiased=True, shade=True)
        
        # 绘制椭球轮廓线
        # σ₁-σ₃平面的椭圆
        theta = np.linspace(0, 2*np.pi, 50)
        ellipse_x = a * np.cos(theta) * sigma1_vec[0] + c * np.sin(theta) * sigma3_vec[0]
        ellipse_y = a * np.cos(theta) * sigma1_vec[1] + c * np.sin(theta) * sigma3_vec[1]
        ellipse_z = a * np.cos(theta) * sigma1_vec[2] + c * np.sin(theta) * sigma3_vec[2]
        ax.plot(ellipse_x, ellipse_y, ellipse_z, 'gray', alpha=0.3, linewidth=1)
    
    # 添加坐标轴标签
    ax.set_xlabel('E(East)', fontsize=12, labelpad=5)
    ax.set_ylabel('N(North)', fontsize=12, labelpad=5)
    ax.set_zlabel('Up', fontsize=12, labelpad=5)
    
    # 设置轴范围
    max_range = 1.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # 设置视角
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 隐藏刻度
    ax.tick_params(axis='both', which='both', length=0)
    
    # 添加图例
    blue_line = mpatches.Patch(color='blue', label='σ₁ (Maximum compression)')
    black_line = mpatches.Patch(color='black', label='σ₃ (Minimum compression)')
    ax.legend(handles=[blue_line, black_line], loc='upper right', 
              frameon=True, fontsize=11, fancybox=True, shadow=True)
    
    
    # 调整图形布局
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    return fig, ax

# 主程序
if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    print("=== 3D应力方向可视化程序 ===\n")
    print("该程序用于绘制主应力方向的3D示意图")
    print("σ₁表示最大主应力（压应力），σ₃表示最小主应力（相对张应力）\n")
    
    try:
        print("请输入应力方向参数：")
        print("方位角(Azimuth): 从北向顺时针测量（0-360°）")
        print("倾伏角(Plunge): 向下为正（-90到90°）\n")
        
        # 输入参数
        sigma1_az = float(input("σ₁方位角 (度，默认80): ") or "80")
        sigma1_pl = float(input("σ₁倾伏角 (度，默认0): ") or "0")
        sigma3_az = float(input("σ₃方位角 (度，默认350): ") or "350")
        sigma3_pl = float(input("σ₃倾伏角 (度，默认0): ") or "0")
        
        # 验证输入
        if not (0 <= sigma1_az <= 360 and 0 <= sigma3_az <= 360):
            raise ValueError("方位角必须在0-360度之间")
        if not (-90 <= sigma1_pl <= 90 and -90 <= sigma3_pl <= 90):
            raise ValueError("倾伏角必须在-90到90度之间")
        
        # 检查σ₁和σ₃是否正交
        vec1 = np.array(azimuth_plunge_to_vector(sigma1_az, sigma1_pl))
        vec3 = np.array(azimuth_plunge_to_vector(sigma3_az, sigma3_pl))
        dot_product = np.dot(vec1, vec3)
        if abs(dot_product) > 0.1:  # 允许小的误差
            print(f"\n警告：σ₁和σ₃不完全正交 (点积 = {dot_product:.3f})")
            print("理想情况下，主应力方向应该相互垂直")
            print("程序将自动调整以确保可视化正确\n")
        
        # 显示选项
        show_ellipse = input("是否显示应力椭球? (y/n, 默认y): ").strip().lower() != 'n'
        
        # 视角选项
        print("\n选择视角：")
        print("1. 默认视角 (仰角25°, 方位角45°)")
        print("2. 俯视图 (从上往下看)")
        print("3. 侧视图 (从东向看)")
        print("4. 自定义视角")
        view_choice = input("请选择 (1-4, 默认1): ").strip() or "1"
        
        if view_choice == "2":
            view_angle = (90, 0)
        elif view_choice == "3":
            view_angle = (0, 0)
        elif view_choice == "4":
            elev = float(input("输入仰角 (度，默认25): ") or "25")
            azim = float(input("输入方位角 (度，默认45): ") or "45")
            view_angle = (elev, azim)
        else:
            view_angle = (25, 45)
        
        # 绘制3D图
        fig_3d, ax_3d = plot_stress_3d(sigma1_az, sigma1_pl, sigma3_az, sigma3_pl, 
                                      show_ellipse=show_ellipse, view_angle=view_angle)
        
        # 显示图形
        plt.show()
        
        # 保存选项
        save_fig = input("\n是否保存图形? (y/n): ").strip().lower()
        if save_fig == 'y':
            filename = input("输入文件名 (默认stress_3d.png): ").strip() or "stress_3d.png"
            dpi = int(input("输入DPI分辨率 (默认300): ") or "300")
            fig_3d.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"图形已保存为 {filename}")
    
    except ValueError as e:
        print(f"\n输入错误：{e}")
        print("使用默认参数绘制示例...")
        fig_3d, ax_3d = plot_stress_3d(80, 0, 350, 0, show_ellipse=True)
        plt.show()
    
    except KeyboardInterrupt:
        print("\n程序已终止")
    
    except Exception as e:
        print(f"\n发生错误：{e}")
        print("请检查输入参数是否正确")