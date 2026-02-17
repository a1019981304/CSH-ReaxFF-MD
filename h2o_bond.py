# -*- coding: utf-8 -*-
import os
import math
from collections import defaultdict

# =============================================================================
# --- 全局配置参数 (Global Configuration Parameters) ---
# =============================================================================

# --- 文件路径 (File Paths) ---
# 按需修改你的工作目录
WORK_PATH = "f:\\MD\\python-code\\水键转换\\H2O--5A\\"
BOND_FILE = os.path.join(WORK_PATH, "all_frames_bonds.reaxff")
TRAJ_FILE = os.path.join(WORK_PATH, "all_frames_trajectory.lammpstrj")
OUTPUT_PREFIX = "data_frame_classified"

# --- 模拟参数 (Simulation Parameters) ---
N_FRAME = 200

# --- 成键判断标准 (Bonding Criteria) ---
BOND_CRITERIA = {
    'Si-O': {'bo': 0.3, 'dist': 2.0},
    'O-H':  {'bo': 0.3, 'dist': 1.23},
    'Ca-O': {'bo': 0.3, 'dist': 3.0},
}

#O–H 几何兜底半径（Å），建议 1.25；可微调 1.22~1.25
OH_GEOM_FALLBACK = 1.23


INCLUDE_GEOM_OH_IN_BONDS = True
# --- 原始原子类型定义 (Initial Atom Types from Simulation, 有水界面模型) ---
# 1  Ca-CSH
# 2  st-CSH
# 3  ob-CSH
# 4  obos-CSH
# 5  o2*-界面水氧
# 6  h1o-界面水氢
# 7  o*-CSH
# 8  h*-CSH
# 9  oz-SiO2的氧
# 10 si-SiO2的硅
# 11 ho-SiO2的羟基氢


INITIAL_TYPES = {
    'Ca': 1,
    'CSH_Si': 2,
    'CSH_O': [3, 4, 7],     # CSH 中的氧
    'IF_Water_O': 5,        # 界面水 氧
    'IF_Water_H': 6,        # 界面水 氢
    'CSH_H': 8,             # CSH 内部氢
    'SiO2_O': 9,
    'SiO2_Si': 10,
    'SiO2_H': 11,
}

# 便于后续统一筛选
O_INIT_TYPES  = [3, 4, 5, 7, 9]   # 所有氧的初始类型
H_INIT_TYPES  = [6, 8, 11]        # 所有氢的初始类型
SI_INIT_TYPES = [2, 10]           # 所有硅的初始类型

# --- 最终输出的原子类型定义 ---
FINAL_ATOM_INFO = [
    # --- 钙和硅 ---
    {"mass": 40.079800, "name": "Ca"},                    # Type 1
    {"mass": 28.085500, "name": "CSH_Si"},                # Type 2
    # --- CSH 氧类型 ---
    {"mass": 15.999400, "name": "O_Bridging_CSH"},        # Type 3
    {"mass": 15.999400, "name": "O_Water_CSH"},           # Type 4
    {"mass": 15.999400, "name": "O_Hydroxyl_CSH"},        # Type 5
    {"mass": 15.999400, "name": "O_NBO_CSH"},             # Type 6
    {"mass": 15.999400, "name": "O_Unclassified_CSH"},    # Type 7
    # --- SiO2 氧类型 ---
    {"mass": 15.999400, "name": "O_SiO2_Structural"},     # Type 8
    {"mass": 15.999400, "name": "O_SiO2_Surface"},        # Type 9
    # --- SiO2 硅类型 ---
    {"mass": 28.085500, "name": "Si_Bulk"},               # Type 10
    {"mass": 28.085500, "name": "Si_Surface"},            # Type 11
    # --- 氢类型 ---
    {"mass": 1.007970,  "name": "H_Water_CSH"},           # Type 12
    {"mass": 1.007970,  "name": "H_Hydroxyl_CSH"},        # Type 13
    {"mass": 1.007970,  "name": "H_SiO2_Hydroxyl"},       # Type 14
    {"mass": 1.007970,  "name": "H_Unbonded_CSH"},        # Type 15
    {"mass": 1.007970,  "name": "H_Unbonded_SiO2"},       # Type 16
    # --- 界面水新增 ---
    {"mass": 15.999400, "name": "O_Water_IF"},            # Type 17
    {"mass": 15.999400, "name": "O_Water_IF_CaCoord"},    # Type 18
    {"mass": 1.007970,  "name": "H_Water_IF"},            # Type 19
]

# --- 自动生成类型映射 ---
FINAL_TYPE_MAP = {info["name"]: i + 1 for i, info in enumerate(FINAL_ATOM_INFO)}
NUM_FINAL_TYPES = len(FINAL_ATOM_INFO)

print(f"总共定义了 {NUM_FINAL_TYPES} 种最终原子类型")

# =============================================================================
# --- 辅助函数 (Helper Functions) ---
# =============================================================================

def calculate_distance(pos1, pos2, box_bounds):
    """计算考虑周期性边界条件的两个原子间的最小镜像距离"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]

    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]

    dx -= lx * round(dx / lx)
    dy -= ly * round(dy / ly)
    dz -= lz * round(dz / lz)

    return math.sqrt(dx*dx + dy*dy + dz*dz)

def distance_sq_pbc(p1, p2, box_bounds):
    dx = p1[0] - p2[0]; dy = p1[1] - p2[1]; dz = p1[2] - p2[2]
    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]
    dx -= lx * round(dx / lx)
    dy -= ly * round(dy / ly)
    dz -= lz * round(dz / lz)
    return dx*dx + dy*dy + dz*dz  

# =============================================================================
# --- 核心分类逻辑 (Core Classification Logic) ---
# =============================================================================

def classify_atoms_and_bonds(atom_data_full, box_bounds, bonds_data):
    """
    有水界面模型分类：
    - IF 界面水氧（初始type=5）按 nSi_total、nCa、nH 判：
      nSi_total≥2→O_Bridging_CSH；nSi_total=1→(nH≥1)O_Hydroxyl_CSH/(nH=0)O_NBO_CSH；
      nSi_total=0 & nCa≥1→(nH≥2)O_Water_IF_CaCoord/(nH=1)O_Hydroxyl_CSH/(nH=0)O_NBO_CSH；
      nSi_total=0 & nCa=0→O_Water_IF
    - IF 界面水氢（初始type=6）：若找不到母氧，兜底 H_Water_IF
    - 其它逻辑与阈值不变
    """
    # 0) 初始化
# 本地阈值与全体 O/H 列表（供兜底几何搜索用）
    OH_BO   = BOND_CRITERIA['O-H']['bo']
    OH_DIST = BOND_CRITERIA['O-H']['dist']
    OH_GEOM = OH_GEOM_FALLBACK

    OH_GEOM2 = OH_GEOM * OH_GEOM

    ALL_O_IDS = [aid for aid, d in atom_data_full.items() if d['type'] in O_INIT_TYPES]
    ALL_H_IDS = [aid for aid, d in atom_data_full.items() if d['type'] in H_INIT_TYPES]

    classified_atoms = {aid: {**d, 'final_type': 0} for aid, d in atom_data_full.items()}

    # -------------------------------------------------------------
    # 1) 构建无向边表 edge_best：每对(i,j)只保留 BO 最大、若相等取距离更小
    # -------------------------------------------------------------
    edge_best = {}
    for atom_id, data in bonds_data.items():
        if atom_id not in atom_data_full:
            continue
        pos1 = atom_data_full[atom_id]['pos']
        for partner_id, bo in data.get('bonds', []):
            if partner_id not in atom_data_full:
                continue
            pos2 = atom_data_full[partner_id]['pos']
            dist = calculate_distance(pos1, pos2, box_bounds)
            a, b = (atom_id, partner_id) if atom_id < partner_id else (partner_id, atom_id)
            rec = edge_best.get((a, b))
            if rec is None or (bo > rec['bo']) or (bo == rec['bo'] and dist < rec['dist']):
                edge_best[(a, b)] = {'bo': bo, 'dist': dist}

    # 由边表生成对称邻接列表
    bonded_partners = defaultdict(list)
    for (a, b), rec in edge_best.items():
        bo = rec['bo']; dist = rec['dist']
        ta = atom_data_full[a]['type']; tb = atom_data_full[b]['type']
        bonded_partners[a].append({'id': b, 'bo': bo, 'dist': dist, 'type': tb})
        bonded_partners[b].append({'id': a, 'bo': bo, 'dist': dist, 'type': ta})


    # -------------------------------------------------------------
    # 2) 统计每个氧的邻接（按阈值）+ 几何兜底
    # -------------------------------------------------------------
    oxygen_neighbors = {}
    for aid, d in atom_data_full.items():
        itype = d['type']
        if itype not in O_INIT_TYPES:
            continue
        nSi_CSH = 0        # 初始化
        nSi_SiO2 = 0
        nCa = 0
        nH = 0

        # 先用邻接表统计（bo+dist 合格，几何兜底 dist≤OH_GEOM）
        for p in bonded_partners.get(aid, []):
            ptype = p['type']
            bo    = p['bo']
            dist  = p['dist']

            # Si–O
            if ptype in SI_INIT_TYPES:
                if bo >= BOND_CRITERIA['Si-O']['bo'] and dist <= BOND_CRITERIA['Si-O']['dist']:
                    if ptype == INITIAL_TYPES['CSH_Si']:
                        nSi_CSH += 1
                    else:
                        nSi_SiO2 += 1

            # Ca–O
            elif ptype == INITIAL_TYPES['Ca']:
                if bo >= BOND_CRITERIA['Ca-O']['bo'] and dist <= BOND_CRITERIA['Ca-O']['dist']:
                    nCa += 1

            # O–H（bo 判据 或 几何兜底）
            elif ptype in H_INIT_TYPES:
                cond_bo   = (bo >= OH_BO and dist <= OH_DIST)
                cond_geom = (dist <= OH_GEOM)  
                if cond_bo or cond_geom:
                    nH += 1

        # 若邻接未找到任何 H，再做一次全局几何兜底（找所有 H 中距离 ≤ OH_GEOM 的）
        if nH == 0:
            posO = atom_data_full[aid]['pos']
            zO = posO[2]
            for hid in ALL_H_IDS:
                posH = atom_data_full[hid]['pos']
                if abs(posH[2] - zO) > OH_GEOM:
                    continue
                if distance_sq_pbc(posO, posH, box_bounds) <= OH_GEOM2:
                    nH = 1
                    break

        # 统一写回
        oxygen_neighbors[aid] = {
            'nSi_CSH': nSi_CSH,
            'nSi_SiO2': nSi_SiO2,
            'nSi_total': nSi_CSH + nSi_SiO2,
            'nCa': nCa,
            'nH': nH
        }

    # -------------------------------------------------------------
    # 3) 第一遍：氧分类（互斥）
    # -------------------------------------------------------------
    # 3A) 界面水氧（初始类型 = 5）：按给定判据优先分类
    for aid, d in atom_data_full.items():
        if d['type'] != INITIAL_TYPES['IF_Water_O']:
            continue
        c = oxygen_neighbors.get(aid, {'nSi_total': 0, 'nCa': 0, 'nH': 0})
        if c['nSi_total'] >= 2:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Bridging_CSH"]
        elif c['nSi_total'] == 1:
            if c['nH'] >= 1:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Hydroxyl_CSH"]   # Si–O–H
            else:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_NBO_CSH"]       # Si–O(–Ca可有可无)
        elif c['nSi_total'] == 0 and c['nCa'] >= 1:
            if c['nH'] >= 2:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Water_IF_CaCoord"]  # Ca–(OH2)
            elif c['nH'] == 1:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Hydroxyl_CSH"]      # Ca–O–H
            else:  # nH = 0
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_NBO_CSH"]           # Ca–O
        elif c['nSi_total'] == 0 and c['nCa'] == 0:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Water_IF"]              # 统一合并
        else:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Unclassified_CSH"]

    # 3B) CSH 氧（初始类型 ∈ {3,4,7}）
    for aid, d in atom_data_full.items():
        if d['type'] not in INITIAL_TYPES['CSH_O']:
            continue
        # 若前一步（界面水氧）已分类则跳过（正常不会撞上）
        if classified_atoms[aid]['final_type'] != 0:
            continue
        c = oxygen_neighbors.get(aid, {'nSi_total': 0, 'nCa': 0, 'nH': 0})
        if c['nSi_total'] >= 2:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Bridging_CSH"]
        elif c['nSi_total'] == 0 and c['nH'] >= 2:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Water_CSH"]
        elif c['nH'] >= 1:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Hydroxyl_CSH"]
        elif c['nH'] == 0 and (c['nSi_total'] >= 1 or c['nCa'] >= 1):
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_NBO_CSH"]
        else:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_Unclassified_CSH"]

    # 3C) SiO2 氧（初始类型 = 9）：是否带H决定表面/结构
    for aid, d in atom_data_full.items():
        if d['type'] != INITIAL_TYPES['SiO2_O']:
            continue
        c = oxygen_neighbors.get(aid, {'nH': 0})
        if c['nH'] >= 1:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_SiO2_Surface"]
        else:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["O_SiO2_Structural"]

    # -------------------------------------------------------------
    # 4) 第二遍：硅分类（互斥）
    # -------------------------------------------------------------
    for aid, d in atom_data_full.items():
        itype = d['type']
        if itype == INITIAL_TYPES['CSH_Si']:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["CSH_Si"]
        elif itype == INITIAL_TYPES['SiO2_Si']:
            all_struct = True
            has_o = False
            for p in bonded_partners.get(aid, []):
                if p['type'] == INITIAL_TYPES['SiO2_O']:
                    bo = p['bo']; dist = p['dist']
                    if bo >= BOND_CRITERIA['Si-O']['bo'] and dist <= BOND_CRITERIA['Si-O']['dist']:
                        has_o = True
                        oid = p['id']
                        if classified_atoms[oid]['final_type'] != FINAL_TYPE_MAP["O_SiO2_Structural"]:
                            all_struct = False
            if has_o and all_struct:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["Si_Bulk"]
            else:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["Si_Surface"]

    # -------------------------------------------------------------
    # 5) 第二遍：氢分类（互斥；按母氧）
    # -------------------------------------------------------------
    for aid, d in atom_data_full.items():
        itype = d['type']
        if itype not in H_INIT_TYPES:
            continue
        best_oxygen = None
        best_rank = (999, float('inf'), -1.0)  # (优先级, 距离, -BO)：优先级 0 表示 bo+dist 合格，1 表示仅几何兜底
        posH = atom_data_full[aid]['pos']

        # 先在邻接表里选“最佳 O”：（bo+dist 合格优先，其次几何兜底；再按距离、再按 BO）
        for p in bonded_partners.get(aid, []):
            if p['type'] not in O_INIT_TYPES:
                continue
            bo   = p['bo']
            dist = p['dist']
            cond_bo   = (bo >= OH_BO and dist <= OH_DIST)
            cond_geom = (dist <= OH_GEOM)
            if cond_bo or cond_geom:
                # 打分：bo 合格优先；距离更近优先；BO 更大优先
                rank = (0 if cond_bo else 1, dist, -bo)
                if rank < best_rank:
                    best_rank = rank
                    best_oxygen = p['id']

        # 若邻接仍无，则做全局几何兜底（找所有 O 中最近且 ≤ OH_GEOM 的）
        if best_oxygen is None:
            nearest_oid = None
            nearest_d2 = OH_GEOM2  # 只关心 ≤ 阈值内的最近
            zH = posH[2]
            for oid in ALL_O_IDS:
                posO = atom_data_full[oid]['pos']
                if abs(posO[2] - zH) > OH_GEOM:
                    continue
                d2 = distance_sq_pbc(posH, posO, box_bounds)
                if d2 <= nearest_d2:
                    nearest_d2 = d2
                    nearest_oid = oid
            if nearest_oid is not None:
                best_oxygen = nearest_oid

        if best_oxygen is None:
            
            if itype == INITIAL_TYPES['IF_Water_H']:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Water_IF"]
            elif itype == INITIAL_TYPES['SiO2_H']:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Unbonded_SiO2"]
            else:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Unbonded_CSH"]
        else:
            host = classified_atoms[best_oxygen]['final_type']
            if host == FINAL_TYPE_MAP["O_Water_CSH"]:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Water_CSH"]
            elif host in [FINAL_TYPE_MAP["O_Water_IF"], FINAL_TYPE_MAP["O_Water_IF_CaCoord"]]:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Water_IF"]
            elif host in [FINAL_TYPE_MAP["O_Hydroxyl_CSH"], FINAL_TYPE_MAP["O_Bridging_CSH"], FINAL_TYPE_MAP["O_NBO_CSH"], FINAL_TYPE_MAP["O_Unclassified_CSH"]]:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Hydroxyl_CSH"]
            elif host == FINAL_TYPE_MAP["O_SiO2_Surface"]:
                classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_SiO2_Hydroxyl"]
            else:
                if itype == INITIAL_TYPES['IF_Water_H']:
                    classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Water_IF"]
                elif itype == INITIAL_TYPES['SiO2_H']:
                    classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Unbonded_SiO2"]
                else:
                    classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["H_Unbonded_CSH"]

    # -------------------------------------------------------------
    # 6) 钙
    # -------------------------------------------------------------
    for aid, d in atom_data_full.items():
        if d['type'] == INITIAL_TYPES['Ca']:
            classified_atoms[aid]['final_type'] = FINAL_TYPE_MAP["Ca"]

    # -------------------------------------------------------------
    # 7) 最终键列表（硬阈值过滤 + 去重）
    # -------------------------------------------------------------
    final_bonds = []
    processed_pairs = set()
    for (a, b), rec in edge_best.items():
        ta = atom_data_full[a]['type']; tb = atom_data_full[b]['type']
        bo = rec['bo']; dist = rec['dist']
        is_bond = False
        # Si–O
        if ((ta in SI_INIT_TYPES and tb in O_INIT_TYPES) or (tb in SI_INIT_TYPES and ta in O_INIT_TYPES)):
            is_bond = (bo >= BOND_CRITERIA['Si-O']['bo'] and dist <= BOND_CRITERIA['Si-O']['dist'])
        # O–H（若开启，几何兜底也视为成键）
        elif ((ta in O_INIT_TYPES and tb in H_INIT_TYPES) or (tb in O_INIT_TYPES and ta in H_INIT_TYPES)):
            cond_bo   = (bo >= OH_BO and dist <= OH_DIST)
            cond_geom = (dist <= OH_GEOM) if INCLUDE_GEOM_OH_IN_BONDS else False
            is_bond = (cond_bo or cond_geom)
        # Ca–O
        elif ((ta == INITIAL_TYPES['Ca'] and tb in O_INIT_TYPES) or (tb == INITIAL_TYPES['Ca'] and ta in O_INIT_TYPES)):
            is_bond = (bo >= BOND_CRITERIA['Ca-O']['bo'] and dist <= BOND_CRITERIA['Ca-O']['dist'])
        if is_bond:
            pair = (a, b)
            if pair not in processed_pairs:
                final_bonds.append(pair)
                processed_pairs.add(pair)

    return classified_atoms, final_bonds

# =============================================================================
# --- 文件写入和主函数 ---
# =============================================================================

def write_data_file(frame_idx, box_bounds, atom_data, all_bonds):
    """写入LAMMPS data文件"""
    output_path = os.path.join(WORK_PATH, f"{OUTPUT_PREFIX}_{frame_idx}.data")

    # 统计各类型原子数量
    type_counts = defaultdict(int)
    for atom_id, data in atom_data.items():
        final_type = data.get('final_type', 0)
        if final_type < 1 or final_type > NUM_FINAL_TYPES:
            print(f"  ❌ 错误: 原子 {atom_id} 的最终类型 {final_type} 超出范围")
            final_type = 1
            data['final_type'] = final_type
        type_counts[final_type] += 1

    print(f"  各类型原子数量:")
    for t in range(1, NUM_FINAL_TYPES + 1):
        if type_counts[t] > 0:
            print(f"    类型 {t:2d} ({FINAL_ATOM_INFO[t-1]['name']:20s}): {type_counts[t]:5d}")

    with open(output_path, 'w') as f:
        f.write("LAMMPS Data File from ReaxFF Classification\n\n")
        f.write(f"{len(atom_data)} atoms\n")
        f.write(f"{NUM_FINAL_TYPES} atom types\n")
        f.write(f"{len(all_bonds)} bonds\n")
        f.write("1 bond types\n\n")
        f.write(f"{box_bounds[0][0]:.6f} {box_bounds[0][1]:.6f} xlo xhi\n")
        f.write(f"{box_bounds[1][0]:.6f} {box_bounds[1][1]:.6f} ylo yhi\n")
        f.write(f"{box_bounds[2][0]:.6f} {box_bounds[2][1]:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")

        for i in range(1, NUM_FINAL_TYPES + 1):
            info = FINAL_ATOM_INFO[i-1]
            f.write(f"  {i}  {info['mass']:.6f}  # {info['name']}\n")

        f.write("\nAtoms\n\n")
        for atom_id, data in sorted(atom_data.items()):
            final_type = data['final_type']
            f.write(f"{atom_id} 1 {final_type} {data['charge']:.6f} "
                    f"{data['pos'][0]:.6f} {data['pos'][1]:.6f} {data['pos'][2]:.6f} "
                    f"{data['images'][0]} {data['images'][1]} {data['images'][2]}\n")

        if all_bonds:
            f.write("\nBonds\n\n")
            for idx, (a1, a2) in enumerate(all_bonds, 1):
                f.write(f"{idx} 1 {a1} {a2}\n")

    print(f"✅ 成功处理帧 {frame_idx}: 已生成 {os.path.basename(output_path)}")

def main():
    """主函数"""
    print(f"开始处理 {N_FRAME} 帧数据...")
    print(f"定义的最终原子类型数: {NUM_FINAL_TYPES}")

    try:
        with open(BOND_FILE, 'r') as bonds_f, open(TRAJ_FILE, 'r') as dump_f:
            frames_processed = 0

            while frames_processed < N_FRAME:
                print(f"\n--- 开始处理第 {frames_processed} 帧 ---")

                # --- 读取 trajectory 文件 ---
                line = dump_f.readline()
                while line and "ITEM: TIMESTEP" not in line:
                    line = dump_f.readline()
                if not line:
                    print(f"⚠️ trajectory文件结束，只处理了 {frames_processed} 帧")
                    break
                dump_f.readline()  # 跳过时间步值

                # 读取原子数
                line = dump_f.readline()
                while line and "ITEM: NUMBER OF ATOMS" not in line:
                    line = dump_f.readline()
                if not line:
                    break
                n_atom_dump = int(dump_f.readline().strip())

                # 读取盒子边界
                line = dump_f.readline()
                while line and "ITEM: BOX BOUNDS" not in line:
                    line = dump_f.readline()
                if not line:
                    break
                box_bounds = []
                for _ in range(3):
                    bounds_line = dump_f.readline()
                    if not bounds_line:
                        break
                    box_bounds.append(tuple(map(float, bounds_line.split())))
                if len(box_bounds) != 3:
                    break

                # 读取原子数据
                line = dump_f.readline()
                while line and "ITEM: ATOMS" not in line:
                    line = dump_f.readline()
                if not line:
                    break

                atom_pos_type_images = {}
                for _ in range(n_atom_dump):
                    line = dump_f.readline()
                    if not line:
                        break
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    ix, iy, iz = int(parts[5]), int(parts[6]), int(parts[7])
                    atom_pos_type_images[atom_id] = {
                        'type': atom_type,
                        'pos': (x, y, z),
                        'images': (ix, iy, iz)
                    }

                # --- 读取 bonds 文件 ---
                line = bonds_f.readline()
                while line and "# Timestep" not in line:
                    line = bonds_f.readline()
                if not line:
                    print(f"⚠️ bonds文件结束，只处理了 {frames_processed} 帧")
                    break

                # 跳过到粒子连接表
                while True:
                    line = bonds_f.readline()
                    if not line or "# Particle connection table" in line:
                        break
                if not line:
                    break
                bonds_f.readline()  # 跳过表头

                # 读取键合数据
                atom_bonds_charge = defaultdict(lambda: {'charge': 0.0, 'bonds': []})
                atoms_read = 0

                while atoms_read < n_atom_dump:
                    line = bonds_f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    try:
                        atom1_id = int(parts[0])
                        num_bonds = int(parts[2])
                        charge = float(parts[-1])
                        atom_bonds_charge[atom1_id]['charge'] = charge

                        if num_bonds > 0 and len(parts) >= 3 + 2 * num_bonds:
                            ids = [int(parts[3 + i]) for i in range(num_bonds)]
                            bos = [float(parts[3 + num_bonds + i]) for i in range(num_bonds)]
                            atom_bonds_charge[atom1_id]['bonds'] = list(zip(ids, bos))

                        atoms_read += 1
                    except (ValueError, IndexError):
                        continue

                # --- 合并数据并执行分类 ---
                combined_atom_data = {}
                for atom_id, data in atom_pos_type_images.items():
                    combined_atom_data[atom_id] = {
                        **data,
                        'charge': atom_bonds_charge.get(atom_id, {'charge': 0.0})['charge']
                    }

                if len(combined_atom_data) == n_atom_dump:
                    final_atom_data, final_bonds = classify_atoms_and_bonds(
                        combined_atom_data, box_bounds, atom_bonds_charge
                    )
                    write_data_file(frames_processed, box_bounds, final_atom_data, final_bonds)
                    frames_processed += 1
                else:
                    print(f"  ❌ 数据不完整，跳过此帧")

            print(f"\n处理完成，共处理了 {frames_processed} 帧")

    except FileNotFoundError as e:
        print(f"❌ 错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"❌ 发生意外错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()