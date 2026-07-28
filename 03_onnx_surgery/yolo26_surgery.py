import onnx
from onnx import helper, TensorProto
import numpy as np
from collections import Counter

SRC = './source_onnx/yolo26s_ball_opset11.onnx'
DST = './yolo26s_ball_topk.onnx'

NA = 8400   # anchors
NC = 1      # 乒乓球 nc=1
TK = 300    # top-k

m = onnx.load(SRC)
g = m.graph

# 1. 找检测头最后的 /model.23/Split（输出 bbox + cls）
split_node = None
for n in g.node:
    if n.op_type == 'Split' and n.name == '/model.23/Split':
        split_node = n
        break
assert split_node, '未找到 /model.23/Split'
bbox_out = split_node.output[0]   # /model.23/Split_output_0  bbox [1,8400,4]
cls_out  = split_node.output[1]   # /model.23/Split_output_1  cls  [1,8400,1]
print(f'连接点: bbox={bbox_out}, cls={cls_out}')

# 2. 反向 BFS：从 Split 输出向前追溯，保留 Split 及其所有前驱依赖（剪掉 Split 之后的官方头）
reachable = {bbox_out, cls_out}
keep_idx = set()
changed = True
while changed:
    changed = False
    for idx, n in enumerate(g.node):
        if idx in keep_idx:
            continue
        if any(o in reachable for o in n.output):
            keep_idx.add(idx)
            for inp in n.input:
                reachable.add(inp)
            changed = True
# 按原顺序保留（维持拓扑序）
keep = [g.node[idx] for idx in range(len(g.node)) if idx in keep_idx]
print(f'剪枝: {len(g.node)} -> {len(keep)} 节点（剪掉 {len(g.node)-len(keep)} 个官方头节点）')

# 3. 重置 node 为保留节点
del g.node[:]
g.node.extend(keep)

# 4. 清空 output，准备接子图
del g.output[:]

# 5. 子图常量 initializer（int64 shape 张量）
existing = {x.name for x in g.initializer}
def add_init(name, arr):
    if name not in existing:
        g.initializer.append(onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name))
        existing.add(name)

add_init('K_val',        [TK])
add_init('scores_shape', [NA])
add_init('rs_bbox',      [NA, 4])
add_init('rs_cls',       [NA, NC])
add_init('rs_cls_out',   [TK, NC])
add_init('rs_val_out',   [TK, 1])
add_init('rs_out',       [1, TK, 6])

# 6. 追加 14 节点 1D-gather TopK 子图（全部 axis=0 的 1D Gather，310B 可编）
sub = [
    helper.make_node('ReduceMax', [cls_out],                ['scores_raw'],    name='topk_rmax',         axes=[2]),
    helper.make_node('Reshape',   ['scores_raw','scores_shape'], ['scores_1d'], name='topk_rs_score'),
    helper.make_node('TopK',      ['scores_1d','K_val'],    ['topk_vals','topk_idx'], name='topk_node'),
    helper.make_node('Cast',      ['topk_idx'],             ['topk_idx_i32'],  name='topk_cast_idx', to=TensorProto.INT32),
    helper.make_node('Reshape',   [bbox_out,'rs_bbox'],     ['bbox_2d'],       name='topk_rs_bbox'),
    helper.make_node('Reshape',   [cls_out,'rs_cls'],       ['cls_2d'],        name='topk_rs_cls'),
    helper.make_node('ArgMax',    ['cls_2d'],               ['cls_ids'],       name='topk_argmax',  axis=1),
    helper.make_node('Gather',    ['bbox_2d','topk_idx_i32'], ['bbox_topk'],  name='topk_gather_bbox', axis=0),
    helper.make_node('Gather',    ['cls_ids','topk_idx_i32'],  ['cls_topk'],   name='topk_gather_cls',  axis=0),
    helper.make_node('Reshape',   ['cls_topk','rs_cls_out'], ['cls_topk_2d'],  name='topk_rs_cls_out'),
    helper.make_node('Reshape',   ['topk_vals','rs_val_out'],['vals_2d'],      name='topk_rs_val_out'),
    helper.make_node('Cast',      ['cls_topk_2d'],          ['cls_topk_f'],    name='topk_cast_cls', to=TensorProto.FLOAT),
    helper.make_node('Concat',    ['cls_topk_f','bbox_topk','vals_2d'], ['output0'], name='topk_concat', axis=1),
    helper.make_node('Reshape',   ['output0','rs_out'],     ['dets'],          name='topk_rs_out'),
]
g.node.extend(sub)

# 7. 设置输出 dets [1,300,6]
g.output.append(helper.make_tensor_value_info('dets', TensorProto.FLOAT, [1, TK, 6]))

# 8. 清理无用 initializer（官方头遗留常量，不再被任何节点引用）
keep_names = set()
for n in g.node:
    for i in n.input:
        keep_names.add(i)
for i in g.input:
    keep_names.add(i.name)
new_init = [x for x in g.initializer if x.name in keep_names]
print(f'initializer: {len(g.initializer)} -> {len(new_init)}')
del g.initializer[:]
g.initializer.extend(new_init)

# 9. 保存 + 验证
onnx.save(m, DST)
print(f'\n保存: {DST}')
print(f'最终节点: {len(g.node)}')
c = Counter(n.op_type for n in g.node)
print(f'算子分布: {dict(c)}')
print(f'310B不支持算子残留: GatherElements={c.get("GatherElements",0)} Tile={c.get("Tile",0)} Mod={c.get("Mod",0)}')
print(f'输出: {[(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]}')

# onnx checker
try:
    onnx.checker.check_model(m)
    print('onnx.checker: PASS ✅')
except Exception as e:
    print(f'onnx.checker: FAIL - {e}')
