import onnx
from collections import Counter

paths = {
    "NEW (head_out_new)":    "/tmp/onnx_inspect/AscendWing-export/code/yolo26/yolo26s_head_out_new.onnx",
    "PRUNED (head_out_pruned)": "/tmp/onnx_inspect/AscendWing-export/code/yolo26/yolo26s_head_out_pruned.onnx",
    "TOPK (topk)":           "/tmp/onnx_inspect/AscendWing-export/code/yolo26/yolo26s_topk.onnx",
}

def dims(vi):
    tt = vi.type.tensor_type
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in tt.shape.dim]

for tag, p in paths.items():
    m = onnx.load(p)
    g = m.graph
    print("="*72)
    print(f"### {tag}")
    print("opset:", [(o.domain, o.version) for o in m.opset_import])
    print(f"nodes: {len(g.node)}")
    print("op counts:", dict(Counter(n.op_type for n in g.node)))
    print("-- inputs --")
    for i in g.input:
        print(f"   {i.name}: {dims(i)}")
    print("-- outputs --")
    for o in g.output:
        print(f"   {o.name}: {dims(o)}")
    sigs = [n.name for n in g.node if n.op_type == "Sigmoid"]
    print(f"-- Sigmoid nodes: {len(sigs)} ->", sigs[:6])
    for n in g.node:
        if n.op_type == "TopK":
            attrs = {a.name: (a.i if a.type == 2 else (list(a.ints) if a.type == 7 else None)) for a in n.attribute}
            print(f"-- TopK: in={[i[:20] for i in n.input]} out={[o[:20] for o in n.output]} attrs={attrs}")
    for n in g.node:
        if n.op_type == "Concat":
            attrs = {a.name: a.i for a in n.attribute}
            print(f"-- Concat: in={[i[:22] for i in n.input]} out={[o[:22] for o in n.output]} axis={attrs.get('axis')}")
    print("-- last 24 nodes (op | name | inputs -> outputs) --")
    for n in g.node[-24:]:
        print(f"   {n.op_type:11s} {(n.name or '')[:38]:38s} {[i[:16] for i in n.input]} -> {[o[:16] for o in n.output]}")
    print()
