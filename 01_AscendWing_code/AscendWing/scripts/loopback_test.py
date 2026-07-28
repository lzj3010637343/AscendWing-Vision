#!/usr/bin/env python3
"""ttyAMA3 串口物理回环测试。

需要将串口的 TX/RX 引脚短接，发出去的数据应原样回到 RX。
用法:
    ./loopback_test.py                       # 默认 /dev/ttyAMA3 @57600
    ./loopback_test.py --port /dev/ttyAMA3 --baud 57600 --rounds 20 --size 64
"""
import argparse
import serial
import sys
import time


def make_payload(size: int, rnd: int) -> bytes:
    """生成本轮测试数据: 帧头 + 轮次号 + 递增字节 + 帧尾, 便于发现错位。"""
    body = bytes((i & 0xFF) for i in range(size))
    head = b"\xAA\x55" + rnd.to_bytes(2, "little")
    tail = b"\x55\xAA"
    inner = head + body[len(head) + len(tail):]  # 保持总长 == size
    return inner + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyAMA3")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--size", type=int, default=64, help="每轮发送字节数")
    ap.add_argument("--timeout", type=float, default=0.5, help="每轮读超时(秒)")
    args = ap.parse_args()

    print(f"[loopback] {args.port} @ {args.baud}, {args.rounds} 轮 x {args.size}B, 读超时 {args.timeout}s")
    try:
        s = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except Exception as e:
        print(f"[loopback] 打开串口失败: {e}")
        return 2

    s.reset_input_buffer()
    s.reset_output_buffer()

    ok_rounds = 0
    bad_bytes = 0
    total_bytes = 0
    latencies = []

    for r in range(args.rounds):
        payload = make_payload(args.size, r)
        s.reset_input_buffer()          # 清掉残留, 保证读到的是本轮回环数据
        t0 = time.perf_counter()
        s.write(payload)
        s.flush()
        got = s.read(len(payload))
        dt = time.perf_counter() - t0

        total_bytes += len(payload)
        if len(got) != len(payload):
            print(f"  轮{r:2d}: 收回 {len(got)}/{len(payload)} 字节 -> 失败 (检查 TX/RX 是否短接)")
            bad_bytes += abs(len(payload) - len(got))
            continue
        mism = sum(1 for a, b in zip(payload, got) if a != b)
        if mism == 0:
            ok_rounds += 1
            latencies.append(dt)
            print(f"  轮{r:2d}: OK  {len(got)}B 往返 {dt*1000:.2f}ms")
        else:
            bad_bytes += mism
            print(f"  轮{r:2d}: 数据错 {mism} 字节")

    s.close()

    print("-" * 50)
    print(f"成功 {ok_rounds}/{args.rounds} 轮, 字节错误 {bad_bytes}/{total_bytes}")
    if latencies:
        print(f"平均往返延迟 {sum(latencies)/len(latencies)*1000:.2f}ms (min {min(latencies)*1000:.2f} / max {max(latencies)*1000:.2f})")

    if ok_rounds == args.rounds and bad_bytes == 0:
        print("[loopback] PASS: 回环链路正常")
        return 0
    print("[loopback] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
