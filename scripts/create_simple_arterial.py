from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("scenarios/simple_arterial")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def create_nodes(intersections: int, spacing: float, downstream: float) -> str:
    lines = ["<nodes>"]
    lines.append('    <node id="n0" x="0.0" y="0.0" type="priority"/>')
    for idx in range(1, intersections + 1):
        lines.append(
            f'    <node id="j{idx}" x="{idx * spacing:.1f}" y="0.0" '
            f'type="traffic_light"/>'
        )
    end_idx = intersections + 1
    end_position = intersections * spacing + downstream
    lines.append(
        f'    <node id="n{end_idx}" x="{end_position:.1f}" y="0.0" '
        f'type="priority"/>'
    )
    lines.append("</nodes>")
    return "\n".join(lines)


def create_edges(intersections: int, speed: float) -> str:
    lines = ["<edges>"]
    for idx in range(intersections + 1):
        from_node = "n0" if idx == 0 else f"j{idx}"
        to_node = f"j{idx + 1}" if idx < intersections else f"n{idx + 1}"
        lines.append(
            f'    <edge id="e{idx}" from="{from_node}" to="{to_node}" '
            f'numLanes="1" speed="{speed:.2f}"/>'
        )
    lines.append("</edges>")
    return "\n".join(lines)


def create_tls(intersections: int, green: float, red: float) -> str:
    lines = ["<additional>"]
    for idx in range(1, intersections + 1):
        lines.extend(
            [
                f'    <tlLogic id="j{idx}" type="static" programID="arterial_90s" offset="0">',
                f'        <phase duration="{green:g}" state="G"/>',
                f'        <phase duration="{red:g}" state="r"/>',
                "    </tlLogic>",
            ]
        )
    lines.append("</additional>")
    return "\n".join(lines)


def create_routes(
    intersections: int,
    speed: float,
    flow_probability: float,
    use_glosa: bool,
) -> str:
    edges = " ".join(f"e{idx}" for idx in range(intersections + 1))
    ego_type = "ego_glosa_type" if use_glosa else "ego_type"
    glosa_params = ""
    if use_glosa:
        glosa_params = """
            <param key="has.glosa.device" value="true"/>
            <param key="device.glosa.range" value="300"/>
            <param key="device.glosa.min-speed" value="2.0"/>
            <param key="device.glosa.max-speedfactor" value="1.1"/>
            <param key="device.glosa.add-switchtime" value="2.0"/>"""

    return f"""
<routes>
    <vType id="human" accel="2.0" decel="4.5" sigma="0.5" length="5.0"
           minGap="2.5" maxSpeed="{speed:.2f}" carFollowModel="Krauss"/>
    <vType id="{ego_type}" accel="2.0" decel="4.5" sigma="0.0" length="5.0"
           minGap="2.5" maxSpeed="{speed:.2f}" carFollowModel="Krauss">{glosa_params}
    </vType>

    <route id="arterial_route" edges="{edges}"/>

    <flow id="background" type="human" route="arterial_route"
          begin="0" end="900" probability="{flow_probability:g}" departSpeed="max"/>
</routes>
"""


def create_cfg(route_file: str, step_length: float) -> str:
    return f"""
<configuration>
    <input>
        <net-file value="simple_arterial.net.xml"/>
        <route-files value="{route_file}"/>
        <additional-files value="simple_arterial.tls.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="900"/>
        <step-length value="{step_length:g}"/>
    </time>
</configuration>
"""


def create_sumo_inputs(
    output_dir: Path,
    intersections: int,
    spacing: float,
    downstream: float,
    step_length: float,
    speed: float,
    flow_probability: float,
    green: float,
    red: float,
):
    write_text(
        output_dir / "simple_arterial.nod.xml",
        create_nodes(intersections, spacing, downstream),
    )
    write_text(output_dir / "simple_arterial.edg.xml", create_edges(intersections, speed))
    write_text(output_dir / "simple_arterial.tls.xml", create_tls(intersections, green, red))
    write_text(
        output_dir / "simple_arterial.rou.xml",
        create_routes(intersections, speed, flow_probability, use_glosa=False),
    )
    write_text(
        output_dir / "simple_arterial_glosa.rou.xml",
        create_routes(intersections, speed, flow_probability, use_glosa=True),
    )
    write_text(
        output_dir / "simple_arterial.sumocfg",
        create_cfg("simple_arterial.rou.xml", step_length),
    )
    write_text(
        output_dir / "simple_arterial_glosa.sumocfg",
        create_cfg("simple_arterial_glosa.rou.xml", step_length),
    )


def build_network(output_dir: Path):
    command = [
        "netconvert",
        "--node-files",
        str(output_dir / "simple_arterial.nod.xml"),
        "--edge-files",
        str(output_dir / "simple_arterial.edg.xml"),
        "--output-file",
        str(output_dir / "simple_arterial.net.xml"),
        "--no-turnarounds",
        "true",
    ]
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single-lane SUMO arterial scenario."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--intersections", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=300.0)
    parser.add_argument("--downstream", type=float, default=50.0)
    parser.add_argument("--step-length", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=13.89)
    parser.add_argument("--flow-probability", type=float, default=0.0625)
    parser.add_argument("--green", type=float, default=45.0)
    parser.add_argument("--red", type=float, default=45.0)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run netconvert to create simple_arterial.net.xml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    create_sumo_inputs(
        output_dir=output_dir,
        intersections=args.intersections,
        spacing=args.spacing,
        downstream=args.downstream,
        step_length=args.step_length,
        speed=args.speed,
        flow_probability=args.flow_probability,
        green=args.green,
        red=args.red,
    )
    if args.build:
        build_network(output_dir)
    print(f"wrote SUMO arterial inputs to {output_dir}")
    print(
        f"intersections={args.intersections} spacing={args.spacing:g}m "
        f"downstream={args.downstream:g}m step_length={args.step_length:g}s"
    )
    if not args.build:
        print("run again with --build, or run netconvert manually, to create the .net.xml")


if __name__ == "__main__":
    main()
