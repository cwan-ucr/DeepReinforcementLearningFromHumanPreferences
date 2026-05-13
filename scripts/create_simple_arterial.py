from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("scenarios/simple_arterial")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def create_sumo_inputs(output_dir: Path):
    node_xml = """
    <nodes>
        <node id="n0" x="0.0" y="0.0" type="priority"/>
        <node id="j1" x="300.0" y="0.0" type="traffic_light"/>
        <node id="j2" x="600.0" y="0.0" type="traffic_light"/>
        <node id="j3" x="900.0" y="0.0" type="traffic_light"/>
        <node id="n4" x="1200.0" y="0.0" type="priority"/>
    </nodes>
    """

    edge_xml = """
    <edges>
        <edge id="e0" from="n0" to="j1" numLanes="1" speed="13.89"/>
        <edge id="e1" from="j1" to="j2" numLanes="1" speed="13.89"/>
        <edge id="e2" from="j2" to="j3" numLanes="1" speed="13.89"/>
        <edge id="e3" from="j3" to="n4" numLanes="1" speed="13.89"/>
    </edges>
    """

    route_xml = """
    <routes>
        <vType id="human" accel="2.0" decel="4.5" sigma="0.5" length="5.0"
               minGap="2.5" maxSpeed="13.89" carFollowModel="Krauss"/>
        <vType id="ego_type" accel="2.0" decel="4.5" sigma="0.0" length="5.0"
               minGap="2.5" maxSpeed="13.89" carFollowModel="Krauss"/>

        <route id="arterial_route" edges="e0 e1 e2 e3"/>

        <flow id="background" type="human" route="arterial_route"
              begin="0" end="900" period="8" departSpeed="max"/>
        <vehicle id="ego" type="ego_type" route="arterial_route"
                 depart="5" departSpeed="0"/>
    </routes>
    """

    tls_xml = """
    <additional>
        <tlLogic id="j1" type="static" programID="arterial_90s" offset="0">
            <phase duration="45" state="G"/>
            <phase duration="45" state="r"/>
        </tlLogic>
        <tlLogic id="j2" type="static" programID="arterial_90s" offset="0">
            <phase duration="45" state="G"/>
            <phase duration="45" state="r"/>
        </tlLogic>
        <tlLogic id="j3" type="static" programID="arterial_90s" offset="0">
            <phase duration="45" state="G"/>
            <phase duration="45" state="r"/>
        </tlLogic>
    </additional>
    """

    cfg_xml = """
    <configuration>
        <input>
            <net-file value="simple_arterial.net.xml"/>
            <route-files value="simple_arterial.rou.xml"/>
            <additional-files value="simple_arterial.tls.xml"/>
        </input>
        <time>
            <begin value="0"/>
            <end value="900"/>
            <step-length value="1.0"/>
        </time>
    </configuration>
    """

    write_text(output_dir / "simple_arterial.nod.xml", node_xml)
    write_text(output_dir / "simple_arterial.edg.xml", edge_xml)
    write_text(output_dir / "simple_arterial.rou.xml", route_xml)
    write_text(output_dir / "simple_arterial.tls.xml", tls_xml)
    write_text(output_dir / "simple_arterial.sumocfg", cfg_xml)


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
        description="Create a three-intersection single-lane SUMO arterial scenario."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run netconvert to create simple_arterial.net.xml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    create_sumo_inputs(output_dir)
    if args.build:
        build_network(output_dir)
    print(f"wrote SUMO arterial inputs to {output_dir}")
    if not args.build:
        print("run again with --build, or run netconvert manually, to create the .net.xml")


if __name__ == "__main__":
    main()

