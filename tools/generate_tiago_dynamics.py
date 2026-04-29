#!/usr/bin/env python3
"""Generate Tiago arm dynamics artifacts from the source URDF.

- GRiD can't parse negative principal joint axes. Workaround: create temporary GRiD-only URDF with flipped axes. Invert respective parameters in the plant wrapper:
    ./gato/dynamics/tiago_right/tiago_right_plant.cuh
- This also writes a native arm-only URDF for Pinocchio, using the same
  extraction path but without GRiD-only axis normalization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy


ARM_CHOICES = ("left", "right")
REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_REQUIREMENTS = {
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "numpy": "numpy",
    "sympy": "sympy",
}
GRID_VENV = REPO_ROOT / ".venv_grid"


@dataclass(frozen=True)
class AxisFlip:
    joint_name: str
    original_axis: str
    grid_axis: str
    original_lower: float | None
    original_upper: float | None
    grid_lower: float | None
    grid_upper: float | None

    def wrapper_note(self) -> str:
        if self.original_lower is None or self.original_upper is None:
            limits = "no joint limits were present"
        else:
            limits = (
                f"limits {self.original_lower:.17g}..{self.original_upper:.17g} "
                f"became {self.grid_lower:.17g}..{self.grid_upper:.17g}"
            )
        return (
            f"{self.joint_name}: {self.original_axis} -> {self.grid_axis}; {limits}. "
            "Wrapper convention: q_grid=-q_tiago, qd_grid=-qd_tiago, "
            "u_grid=-u_tiago, qdd_tiago=-qdd_grid for this joint."
        )


@dataclass(frozen=True)
class JointLimit:
    joint_name: str
    lower: float
    upper: float
    velocity: float
    effort: float


def parse_vector(raw: str) -> list[float]:
    return [float(value) for value in raw.split()]


def format_vector(values: list[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def indent(elem: ET.Element) -> None:
    ET.indent(elem, space="  ")


def ensure_joint_origin(joint: ET.Element) -> None:
    """GRiD's parser expects every joint origin to include xyz and rpy."""

    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.attrib.setdefault("xyz", "0 0 0")
    origin.attrib.setdefault("rpy", "0 0 0")


def normalize_axis_for_grid(joint: ET.Element) -> AxisFlip | None:
    """Flip negative principal axes for GRiD.

    This intentionally changes only the temporary generation URDF. The original
    Tiago convention remains the public convention that the plant wrapper should
    expose to the rest of GATO and to robot-facing code.
    """

    axis = joint.find("axis")
    if axis is None:
        return None

    original_axis = axis.attrib["xyz"]
    axis_values = parse_vector(original_axis)
    non_zero = [idx for idx, value in enumerate(axis_values) if abs(value) > 1e-12]
    if len(non_zero) != 1:
        raise ValueError(
            f"Joint {joint.attrib['name']} has unsupported non-principal axis {original_axis}"
        )

    idx = non_zero[0]
    value = axis_values[idx]
    if abs(abs(value) - 1.0) > 1e-12:
        raise ValueError(
            f"Joint {joint.attrib['name']} has unsupported non-unit axis {original_axis}"
        )

    if value > 0:
        return None

    grid_axis_values = [0.0, 0.0, 0.0]
    grid_axis_values[idx] = 1.0
    grid_axis = format_vector(grid_axis_values)
    axis.attrib["xyz"] = grid_axis

    original_lower = original_upper = grid_lower = grid_upper = None
    limit = joint.find("limit")
    if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
        original_lower = float(limit.attrib["lower"])
        original_upper = float(limit.attrib["upper"])
        grid_lower = -original_upper
        grid_upper = -original_lower
        limit.attrib["lower"] = f"{grid_lower:.17g}"
        limit.attrib["upper"] = f"{grid_upper:.17g}"

    return AxisFlip(
        joint_name=joint.attrib["name"],
        original_axis=original_axis,
        grid_axis=grid_axis,
        original_lower=original_lower,
        original_upper=original_upper,
        grid_lower=grid_lower,
        grid_upper=grid_upper,
    )


def build_arm_urdf(input_path: Path, arm: str, output_path: Path, normalize_axes_for_grid: bool) -> list[AxisFlip]:
    tree = ET.parse(input_path)
    root = tree.getroot()

    link_by_name: dict[str, ET.Element] = {}
    joint_by_name: dict[str, ET.Element] = {}
    for child in root:
        name = child.attrib.get("name")
        if child.tag == "link" and name:
            link_by_name[name] = child
        elif child.tag == "joint" and name:
            joint_by_name[name] = child

    root_link = "torso_lift_link"
    actuated_joint_names = [f"arm_{arm}_{idx}_joint" for idx in range(1, 8)]
    tool_joint = f"arm_{arm}_tool_joint"
    all_joint_names = [*actuated_joint_names, tool_joint]

    missing_joints = [name for name in all_joint_names if name not in joint_by_name]
    if missing_joints:
        raise ValueError(f"Missing expected Tiago {arm} arm joints: {missing_joints}")

    selected_links = {root_link}
    for joint_name in all_joint_names:
        joint = joint_by_name[joint_name]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint_name} is missing a parent or child tag")
        selected_links.add(parent.attrib["link"])
        selected_links.add(child.attrib["link"])

    missing_links = sorted(name for name in selected_links if name not in link_by_name)
    if missing_links:
        raise ValueError(f"Missing expected Tiago {arm} arm links: {missing_links}")

    out_root = ET.Element(
        "robot",
        {
            "name": f"tiago_pro_{arm}_arm{'_grid' if normalize_axes_for_grid else ''}",
            "xmlns:xacro": "http://www.ros.org/wiki/xacro",
        },
    )

    out_root.append(deepcopy(link_by_name[root_link]))
    for idx in range(1, 8):
        out_root.append(deepcopy(link_by_name[f"arm_{arm}_{idx}_link"]))
    out_root.append(deepcopy(link_by_name[f"arm_{arm}_tool_link"]))

    flips: list[AxisFlip] = []
    for joint_name in all_joint_names:
        joint_copy = deepcopy(joint_by_name[joint_name])
        ensure_joint_origin(joint_copy)
        if normalize_axes_for_grid:
            flip = normalize_axis_for_grid(joint_copy)
            if flip is not None:
                flips.append(flip)
        out_root.append(joint_copy)

    indent(out_root)
    ET.ElementTree(out_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return flips


def build_grid_urdf(input_path: Path, arm: str, output_path: Path) -> list[AxisFlip]:
    return build_arm_urdf(input_path, arm, output_path, normalize_axes_for_grid=True)


def extract_arm_limits(input_path: Path, arm: str) -> list[JointLimit]:
    tree = ET.parse(input_path)
    root = tree.getroot()

    joint_by_name = {
        child.attrib["name"]: child
        for child in root
        if child.tag == "joint" and "name" in child.attrib
    }
    joint_names = [f"arm_{arm}_{idx}_joint" for idx in range(1, 8)]

    limits: list[JointLimit] = []
    for joint_name in joint_names:
        joint = joint_by_name.get(joint_name)
        if joint is None:
            raise ValueError(f"Missing expected Tiago {arm} arm joint: {joint_name}")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Joint {joint_name} is missing a limit tag")

        missing = [attr for attr in ("lower", "upper", "velocity", "effort") if attr not in limit.attrib]
        if missing:
            raise ValueError(f"Joint {joint_name} limit is missing attributes: {missing}")

        limits.append(
            JointLimit(
                joint_name=joint_name,
                lower=float(limit.attrib["lower"]),
                upper=float(limit.attrib["upper"]),
                velocity=float(limit.attrib["velocity"]),
                effort=float(limit.attrib["effort"]),
            )
        )

    return limits


def format_cpp_float(value: float) -> str:
    return f"{value:.17g}"


def format_limit_row(lower: float, upper: float, joint_name: str) -> str:
    return (
        "            {static_cast<T>("
        f"{format_cpp_float(lower)}), static_cast<T>({format_cpp_float(upper)})"
        f"}},  // {joint_name}"
    )


def write_limits_header(output_path: Path, arm: str, limits: list[JointLimit]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joint_rows = "\n".join(
        format_limit_row(limit.lower, limit.upper, limit.joint_name)
        for limit in limits
    )
    velocity_rows = "\n".join(
        format_limit_row(-limit.velocity, limit.velocity, limit.joint_name)
        for limit in limits
    )
    effort_rows = "\n".join(
        format_limit_row(-limit.effort, limit.effort, limit.joint_name)
        for limit in limits
    )

    text = f"""#pragma once

// Generated by tools/generate_tiago_dynamics.py from the native Tiago URDF.
// These limits are the raw URDF limits for the {arm} arm. Safety margins belong
// in solver policy/tuning code, not in this source-of-truth robot data.

namespace gato {{
namespace plant {{

        constexpr int TIAGO_LIMIT_JOINTS = {len(limits)};

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[TIAGO_LIMIT_JOINTS][2] = {{
{joint_rows}
        }};

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[TIAGO_LIMIT_JOINTS][2] = {{
{velocity_rows}
        }};

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[TIAGO_LIMIT_JOINTS][2] = {{
{effort_rows}
        }};

        template<class T>
        __host__ __device__ constexpr const T (&JOINT_LIMITS())[TIAGO_LIMIT_JOINTS][2]
        {{
                return JOINT_LIMITS_DATA<T>;
        }}

        template<class T>
        __host__ __device__ constexpr const T (&VEL_LIMITS())[TIAGO_LIMIT_JOINTS][2]
        {{
                return VEL_LIMITS_DATA<T>;
        }}

        template<class T>
        __host__ __device__ constexpr const T (&CTRL_LIMITS())[TIAGO_LIMIT_JOINTS][2]
        {{
                return CTRL_LIMITS_DATA<T>;
        }}

}}  // namespace plant
}}  // namespace gato
"""
    output_path.write_text(text, encoding="utf-8")


def require_grid_checkout(grid_dir: Path) -> None:
    expected = [
        grid_dir / "generateGRiD.py",
        grid_dir / "URDFParser" / "URDFParser.py",
        grid_dir / "GRiDCodeGenerator" / "GRiDCodeGenerator.py",
        grid_dir / "RBDReference" / "RBDReference.py",
    ]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise SystemExit(
        "GRiD checkout is incomplete. Missing:\n"
        f"{missing_text}\n\n"
        "Initialize it with:\n"
        "  git -c url.https://github.com/.insteadOf=git@github.com: "
        "submodule update --init --recursive GRiD"
    )


def require_grid_python_deps() -> None:
    missing = [
        package_name
        for package_name, module_name in GRID_REQUIREMENTS.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return

    raise SystemExit(
        "Missing Python packages required by GRiD: "
        + ", ".join(missing)
        + "\nInstall them with:\n"
        + "  python -m pip install -r GRiD/requirements.txt\n"
        + "or run this generator with the repo-local GRiD venv:\n"
        + f"  {GRID_VENV / 'bin' / 'python'} tools/generate_tiago_dynamics.py"
    )


def run_grid(
    grid_dir: Path,
    urdf_path: Path,
    namespace: str,
    debug: bool,
    work_dir: Path,
    fixed_target_name: str,
) -> Path:
    command = [
        sys.executable,
        str(grid_dir / "generateGRiD.py"),
        str(urdf_path),
        "--namespace",
        namespace,
    ]
    if fixed_target_name:
        command.extend(["--fixed-target-names", fixed_target_name])
    if debug:
        command.append("--debug")

    subprocess.run(command, cwd=work_dir, check=True)

    generated = work_dir / f"{namespace}.cuh"
    if not generated.exists():
        raise FileNotFoundError(f"GRiD completed but did not create {generated}")
    return generated


def default_output_for_arm(arm: str) -> Path:
    return REPO_ROOT / "gato" / "dynamics" / f"tiago_{arm}" / f"tiago_{arm}_grid.cuh"


def default_limits_output_for_arm(arm: str) -> Path:
    return REPO_ROOT / "gato" / "dynamics" / f"tiago_{arm}" / f"tiago_{arm}_limits.cuh"


def default_arm_urdf_output_for_arm(arm: str) -> Path:
    return REPO_ROOT / "gato" / "dynamics" / f"tiago_{arm}" / f"tiago_{arm}_arm.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "TiagoProURDF" / "tiago_pro.urdf",
        help="Path to the full Tiago Pro URDF.",
    )
    parser.add_argument(
        "--arm",
        choices=ARM_CHOICES,
        default="right",
        help="Which Tiago arm to generate.",
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=REPO_ROOT / "GRiD",
        help="Path to the A2R-Lab GRiD checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Generated header path. Defaults to gato/dynamics/tiago_<arm>/tiago_<arm>_grid.cuh.",
    )
    parser.add_argument(
        "--limits-output",
        type=Path,
        default=None,
        help="Generated raw-limit header path. Defaults to gato/dynamics/tiago_<arm>/tiago_<arm>_limits.cuh.",
    )
    parser.add_argument(
        "--arm-urdf-output",
        type=Path,
        default=None,
        help="Generated native arm-only URDF path. Defaults to gato/dynamics/tiago_<arm>/tiago_<arm>_arm.urdf.",
    )
    parser.add_argument(
        "--namespace",
        default="grid",
        help="C++ namespace and temporary header basename passed to GRiD.",
    )
    parser.add_argument(
        "--debug-grid",
        action="store_true",
        help="Pass GRiD's -D debug flag.",
    )
    parser.add_argument(
        "--keep-temp",
        type=Path,
        default=None,
        help="Debug option: keep temporary generation files in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    grid_dir = args.grid_dir.resolve()
    output_path = (args.output.resolve() if args.output is not None else default_output_for_arm(args.arm))
    limits_output_path = (
        args.limits_output.resolve()
        if args.limits_output is not None
        else default_limits_output_for_arm(args.arm)
    )
    arm_urdf_output_path = (
        args.arm_urdf_output.resolve()
        if args.arm_urdf_output is not None
        else default_arm_urdf_output_for_arm(args.arm)
    )

    if not input_path.exists():
        raise SystemExit(f"Input URDF does not exist: {input_path}")

    require_grid_checkout(grid_dir)
    require_grid_python_deps()

    if args.keep_temp is None:
        temp_context = tempfile.TemporaryDirectory(prefix="gato_tiago_grid_")
        work_dir_path = Path(temp_context.name)
    else:
        args.keep_temp.mkdir(parents=True, exist_ok=True)
        temp_context = None
        work_dir_path = args.keep_temp.resolve()

    try:
        grid_urdf = work_dir_path / f"tiago_{args.arm}_arm_grid.urdf"
        flips = build_grid_urdf(input_path, args.arm, grid_urdf)

        generated = run_grid(
            grid_dir,
            grid_urdf,
            args.namespace,
            args.debug_grid,
            work_dir_path,
            fixed_target_name=f"arm_{args.arm}_tool_joint",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, output_path)
        build_arm_urdf(input_path, args.arm, arm_urdf_output_path, normalize_axes_for_grid=False)
        write_limits_header(limits_output_path, args.arm, extract_arm_limits(input_path, args.arm))

        print(f"Wrote {output_path}")
        print(f"Wrote {arm_urdf_output_path}")
        print(f"Wrote {limits_output_path}")
        if args.keep_temp is None:
            print("Temporary GRiD URDF was removed after generation.")
        else:
            print(f"Temporary GRiD URDF: {grid_urdf}")
        if flips:
            print("\nWrapper sign convention required for flipped axes:")
            for flip in flips:
                print(f"  - {flip.wrapper_note()}")
        else:
            print("\nNo negative principal axes were flipped.")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
