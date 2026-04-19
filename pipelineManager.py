from pathlib import Path
import subprocess
import sys


STAGES = {
    "gear": {
        "script": "gearEstimater.py",
        "input_from": "smoothed",
        "output_suffix": "_gears.csv",
    },
    "fan": {
        "script": "fanSpeedEstimater.py",
        "input_from": "smoothed",
        "output_suffix": "_fan.csv",
    },
}


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def ensure_script_exists(script_name: str) -> None:
    script_path = Path(script_name)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_name}")


def run_python_script(script_name: str, input_file: Path, extra_args: list[str] | None = None) -> None:
    ensure_script_exists(script_name)

    cmd = ["python", script_name, str(input_file)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_stage_list(text: str | None) -> list[str]:
    if not text:
        return []

    stages = [item.strip().lower() for item in text.split(",")]
    stages = [item for item in stages if item]

    invalid = [item for item in stages if item not in STAGES]
    if invalid:
        raise ValueError(f"Unknown stage(s): {', '.join(invalid)}")

    return stages


def build_pipeline_paths(raw_input: Path) -> dict[str, Path]:
    clean_path = raw_input.with_name(f"{raw_input.stem}_clean.csv")
    smoothed_path = raw_input.with_name(f"{raw_input.stem}_clean_smoothed.csv")

    return {
        "raw": raw_input,
        "clean": clean_path,
        "smoothed": smoothed_path,
    }


def build_stage_output(smoothed_file: Path, suffix: str) -> Path:
    return smoothed_file.with_name(f"{smoothed_file.stem}{suffix}")


def run_pipeline(input_file: str, selected_stages: list[str], run_plots: bool) -> None:
    raw_input = Path(input_file)
    ensure_file_exists(raw_input)

    outputs = build_pipeline_paths(raw_input)

    run_python_script("dataHandlerV2.py", outputs["raw"])
    ensure_file_exists(outputs["clean"])

    run_python_script("dataSmoother.py", outputs["clean"])
    ensure_file_exists(outputs["smoothed"])

    for stage_name in selected_stages:
        stage_cfg = STAGES[stage_name]

        if stage_cfg["input_from"] == "smoothed":
            stage_input = outputs["smoothed"]
        else:
            raise ValueError(f"Unsupported input source for stage: {stage_name}")

        run_python_script(stage_cfg["script"], stage_input)

        stage_output = build_stage_output(outputs["smoothed"], stage_cfg["output_suffix"])
        outputs[stage_name] = stage_output

    if run_plots:
        ensure_script_exists("plotResults.py")

        plot_inputs = [str(outputs["raw"]), str(outputs["smoothed"])]

        for stage_name in selected_stages:
            if stage_name in outputs:
                plot_inputs.append(str(outputs[stage_name]))

        cmd = ["python", "plotResults.py", *plot_inputs]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\nPipeline done")
    for name, path in outputs.items():
        print(f"{name}: {path.name}")


def print_usage() -> None:
    print(
        "\nUsage:\n"
        "python runPipeline.py <input_file> [stages] [--plot]\n\n"
        "Examples:\n"
        "python runPipeline.py NewDrive.csv\n"
        "python runPipeline.py NewDrive.xlsx gear\n"
        "python runPipeline.py NewDrive.csv fan\n"
        "python runPipeline.py NewDrive.csv gear,fan --plot\n"
    )


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print_usage()
        raise SystemExit(1)

    input_file = argv[1]
    stages_arg = None
    run_plots = False

    for arg in argv[2:]:
        if arg == "--plot":
            run_plots = True
        else:
            stages_arg = arg

    selected_stages = parse_stage_list(stages_arg)
    run_pipeline(input_file=input_file, selected_stages=selected_stages, run_plots=run_plots)


if __name__ == "__main__":
    main(sys.argv)