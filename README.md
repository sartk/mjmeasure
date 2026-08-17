# mjmeasure

`mjmeasure` is a browser-based viewer for inspecting static MuJoCo scenes. It
supports point-to-point and plane-to-point measurements, named-camera
visualization, camera renders, and wrist-camera placement.

## Install

The project uses Python 3.10 or newer. With
[uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:abc-hands/mjmeasure.git
cd mjmeasure
uv sync
```

## Use

Open a MuJoCo XML scene:

```bash
uv run mjmeasure path/to/scene.xml
```

Then visit [http://localhost:8080](http://localhost:8080). Select a
measurement mode in the viewer and click rendered surfaces.

Useful options:

```bash
# Do not draw the z=0 reference grid.
uv run mjmeasure scene.xml --no-grid

# Draw origins and optical-axis rays for named cameras.
uv run mjmeasure scene.xml --cam-rays top left right

# Add still renders for the named cameras to the GUI.
uv run mjmeasure scene.xml --cam-rays top left right --render-cams

# Start the specialized mirrored wrist-camera placement workflow.
uv run mjmeasure scene.xml --tune-wrist-cam

# Listen on a different interface or port.
uv run mjmeasure scene.xml --host 127.0.0.1 --port 9090
```

Run `uv run mjmeasure --help` for the complete command reference.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```
