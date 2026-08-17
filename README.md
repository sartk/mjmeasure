# mjmeasure

Browser-based point-to-point and plane-to-point measurements for MuJoCo scenes.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/sartk/mjmeasure.git
```

## Run

```console
mjmeasure path/to/scene.xml
```

Open [http://localhost:8080](http://localhost:8080), choose a measurement
mode, and click rendered surfaces.

Useful options:

```bash
mjmeasure scene.xml --no-grid
mjmeasure scene.xml --cam-rays top left right --render-cams
mjmeasure scene.xml --tune-wrist-cam
```

Run `mjmeasure --help` for all options.
